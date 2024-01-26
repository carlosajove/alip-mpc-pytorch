import torch
import os

from mpc import mpc
from mpc.mpc import QuadCost, LinDx, GradMethods

from ALIP_model import ALIP_error_model


import shutil
FFMPEG_BIN = shutil.which('ffmpeg')

#TODO: setter function for variables
#TODO: x boundaries 
class ALIP_mpc_torch():
    def __init__(self, Ns, Nt, indata, n_batch):        
        self.ALIP_mod = ALIP_error_model(Ns, Nt, indata, n_batch)
        self.n_state = self.ALIP_mod.n_state
        self.n_ctrl = self.ALIP_mod.n_ctrl
        self.n_batch = n_batch
        self.Ts = indata.Ts
        self.Tr = indata.Tr
        self.Nt = Nt
        self.Ns = Ns
        self.dt = self.Ts/Nt
        self.initial_stance_leg = indata.stance_leg
        self.stance_sign = indata.stance_leg
        self.u_init = None
        self.mass = torch.tensor(indata.mass)
        self.zH = torch.tensor(indata.zH)
        self.g = torch.tensor(9.81)


        self.Qrunning = torch.eye(self.n_state + self.n_ctrl)
        self.prunning = torch.zeros(self.n_state + self.n_ctrl)
        self.Qterminal = 100*torch.eye(self.n_state + self.n_ctrl)     
        self.pterminal = torch.zeros(self.n_state + self.n_ctrl)
        for i in range(self.n_state, self.n_state + self.n_ctrl): #i = 4, i =5
            self.Qrunning[i,i] = 0
            self.Qterminal[i,i] = 0
        self.getCost()

    
        #TODO: getter function from param data
        self.ufp_x_max = 0.6
        self.ufp_y_max = 0.4
        self.ufp_y_min = 0.1
        self.get_u_bounds()


    def solve(self, x, indata):
        print("state in solve", x)
        self.ALIP_mod.update_stance_leg(indata.stance_leg)

        if(indata.stance_leg == 1): #set bounds
            self.u_lower = self.u_lower_plus
            self.u_upper = self.u_upper_plus
        else:      
            self.u_lower = self.u_lower_plus
            self.u_upper = self.u_upper_plus

        nominal_states, nominal_actions, nominal_objs = mpc.MPC(
                self.n_state, self.n_ctrl, self.Ns,
                u_init= self.u_init,
                u_lower= self.u_lower, u_upper= self.u_upper,
                lqr_iter=50,
                verbose=1,
                exit_unconverged=False,
                detach_unconverged=False,
                grad_method=GradMethods.AUTO_DIFF,
                eps=1e-2,
            )(x, QuadCost(self.Q, self.p), self.ALIP_mod)
        self.up_u_init(nominal_actions)
        return nominal_states, nominal_actions, nominal_objs
    
    def get_u_bounds(self): #-1 for LS solver
        assert self.Ns%2 == 0 #Ns need to be even in order to work with the current implementation of the u_bounds

        self.u_upper_plus = torch.tensor([[ self.ufp_x_max/2, self.ufp_y_max], 
                                     [ self.ufp_x_max/2, -self.ufp_y_min]])
        self.u_lower_plus = torch.tensor([[-self.ufp_x_max/2, self.ufp_y_min],
                                     [-self.ufp_x_max/2, -self.ufp_y_max]])

        self.u_upper_minus = torch.tensor([[self.ufp_x_max/2, -self.ufp_y_min], 
                                      [self.ufp_x_max/2, self.ufp_y_max]])
        self.u_lower_minus = torch.tensor([[-self.ufp_x_max/2,- self.ufp_y_max], 
                                      [-self.ufp_x_max/2, self.ufp_y_min]])

        self.u_upper_plus = self.u_upper_plus.repeat(int(self.Ns/2), 1)
        self.u_upper_plus = self.u_upper_plus.unsqueeze(1).repeat(1,self.n_batch,1)

        self.u_lower_plus = self.u_lower_plus.repeat(int(self.Ns/2), 1)
        self.u_lower_plus = self.u_lower_plus.unsqueeze(1).repeat(1,self.n_batch,1)
    
        self.u_upper_minus = self.u_upper_minus.repeat(int(self.Ns/2), 1)
        self.u_upper_minus = self.u_upper_minus.unsqueeze(1).repeat(1,self.n_batch,1)

        self.u_lower_minus = self.u_lower_minus.repeat(int(self.Ns/2), 1)
        self.u_lower_minus = self.u_lower_minus.unsqueeze(1).repeat(1,self.n_batch,1)

    def up_u_init(self, u): #TODO: try placing the first column at the end, so the foot sols switch, since next solver will 
                            #since next solver will be with the next stance foot
        self.u_init = u            

    def getCost(self):
        h = self.Qrunning.unsqueeze(0).unsqueeze(0).repeat(self.Ns-1, self.n_batch, 1, 1)
        Qt = self.Qterminal.repeat(1, self.n_batch, 1, 1)
        self.Q = torch.cat((h, Qt), 0)
        #assumes self.qrunning == self.qterminal TODO: do same thing as with Q with terminal different
        self.p = self.prunning.unsqueeze(0).repeat(self.Ns, n_batch, 1)

    def getLinDin(self):
        self.A = torch.tensor([[0 ,0,0, 1/self.mass/self.zH],
                              [0,0,-1/self.mass/self.zH,0],
                              [0,-self.mass*self.g,0,0],
                              [self.mass*self.g,0,0,0]])
        self.A = self.A.t()
        self.B = torch.tensor([[-1, 0],
                              [ 0,-1],
                              [ 0, 0],
                              [ 0, 0]], dtype=torch.float32)
        self.B = self.B.t()
        self.exp_At = torch.linalg.matrix_exp(self.A*self.dt)    

        



class indata():
    def __init__(self):
        self.Ts = 0.25
        self.Tr = 0.2
        self.mass = 39.15342
        self.zH = 0.7

        self.state = torch.tensor([1,2,3,4])

        self.g = 9.81
        self.w = 0.13
        self.stance_leg = 1
        self.Lx_offset = 0
        self.Ly_des = 1

def uniform(shape, low, high):
    r = high-low
    return torch.rand(shape)*r+low
    

if __name__ == '__main__':
    _indata = indata()
    n_batch, Ns = 3, 4   #Ns = T in mpc formulation

    n_state = 4
    n_ctrl = 2
    a = ALIP_mpc_torch(Ns, 4, _indata, n_batch)
    x = uniform(n_batch, -0.1, 0.1)
    y = uniform(n_batch, -0.1, 0.1)
    Lx = uniform(n_batch, -1, 1)
    Ly = uniform(n_batch, -1, 1)

    x = torch.stack((x,y,Lx,Ly), dim = 1)
    Cost = QuadCost(a.Q, a.p)
    nom_state, nom_u, nom_obj = a.solve(x, _indata)
    print("initial", x)
    print("finished ")
    print("state", nom_state)
    print("nom_ac", nom_u)
    print("obj", nom_obj)




    
