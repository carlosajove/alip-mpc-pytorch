import torch
import os
import sys
import math
#import numpy as np
 
# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
 
# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
 
# adding the parent directory to 
# the sys.path.
sys.path.append(parent)
 

from mpc import mpc
from mpc.mpc import QuadCost, LinDx, GradMethods

from ALIP_model import ALIP_error_model


import shutil
FFMPEG_BIN = shutil.which('ffmpeg')

import shutil
FFMPEG_BIN = shutil.which('ffmpeg')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

#TODO: Curent hypothesis: not working because Cost has not full rank
#                         current solution is consider eps instead of 0
#TODO: setter function for variables
#TODO: x boundaries 
#TODO: x_init has to be in error dynamics, add the first state separately
#TODO: find out why last action is alwa7s [0, 0.1]:
#      Hypothesis: it has to do with x_init in pnqp. 
#TODO: check why doesn't match Ly des with actual state
class ALIP_mpc_torch_LinDx():
    def __init__(self, Ns, Nt, indata, n_batch):     
        self.eps = 1e-4 
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
        self.w = indata.w
        
        self.g = torch.tensor(9.81)

        #desired state
        self.Lx_offset = indata.Lx_offset
        self.Ly_des = indata.Ly_des



        self.getLinDinF()

        self.getCost(True) #if True use eps instead of 0 

    
        #TODO: getter function from param data
        self.ufp_x_max = 0.6
        self.ufp_y_max = 0.4
        self.ufp_y_min = 0.1
        self.get_u_bounds()


    def solve(self, x, indata):
        self.ALIP_mod.update_stance_leg(indata.stance_leg)

        if(indata.stance_leg == 1): #set bounds TODO:CHECK f is the good one
            self.u_lower = self.u_lower_plus
            self.u_upper = self.u_upper_plus
            self.q = self.q_plus
        else:      
            self.u_lower = self.u_lower_minus
            self.u_upper = self.u_upper_minus
            self.q = self.q_minus
        

        nominal_states, nominal_actions, nominal_objs = mpc.MPC(
                self.n_state, self.n_ctrl, self.Ns,
                u_init= self.u_init,
                u_lower= self.u_lower, u_upper= self.u_upper,
                lqr_iter=50,
                verbose=-1,
                exit_unconverged=False,
                detach_unconverged=False,
                grad_method=GradMethods.ANALYTIC,
                eps=1e-2,
            )(x, QuadCost(self.Q, self.q), LinDx(self.F))
        self.up_u_init(nominal_actions)
        return nominal_states, nominal_actions, nominal_objs
    

    #TODO: Control bounds don't seem to work properly
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

        #self.u_lower_minus = torch.randn(self.Ns, self.n_batch, self.n_ctrl)
        #self.u_lower_plus = torch.randn(self.Ns, self.n_batch, self.n_ctrl)
        #self.u_upper_plus = torch.randn(self.Ns, self.n_batch, self.n_ctrl)
        #self.u_upper_minus = torch.randn(self.Ns, self.n_batch, self.n_ctrl)


    def up_u_init(self, u): #TODO: try placing the first column at the end, so the foot sols switch, since next solver will 
                            #since next solver will be with the next stance foot
        self.u_init = u            

    def getCost(self, bool_eps):
        self.Qrunning = 2*torch.eye(self.n_state + self.n_ctrl)
        self.Qterminal = 2*100*torch.eye(self.n_state + self.n_ctrl)
        for i in range(self.n_state, self.n_state + self.n_ctrl): #i = 4, i =5
            if bool_eps:
                self.Qrunning[i,i] = self.eps
                self.Qterminal[i,i] = self.eps 
            else:
                self.Qrunning[i,i] = 0
                self.Qterminal[i,i] = 0     

        h = self.Qrunning.unsqueeze(0).unsqueeze(0).repeat(self.Ns-1, self.n_batch, 1, 1)
        Qt = self.Qterminal.repeat(1, self.n_batch, 1, 1)
        self.Q = torch.cat((h, Qt), 0)


        #desired state
        self.l = torch.sqrt(self.g/self.zH)
        q1 = -2/self.mass/self.zH/self.l * torch.tanh(self.l*self.Ts/2) * self.Ly_des
        #leg dependent desired state
        q2_plus = self.w
        q2_minus = -self.w
        q3_plus = -self.mass*self.zH*self.l*self.w*torch.sqrt(self.l*self.Ts*0.5) - 2*self.Lx_offset
        q3_minus = self.mass*self.zH*self.l*self.w*torch.sqrt(self.l*self.Ts*0.5) - 2*self.Lx_offset
        q4 = -2*self.Ly_des

        self.q_plus = torch.zeros(self.Ns, self.n_batch, self.n_state+self.n_ctrl)  #initial right stance /left swing
        self.q_minus = torch.zeros(self.Ns, self.n_batch, self.n_state+self.n_ctrl) #initial left stance /right swing

        self.q_plus[:,:,0] = self.q_minus[:,:,0] = q1
        self.q_plus[:,:,3] = self.q_minus[:,:,3] = q4

        for n in range(self.Ns-1):
            if n%2 == 0:
                self.q_plus[n, :, 1] = q2_plus
                self.q_minus[n, : ,1] = q2_minus
                self.q_plus[n, :, 2] = q3_plus
                self.q_minus[n, : ,2] = q3_minus
            else:
                self.q_plus[n, :, 1] = q2_minus
                self.q_minus[n, : ,1] = q2_plus
                self.q_plus[n, :, 2] = q3_plus
                self.q_minus[n, : ,2] = q3_minus

        self.q_plus[self.Ns-1,:,0] = 100*q1
        self.q_minus[self.Ns-1,:,0] = 100*q1

        self.q_plus[self.Ns-1,:,1] = 100*self.q_minus[self.Ns-2,:,1]
        self.q_minus[self.Ns-1,:,1] = 100*self.q_plus[self.Ns-2,:,1]

        self.q_plus[self.Ns-1,:,2] = 100*self.q_minus[self.Ns-2,:,2]
        self.q_minus[self.Ns-1,:,2] = 100*self.q_plus[self.Ns-2,:,2]

        self.q_plus[self.Ns-1,:,3] = 100*q4
        self.q_minus[self.Ns-1,:,3] = 100*q4


    def getLinDinF(self):  #[xi+1] = self.F [xi, ui]         
        A = torch.tensor([[0 ,0,0, 1/self.mass/self.zH],
                          [0,0,-1/self.mass/self.zH,0],
                          [0,-self.mass*self.g,0,0],
                          [self.mass*self.g,0,0,0]])

        B = torch.tensor([[-1, 0],
                          [ 0,-1],
                          [ 0, 0],
                          [ 0, 0]], dtype=torch.float32)


        exp_At = torch.linalg.matrix_exp(A*self.dt) 
        AtB = torch.matmul(exp_At, B)

        self.F = torch.cat((exp_At, AtB), dim = 1)  
        self.F = self.F.repeat(self.Ns, self.n_batch, 1,1) 

    def get_frame(self, state):
        assert len(state) == 6
        x, y, Lx, Ly , ufp, ufp2= torch.unbind(state)
        x = x.numpy()
        y = y.numpy()

        fig, ax = plt.subplots(figsize=(6,6))


        ax.scatter(x,y,color='k', marker = 'x', s=50)
        ax.set_xlim((-3, 3))
        ax.set_ylim((-3, 3))
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')   
        return fig, ax

    def plot_mpc_traj(self, state, u):
        fig = plt.subplots
        x, y, Lx, Ly = torch.unbind(state, dim = 2)
        ux, uy = torch.unbind(u, dim = 2)

        n_row = int(math.sqrt(self.n_batch))
        n_col = n_row+1

        #plot control evolution this is wrong
        ev_ux = torch.cumsum(ux, dim = 0)
        ev_uy = torch.cumsum(uy, dim = 0)

        ev_ux = torch.cat([torch.zeros(1, self.n_batch), ev_ux], dim=0)
        ev_uy = torch.cat([torch.zeros(1, self.n_batch), ev_uy], dim=0)
        h_ev_ux_1 = ev_ux[0:self.Ns:2,:]
        h_ev_ux_2 = ev_ux[1:self.Ns:2,:]
        h_ev_uy_1 = ev_uy[0:self.Ns:2,:]
        h_ev_uy_2 = ev_uy[1:self.Ns:2,:]

        ev_x = x + ev_ux[:-1, :]
        ev_y = y + ev_uy[:-1, :]

        fig, ax = plt.subplots(n_row, n_col, figsize=(20,20), sharex = True, sharey = True)
        for i in range(self.n_batch):
            row = i // n_col
            col = i% n_col
            ax[row,col].plot(h_ev_ux_1[:,i].numpy(), h_ev_uy_1[:,i].numpy(), marker = 'x', color = 'r', label = 'second swing leg')
            ax[row,col].plot(h_ev_ux_2[:,i].numpy(), h_ev_uy_2[:,i].numpy(), marker = 'x', color = 'b', label = 'first swing leg')
            ax[row,col].plot(ev_x[:,i].numpy(), ev_y[:,i].numpy(), marker = 'x', color = 'black', label = 'COM')
            ax[row,col].set_title(f'Batch {i}')
            ax[row, col].set_xlabel('X')
            ax[row, col].set_ylabel('Y')
            ax[row, col].set_aspect('equal')
            ax[row,col].legend()

        fig.suptitle(f'with starting leg {self.initial_stance_leg}') 
        plt.savefig('alip_mpc_pytorch_sol')

        fig, ax = plt.subplots(n_row, n_col, figsize=(20,20), sharex = True, sharey = True)
        Lx_plus = 0.5*self.mass*self.zH*self.l*self.w*torch.sqrt(self.l*self.Ts*0.5) + self.Lx_offset
        Lx_minus = -0.5*self.mass*self.zH*self.l*self.w*torch.sqrt(self.l*self.Ts*0.5) + self.Lx_offset

        for i in range(self.n_batch):
            row = i // n_col
            col = i% n_col
            ax[row,col].plot(range(self.Ns), Lx[:,i].numpy(), marker = 'x', color = 'r', label = 'Lx')
            ax[row,col].plot(range(self.Ns), Ly[:,i].numpy(), marker = 'x', color = 'b', label = 'Ly')
            ax[row,col].plot(range(self.Ns), self.Ly_des*torch.ones(self.Ns).numpy() , color = 'cyan', label = 'Ly_des')
            ax[row,col].plot(range(self.Ns), Lx_plus*torch.ones(self.Ns).numpy(), color = 'orange', label = 'Lx_des')
            ax[row,col].plot(range(self.Ns), Lx_minus*torch.ones(self.Ns).numpy(), color = 'orange')
            ax[row,col].set_title(f'Batch {i}')
            ax[row, col].set_xlabel('X')
            ax[row, col].set_ylabel('Y')
            ax[row, col].set_aspect('equal')
            ax[row,col].legend()


        fig.suptitle(f'Ly_des = {self.Ly_des}; Lx = ')
        plt.savefig('alip_mpc_pytorch_angular')







class indata():
    def __init__(self):
        self.Ts = 0.25
        self.Tr = 0.25
        self.mass = 39.15342
        self.zH = 0.7

        #self.state = torch.tensor([1,2,3,4])

        self.g = 9.81
        self.w = 0.13
        self.stance_leg = -1
        self.Lx_offset = 0.
        self.Ly_des = 0.

def uniform(shape, low, high):
    r = high-low
    return torch.rand(shape)*r+low
    

if __name__ == '__main__':
    _indata = indata()
    n_batch, Ns = 6, 12   #Ns = T in mpc formulation

    n_state = 4
    n_ctrl = 2
    a = ALIP_mpc_torch_LinDx(Ns, 1, _indata, n_batch)
    x = uniform(n_batch, -0.1, 0.1)
    #Lx = uniform(n_batch, -0.005, 0.005)
    #Ly = uniform(n_batch, -0.005, 0.005)
    Lx = torch.zeros(n_batch)
    Ly = torch.zeros(n_batch)
    
    if (_indata.stance_leg == -1): #left stance COMy < 0
        y = uniform(n_batch, -0.1, 0)

    else: #right stance COMy > 0
        y = uniform(n_batch, 0, 0.1)

    """
    x = -0.1*torch.ones(n_batch)
    y = 0*torch.ones(n_batch)
    Lx = 0*torch.ones(n_batch)
    Ly = 0*torch.ones(n_batch)
    """
    x = torch.stack((x,y,Lx,Ly), dim = 1)



 
    nom_state, nom_u, nom_obj = a.solve(x, _indata)
    print(nom_state)

    a.plot_mpc_traj(nom_state, nom_u)











    
