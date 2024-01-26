#will make ALIP class 
#two dynamics, one with internal x_i,
#other without, if can not apply the internal constraints then it makes no sense
import torch
from torch import nn
import os
import numpy as np

import shutil
FFMPEG_BIN = shutil.which('ffmpeg')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

#TODO: first iteration, check if i have the correct x_0 or i need to change anyhting
#if i have to change x_0, remember first dt is Tr
#TODO: setter function for variables
class ALIP_error_model(nn.Module):  #maybe two models one for each initial stance_leg 
    def __init__(self, Ns, Nt, indata, n_batch):
        super().__init__()

        self.n_state = 4
        self.n_ctrl = 2
        self.Ts = indata.Ts
        self.Tr = indata.Tr
        self.Nt = Nt
        self.Ns = Ns
        self.dt = self.Ts/Nt
        self.mass = torch.tensor(indata.mass)
        self.zH = torch.tensor(indata.zH)
        self.g = torch.tensor(9.81)
        self.w = indata.w
        self.initial_stance_leg = indata.stance_leg
        self.stance_sign = indata.stance_leg
        self.n_batch = n_batch


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

        #constant desired state
        self.Lx_offset = indata.Lx_offset
        self.l = torch.sqrt(self.g/self.zH)
        self.Ly_des = indata.Ly_des
        self.x_des = 1/self.mass/self.zH/self.l * torch.tanh(self.l*self.Ts/2) * self.Ly_des
        #leg dependent desired state
        self.y_des_plus = -self.w/2
        self.y_des_minus = self.w/2
        self.Lx_des_plus = 0.5*self.mass*self.zH*self.l*self.w*torch.sqrt(self.l*self.Ts*0.5) + self.Lx_offset
        self.Lx_des_minus = -0.5*self.mass*self.zH*self.l*self.w*torch.sqrt(self.l*self.Ts*0.5) + self.Lx_offset
 
    def forward(self, state, u): #same as error_dyn
        #TODO: repeat at the end doens't seem to be needed, same behaviour without  
        #
        #ctrl dyn:state = torch.matmul(state + torch.matmul(u, self.B), self.exp_At)
     
        if (self.stance_sign == 1):
            state = torch.matmul(state + torch.matmul(u, self.B), self.exp_At) - torch.tensor([self.x_des, self.y_des_plus, self.Lx_des_plus, self.Ly_des]).repeat(self.n_batch, 1)
        else:
            state = torch.matmul(state + torch.matmul(u, self.B), self.exp_At) - torch.tensor([self.x_des, self.y_des_minus, self.Lx_des_minus, self.Ly_des]).repeat(self.n_batch, 1)

        self.stance_sign = -self.stance_sign;
        return state

    #another problem with this is that it will also try to find optimal soluions for u inside the step
    #even if the cost was 0, which we still have to see
    #we have to think if we take x_0 like in the papper
    def full_dyn(self, state, u): #Nt must be > 1 // how the optimization is going to perform over discrete t
        #state: [x, y, Lx, Ly, t]
        #u: [u_fp_x, u_fp_y]
        it = state[4]
        s_state = state[0:4]

        if(it % self.Nt == 0):
            if(it == 0): #really we will start it with it = 1
                d = 1
            else:
                s_state = torch.linalg.matrix_exp(self.A*self.dt)*(s_state + self.B*u)
        else:
            s_state = torch.linalg.matrix_exp(self.A*self.dt)*s_state
            
        state[0:4] = s_state
        state[4] += 1

        return state

    def ctrl_dyn(self, state, u): 
        #state: [nbatch,[x, y, Lx, Ly]]
        #u: [nbatch, [u_fp_x, u_fp_y]]
        state = torch.matmul(state + torch.matmul(u, self.B), self.exp_At)
        return state

    """TODO: still need to check for dimension with n_batch"""



    def get_frame(self, state):
        assert len(state) == 4
        x, y, Lx, Ly = torch.unbind(state)

        x = x.numpy()
        y = y.numpy()

        fig, ax = plt.subplots(figsize=(6,6))


        ax.scatter(x,y,color='k', marker = 'x', s=50)
        ax.set_xlim((-3, 3))
        ax.set_ylim((-3, 3))
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')   
        return fig, ax
    

    
    def update_stance_leg(self,stl):
        self.initial_stance_leg = stl
        self.stance_sign = stl





        




    

"""
  step_horizon: "4"
  Ts: 0.25    #0.25  #Greater Ts makes offset increase also
  stance_leg: -1 # -1: RIGHT SWING TO START; 1: LEFT SWING TO START
  leg_width: 0.13 #0.1
  zH: 0.70   #this will be the reference height target. The height for the mpc will be another
  mu: 0.3
  total_mass: 39.15342    
  ufp_x_max: 0.6  #0.6
  ufp_y_max: 0.4  #0.4
  ufp_y_min: 0.1 #0.05
  Lx_offset: 0  #0
  Ly_des:  30 #3
  com_yaw: 0   #degrees
  kx: 0
  ky: 0
"""


class indata():
    def __init__(self):
        self.Ts = 0.25
        self.Tr = 0.2
        self.mass = 39.15342
        self.zH = 0.7

        self.g = 9.81
        self.w = 0.13
        self.stance_leg = 1
        self.Lx_offset = 0
        self.Ly_des = 1
    

   
        

if __name__ == '__main__':
    _indata = indata()
    n_batch, T = 6, 50
    n_state = 4
    n_ctrl = 2
    dx = ALIP_error_model(4, 4, _indata, n_batch)
    u = torch.zeros(T, n_batch, n_ctrl)
    xinit_ctrl = torch.zeros(n_batch, n_state)
    #xinit_full = torch.zeros(n_batch, dx.n_state +1)

    xinit_ctrl[:,0] = 0.005
    xinit_ctrl[:,1] = 0.001

    x = xinit_ctrl
    print("main")

    for t in range(T):
        x = dx(x, u[t])
        
        fig, ax = dx.get_frame(x[0])
        fig.savefig('{:03d}.png'.format(t))
        plt.close(fig)

    vid_file = 'alip_vid.mp4'
    if os.path.exists(vid_file):
        os.remove(vid_file)
    cmd = ('{} -loglevel quiet '
            '-r 32 -f image2 -i %03d.png -vcodec '
            'libx264 -crf 25 -pix_fmt yuv420p {}').format(
        FFMPEG_BIN,
        vid_file
    )
    os.system(cmd)
    for t in range(T):
        os.remove('{:03d}.png'.format(t))







        
    


