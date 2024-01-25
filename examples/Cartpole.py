import torch

from mpc import mpc
from mpc.mpc import QuadCost, LinDx, GradMethods
from mpc.env_dx import cartpole

import numpy as np
import numpy.random as npr

import matplotlib.pyplot as plt

import os
import io
import base64
import tempfile
from IPython.display import HTML

from tqdm import tqdm


dx = cartpole.CartpoleDx()

n_batch, T, mpc_T = 8, 100, 25

def uniform(shape, low, high):
    r = high-low
    return torch.rand(shape)*r+low

torch.manual_seed(0)
th = uniform(n_batch, -2*np.pi, 2*np.pi)
thdot = uniform(n_batch, -.5, .5)
x = uniform(n_batch, -0.5, 0.5)
xdot = uniform(n_batch, -0.5, 0.5)
xinit = torch.stack((x, xdot, torch.cos(th), torch.sin(th), thdot), dim=1)

x = xinit
u_init = None

q, p = dx.get_true_obj()
Q = torch.diag(q).unsqueeze(0).unsqueeze(0).repeat(
    mpc_T, n_batch, 1, 1
)
p = p.unsqueeze(0).repeat(mpc_T, n_batch, 1)

t_dir = tempfile.mkdtemp()
print('Tmp dir: {}'.format(t_dir))

action_history = []
for t in tqdm(range(T)):
    nominal_states, nominal_actions, nominal_objs = mpc.MPC(
        dx.n_state, dx.n_ctrl, mpc_T,
        u_init=u_init,
        u_lower=dx.lower, u_upper=dx.upper,
        lqr_iter=50,
        verbose=0,
        exit_unconverged=False,
        detach_unconverged=False,
        linesearch_decay=dx.linesearch_decay,
        max_linesearch_iter=dx.max_linesearch_iter,
        grad_method=GradMethods.AUTO_DIFF,
        eps=1e-2,
    )(x, QuadCost(Q, p), dx)
    
    next_action = nominal_actions[0]
    action_history.append(next_action)
    u_init = torch.cat((nominal_actions[1:], torch.zeros(1, n_batch, dx.n_ctrl)), dim=0)
    u_init[-2] = u_init[-3]
    x = dx(x, next_action)

    n_col = 4
    n_row = n_batch // n_col
    fig, axs = plt.subplots(n_row, n_col, figsize=(3*n_col,3*n_row), gridspec_kw = {'wspace':0, 'hspace':0})
    axs = axs.reshape(-1)
    for i in range(n_batch):
        dx.get_frame(x[i], ax=axs[i])
        axs[i].get_xaxis().set_visible(False)
        axs[i].get_yaxis().set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(t_dir, 'frame_{:03d}.png'.format(t)))
    plt.close(fig)
    
action_history = torch.stack(action_history).detach()[:,:,0]


# Plot actions
for t in tqdm(range(T)):
    fig, axs = plt.subplots(n_row, n_col, figsize=(3*n_col,3*n_row), gridspec_kw = {'wspace':0, 'hspace':0})
    axs = axs.reshape(-1)
    for i in range(n_batch):
        axs[i].plot(action_history[:,i], color='k')
        axs[i].set_ylim(-15, 15)
        axs[i].axvline(t, color='k', ls='--', linewidth=4)
        axs[i].get_xaxis().set_visible(False)
        axs[i].get_yaxis().set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(t_dir, 'actions_{:03d}.png'.format(t)))
    plt.close(fig)
    
    f1 = os.path.join(t_dir, 'frame_{:03d}.png'.format(t))
    f2 = os.path.join(t_dir, 'actions_{:03d}.png'.format(t))
    f_out = os.path.join(t_dir, '{:03d}.png'.format(t))
    os.system(f'convert {f1} {f2} +append -resize 1200x {f_out}')


vid_fname = 'cartpole.mp4'

if os.path.exists(vid_fname):
    os.remove(vid_fname)
    
cmd = 'ffmpeg -r 16 -f image2 -i {}/%03d.png -vcodec libx264 -crf 25 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p {}'.format(
    t_dir, vid_fname
)
os.system(cmd)
print('Saving video to: {}'.format(vid_fname))


video = io.open(vid_fname, 'r+b').read()
encoded = base64.b64encode(video)
HTML(data='''<video alt="test" controls>
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii')))