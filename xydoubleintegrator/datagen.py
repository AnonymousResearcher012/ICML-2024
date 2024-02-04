from tqdm import tqdm
from multiprocessing import Pool
from datetime import datetime
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
import numpy as np

DELTA = 0.5
FILENAME = 'fourobs'
PROCESSES = 10
PLOT_DATA = False

FILEPATH = 'data/' + FILENAME + datetime.now().strftime('_%Y%m%d-%H%M%S') + '.npy'
print("Writing to " + FILEPATH)

DT = 0.05
TF = 5.
N = round(TF / DT) + 1

RANGES = [
    [-1, 1.],
    [-1, 1.],
    [-1.,1.],
    [-1.,1.]
]
Xmesh = np.meshgrid(*[np.arange(range[0], range[1], DELTA) + DELTA/2 for range in RANGES])
X0 = np.array(Xmesh).reshape(4,-1).T
X0 = X0*10

print(X0)

NUM_SAMPLES = len(X0)

print(f"Generating {(NUM_SAMPLES,NUM_SAMPLES*N)} samples for a {DELTA}-covering")

X = np.ones((NUM_SAMPLES*N, 4))
U = np.ones((NUM_SAMPLES*N, 2))

def init_process () :
    import jax
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_platform_name', 'cpu')
    from xydoubleintegrator import XYDoubleIntegrator
    from mpc import XYDoubleMPC

    global sys
    global mpc
    global to_jax

    def _to_jax (x) :
        return jnp.array(x)
    to_jax = _to_jax

    sys = XYDoubleIntegrator()
    mpc = XYDoubleMPC(sys, DT, TF)

def task (x) :
    return mpc.do_mpc(to_jax(x))

pool = Pool(processes=PROCESSES, initializer=init_process)

# for i, result in enumerate(tqdm(pool.imap_unordered(task, X0), total=NUM_SAMPLES, smoothing=0)) :
for i, result in enumerate(tqdm(pool.imap_unordered(task, X0), total=NUM_SAMPLES, smoothing=0)) :
    uu, xx = result

    X[i*N:(i+1)*N,:] = xx
    U[i*N:(i+1)*N,:] = uu
    
    # if PLOT_DATA :
    #     axsi = (axsi + 1) % len(axs); ax = axs[axsi]; ax.clear()
    #     # ax = axs[0]
    #     ax.add_patch(Circle((4,4),3/1.25,lw=0,fc='salmon'))
    #     ax.add_patch(Circle((-4,4),3/1.25,lw=0,fc='salmon'))
    #     # points = np.array([xx[:,0],xx[:,1]]).T.reshape(-1,1,2)
    #     # segs = np.concatenate([points[:-1],points[1:]],axis=1)
    #     # lc = LineCollection(segs, lw=2, cmap=plt.get_cmap('cividis'))
    #     # lc.set_array(tt)
    #     # ax.add_collection(lc)
    #     # ax.set_xlim([-10,10]); ax.set_ylim([-10,10])
    #     # ax.set_xlabel('$p_x$',labelpad=3); ax.set_ylabel('$p_y$',labelpad=3, rotation='horizontal')

    #     # cmap = sns.cubehelix_palette(rot=-0.4, as_cmap=True)
    #     points = ax.scatter(xx[:,0], xx[:,1], c=tt, cmap=plt.get_cmap('cividis'), s=1)
    #     ax.set_xlim([-10,10]); ax.set_ylim([-10,10])
    #     # ax.set_xlim([-15,15]); ax.set_ylim([-15,15])
    #     # fig.colorbar(points, ax=ax)
    #     # ax.set_title("y vs x, color t")
    #     plt.ion(); plt.show(); plt.pause(0.0001)
    #     pass


def numpy_to_file (X, U, filename) :
    with open(filename, 'wb') as f :
        np.savez(f, X=X, U=U)
    
numpy_to_file(X, U, FILEPATH)

fig, ax = plt.subplots(1,1)

# ax.add_patch(Circle((5,5),3,lw=0,fc='salmon'))
# ax.add_patch(Circle((5,-5),3,lw=0,fc='salmon'))
# ax.add_patch(Circle((-5,5),3,lw=0,fc='salmon'))
# ax.add_patch(Circle((-5,-5),3,lw=0,fc='salmon'))

ax.scatter(X[:,0], X[:,1], c='k', s=0.2)
# ax.quiver(X[:,0], X[:,1], U[:,0], U[:,1])

plt.show()

# # control = XYDoubleIntegratorMPC()
# # model = XYDoubleIntegratorModel(control)

# # t_end = control.u_step * problem_horizon

# # X0 = gen_ics(RANGES, NUM_TRAJS)

# # X = np.ones((NUM_POINTS, 4))
# # U = np.ones((NUM_POINTS, 2))

# # tt = np.arange(0,t_end,control.u_step)

# # def task(x) :
# #     try :
# #         traj = model.compute_trajectory(x0=x, enable_bar=False, t_span=[0,t_end], t_step=0.01, method='euler')
# #         return traj(tt).T, traj.u_disc
# #     except Exception as e:
# #         print('failed... trying with new IC')
# #         print(e)
# #         return task(gen_ics(RANGES, 1)[0,:])


# # if PLOT_DATA :
# #     fig, axs = plt.subplots(4,4,dpi=100,figsize=[10,10])
# #     fig.set_tight_layout(True)
# #     axs = axs.reshape(-1)
# #     axsi = 0

# # for i, result in enumerate(tqdm(pool.imap_unordered(task, X0), total=NUM_TRAJS, smoothing=0)) :
# #     # tt, xx, uu = result
# #     # traj = result
# #     xx, uu = result

# #     X[i*problem_horizon:(i+1)*problem_horizon,:] = xx
# #     U[i*problem_horizon:(i+1)*problem_horizon,:] = uu
# #     if PLOT_DATA :
# #         axsi = (axsi + 1) % len(axs); ax = axs[axsi]; ax.clear()
# #         # ax = axs[0]
# #         ax.add_patch(Circle((4,4),3/1.25,lw=0,fc='salmon'))
# #         ax.add_patch(Circle((-4,4),3/1.25,lw=0,fc='salmon'))
# #         # points = np.array([xx[:,0],xx[:,1]]).T.reshape(-1,1,2)
# #         # segs = np.concatenate([points[:-1],points[1:]],axis=1)
# #         # lc = LineCollection(segs, lw=2, cmap=plt.get_cmap('cividis'))
# #         # lc.set_array(tt)
# #         # ax.add_collection(lc)
# #         # ax.set_xlim([-10,10]); ax.set_ylim([-10,10])
# #         # ax.set_xlabel('$p_x$',labelpad=3); ax.set_ylabel('$p_y$',labelpad=3, rotation='horizontal')

# #         # cmap = sns.cubehelix_palette(rot=-0.4, as_cmap=True)
# #         points = ax.scatter(xx[:,0], xx[:,1], c=tt, cmap=plt.get_cmap('cividis'), s=1)
# #         ax.set_xlim([-10,10]); ax.set_ylim([-10,10])
# #         # ax.set_xlim([-15,15]); ax.set_ylim([-15,15])
# #         # fig.colorbar(points, ax=ax)
# #         # ax.set_title("y vs x, color t")
# #         plt.ion(); plt.show(); plt.pause(0.0001)
# #         pass

# # numpy_to_file(X, U, FILEPATH)
