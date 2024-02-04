import jax
import jax.numpy as jnp
import numpy as onp
import immrax as irx
import matplotlib.pyplot as plt
from segway import Segway
from control import lqr
from pypoman import compute_polytope_vertices
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
    "font.size": 14
})
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
fig.tight_layout()

net = irx.NeuralNetwork('./models/100r100r2_invariant', True)
sys = Segway()

A = jax.jacfwd(sys.f, 1)(0., jnp.array([0., 0., 0.]), jnp.array([0.]), jnp.array([0.]))
B = jax.jacfwd(sys.f, 2)(0., jnp.array([0., 0., 0.]), jnp.array([0.]), jnp.array([0.]))
Q = 10*jnp.eye(3)
R = jnp.eye(1)
LQR_K, _, _ = lqr(A, B, Q, R)
LQR_K = -LQR_K
clsys = irx.NNCSystem(sys, net)

S = onp.array([[-5.149507 , -2.5232546, -2.2642095],
       [ 6.111345 ,  2.9945545,  1.458305 ],
       [ 5.209984 ,  5.362502 ,  1.916359 ]])

Sp = onp.linalg.inv(S)
safe = irx.icentpert(jnp.zeros(3), 0.04)
s = irx.interval(S)@safe

lifted_sys = irx.LiftedSystem(sys, S, Sp)
def lifted_net (y) :
    return net(Sp @ y)
lifted_net.out_len = net.out_len
lifted_net.u = lambda t, x : lifted_net(x)
lifted_clsys = irx.NNCSystem(lifted_sys, lifted_net)
lifted_embsys = irx.NNCEmbeddingSystem(lifted_clsys, 'crown', 'local', 'local')

mcx0s = irx.utils.gen_ics(s, 100)

def w_map (t, x) :
    return irx.interval([0.])

def compute_traj (x0) :
    return clsys.compute_trajectory(0., 5., Sp@x0, (w_map,), 0.01, solver='tsit5')

mc_traj = jax.vmap(compute_traj)(mcx0s)
tfinite = jnp.isfinite(mc_traj.ts[0,:])

for i in range(len(mcx0s)) :
    xx = mc_traj.ys[i]
    ax.plot(xx[tfinite,0], xx[tfinite,2], xx[tfinite,1], color='tab:red', zorder=0)

ax.set_xlabel(r'$\phi$')
ax.set_ylabel(r'$\dot{\phi}$')
ax.set_zlabel(r'$v$')

orderings = irx.standard_ordering(1+len(S)+1+1) 
corners = irx.bot_corner(1+len(S)+1+1)
print(f'{lifted_embsys.E(0., irx.i2ut(s), irx.interval([0.]), orderings=orderings, corners=corners)} >=_SE 0')

H = onp.vstack((-S,S))
b = onp.concatenate((-s.lower,s.upper))

vertices = onp.array(compute_polytope_vertices(H, b))
vertices[[1,2]] = vertices[[2,1]]
vertices[:,[1,2]] = vertices[:,[2,1]]

ax.scatter(vertices[:,0], vertices[:,1], vertices[:,2], c='tab:blue')
# for i in range(len(vertices)) :
#     ax.text(vertices[i,0], vertices[i,1], vertices[i,2], f'{i}')

faces = [[2,1,6], [0,1,3], [0,3,7], [0,2,6], [0,7,6], [0,2,1],
         [7,5,3], [7,5,6], [5,3,4], [5,4,6], [3,4,1], [6,1,4]]
verts = [[vertices[i] for i in f] for f in faces]
ax.add_collection3d(Poly3DCollection(verts, color='tab:blue', linewidths=1, alpha=.25))

ax.view_init(azim=130)
fig.savefig('segway.pdf')

plt.show()