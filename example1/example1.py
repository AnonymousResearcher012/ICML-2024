import jax
import jax.numpy as jnp
import numpy as onp
import immrax as irx
from functools import partial
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pypoman import compute_polytope_vertices, plot_polygon

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
    "font.size": 14
})
fig, ax = plt.subplots(1,1,figsize=(5,4)) 
fig.tight_layout()
ax.set_xlim(-2.1, 2.1)
ax.set_ylim(-3.1, 3.1)
ax.set_xlabel(r'$x_1$', labelpad=-2)
ax.set_ylabel(r'$x_2$', labelpad=0, rotation=0)

class DoubleIntegrator (irx.ControlledSystem) :
    def __init__(self) :
        self.evolution = 'continuous'
        self.xlen = 2
    def f(self, t, x, u, w) :
        return jnp.array([x[1], u[0]])

sys = DoubleIntegrator()

K = jnp.array([[-2., -3.]])
control = irx.LinearControl(K)

x0 = irx.icentpert(jnp.zeros(2), 1.)

# In standard coordinates, [-1,1]\times[-1,1] is not FI.
clsys = irx.ControlledSystem(sys, control)
embsys = irx.jacemb(clsys)
izero = irx.interval([0.])
print(f'{embsys.E(izero, irx.i2ut(x0), izero)} >/=_SE 0')
ax.add_patch(Rectangle((-1.,-1.), 2., 2., fill=False, edgecolor='tab:red', linewidth=2.))

# In transformed coordinates, [-1,1]\times[-1,1] is FI.
T = jnp.array([[1., 1.], [-1., -2.]])
Tinv = jnp.linalg.inv(T) 
lifted_clsys = irx.LiftedSystem(clsys, Tinv, T)
lifted_embsys = irx.jacemb(lifted_clsys)
print(f'{lifted_embsys.E(izero, irx.i2ut(x0), izero)} >=_SE 0')
plot_polygon(compute_polytope_vertices(
    onp.vstack((-Tinv, Tinv)), onp.concatenate((-x0.lower, x0.upper))),
    alpha=1., fill=False, color='tab:blue', linewidth=2.)

# We can also verify another polytope using different lifted coordinates.
H = jnp.array([[1., 0.], [0., 1.], [1.,1.]])
Hp = jnp.linalg.pinv(H)
lifted_clsys = irx.LiftedSystem(clsys, H, Hp)
lifted_embsys = irx.jacemb(lifted_clsys)
N = irx.utils.null_space(H.T)
I_H = partial(irx.utils.I_refine, N.T)
y0 = irx.icentpert(jnp.zeros(3), 1.)
print(f'{lifted_embsys.E(izero, irx.i2ut(y0), izero, refine=I_H)} >=_SE 0')
plot_polygon(compute_polytope_vertices(
    onp.vstack((-H, H)), onp.concatenate((-y0.lower, y0.upper))),
    alpha=1., fill=False, color='tab:green', linewidth=2.)

# Plotting some simulations.
x0s = jnp.array([
    [-2., 3.], [2., -3.],
    [1.,1.], [-1.,-1.],
    [1.,-1.], [-1.,1.],
    [-1.,0.], [1.,0.],
    [0.,1.], [0.,-1.],
])
colors = [
    'tab:blue', 'tab:blue',
    'tab:red', 'tab:red',
    'tab:green', 'tab:green',
    'tab:green', 'tab:green',
    'tab:blue', 'tab:blue',
]

def w_map (t, x) :
    return jnp.array([0.])
for i, x0 in enumerate(x0s) :
    traj = clsys.compute_trajectory(0., 5., x0, (w_map,), 0.01, solver='tsit5')
    tfinite = jnp.isfinite(traj.ts)
    xx = traj.ys[tfinite]
    ax.plot(xx[:,0], xx[:,1], color=colors[i], linestyle='--')
    ax.scatter(xx[0,0], xx[0,1], s=30., color=colors[i])

fig.savefig('example1.pdf')
plt.show()
