import jax
import jax.numpy as jnp
import numpy as onp
import immrax as irx
import matplotlib.pyplot as plt
from xydoubleintegrator import XYDoubleIntegrator
from pypoman import compute_polytope_vertices, plot_polygon, project_polytope

N = 500

S = jnp.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [1., 0., 1., 0.],
    [0., 1., 0., 1.],
])
s = irx.icentpert(jnp.zeros(6), [1., 1., 1., 1., 0.9, 0.9])

Sp = jnp.linalg.pinv(S)
mc_x0s = (Sp@irx.utils.gen_ics(s, N).T).T

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
    "font.size": 14
})
fig, axs = plt.subplots(2, 3, figsize=(10,5), dpi=100)
fig.tight_layout()
fig.subplots_adjust(hspace=0.2, wspace=0.2)

H = onp.vstack((-S,S))
b = onp.concatenate((-s.lower,s.upper))
E = onp.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0]
])
P1 = project_polytope((E, onp.zeros(2)), (H, b))
E = onp.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0]
])
P2 = project_polytope((E, onp.zeros(2)), (H, b))

titles = ['$\\mathcal{L}^1$ - Both', '$\\mathcal{L}^2$ - No Data', '$\\mathcal{L}^3$ - No Inv']

for j, model in enumerate(['100r100r2-final_both', '100r100r2-final_nodata', '100r100r2-final_noinv']) :
    sys = XYDoubleIntegrator()
    net = irx.NeuralNetwork('models/' + model)
    clsys = irx.NNCSystem(sys, net)

    def w_map (t, x) :
        return jax.random.uniform(jax.random.PRNGKey(0), (2,), float, -0.1, 0.1)

    def compute_traj (x0) :
        traj = jax.lax.cond(jnp.logical_or(jnp.any(S@x0 > s.upper), jnp.any(S@x0 < s.lower)), 
                            lambda : clsys.compute_trajectory(0., 1., jnp.zeros(4), (w_map,), 0.0005, solver='tsit5'), 
                            lambda : clsys.compute_trajectory(0., 1., x0, (w_map,), 0.0005, solver='tsit5'))
        return traj

    mc_traj = jax.vmap(compute_traj)(mc_x0s)
    tfinite = jnp.isfinite(mc_traj.ts[0,:])
    ones = jnp.ones((1,len(tfinite)))

    num_violations = 0
    for i in range(len(mc_x0s)) :
        axs[0,j].plot(mc_traj.ys[i,tfinite,0], mc_traj.ys[i,tfinite,2], color='tab:red', zorder=1)
        axs[0,j].scatter(mc_traj.ys[i,0,0], mc_traj.ys[i,0,2], s=2., color='tab:red', zorder=2)
        axs[1,j].plot(mc_traj.ys[i,tfinite,0], mc_traj.ys[i,tfinite,1], color='tab:red', zorder=1)
        axs[1,j].scatter(mc_traj.ys[i,0,0], mc_traj.ys[i,0,1], s=2., color='tab:red', zorder=2)
        if (jnp.any(S@(mc_traj.ys[i].T) >= s.upper.reshape(-1,1)@ones) or jnp.any(S@(mc_traj.ys[i].T) <= s.lower.reshape(-1,1)@ones)) :
            print('Constraint violated')
            num_violations += 1
    axs[0,j].set_title(titles[j])
    axs[0,j].set_xlim(-1.1, 1.1); axs[0,j].set_ylim(-1.1, 1.1)
    axs[0,j].set_xlabel(r'$p_x$', labelpad=-4)
    axs[0,j].set_ylabel(r'$v_x$', labelpad=-2)
    plt.sca(axs[0,j])
    plot_polygon(P1)
    plt.sca(axs[1,j])
    axs[1,j].set_xlim(-1.1, 1.1); axs[1,j].set_ylim(-1.1, 1.1) 
    axs[1,j].set_xlabel(r'$p_x$', labelpad=-2)
    axs[1,j].set_ylabel(r'$p_y$', labelpad=-2)
    plot_polygon(P2)

fig.savefig('compare.pdf')
plt.show()