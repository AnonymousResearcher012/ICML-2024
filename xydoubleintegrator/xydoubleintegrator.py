import jax.numpy as jnp
import immrax as irx

class XYDoubleIntegrator (irx.OpenLoopSystem) :
    """A double integrator in the XY plane

    The state is [px, py, vx, vy]
    """
    def __init__ (self) :
        self.evolution = 'continuous'
        self.xlen = 4
        self.ulim = 10.
    def f(self, t, x, u, w) :
        return jnp.array([
            x[2],
            x[3],
            self.ulim*jnp.tanh(u[0]/self.ulim)*(1+w[0]),
            self.ulim*jnp.tanh(u[1]/self.ulim)*(1+w[1]),
        ])
