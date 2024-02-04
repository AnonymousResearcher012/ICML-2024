import jax
import jax.numpy as jnp
from jax.numpy import cos, sin
import immrax as irx

class Segway (irx.OpenLoopSystem) :
    def __init__ (self) :
        self.evolution = 'continuous'
        self.xlen = 3
    def f(self, t, x, u, w) :
        phi, v, phid = x
        u = u[0]
        c1 = 1.8
        c2 = 11.5
        c3 = 10.9
        c4 = 68.4
        c5 = 1.2
        d1 = 9.3
        d2 = 58.8
        d3 = 38.6
        d4 = 234.5
        d5 = 208.3
        b  = 24.7
        return jnp.array([
            phid,
            (cos(phi)*(-c1*u + c2*v + 9.8*sin(phi)) - c3*u + c4*v - c5*phid**2*sin(phi))/(cos(phi) - b),
            ((d1*u - d2*v)*cos(phi) + d3*u - d4*v - sin(phi)*(d5 + phid**2*cos(phi)))/(cos(phi)**2 - b)
        ])
