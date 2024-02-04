import jax
import jax.numpy as jnp
from typing import Union, Tuple
from jaxtyping import Float, Integer
import immrax as irx
from xydoubleintegrator import XYDoubleIntegrator
from functools import partial
from cyipopt import minimize_ipopt

def jit (f, *args, **kwargs) :
    kwargs.setdefault('backend', 'cpu')
    return jax.jit(f, *args, **kwargs)

class XYDoubleMPC (irx.Control) :
    sys:XYDoubleIntegrator
    dt:float
    tf:float
    def __init__(self, sys, dt, tf) -> None:
        self.sys = sys
        self.dt = dt
        self.tf = tf
        self.N = round(tf / dt) + 1

        print(f'Creating MPC Controller with {dt=}, {tf=}, N={self.N} (JIT compiling)')

        from time import time
        time0 = time()

        self.obj_grad = jit(jax.grad(self.obj))  # Objective Gradient
        self.obj_hess = jit(jax.jacfwd(jax.jacrev(self.obj)))  # Objective Hessian
        self.con_ineq_jac = jit(jax.jacfwd(self.con_ineq))  # Constraint Jacobian
        self.con_ineq_hess = jit(jax.jacfwd(jax.jacrev(self.con_ineq))) # Constraint Hessian
        
        # # JIT compiling
        self.u0 = jnp.zeros((self.N, 2)).reshape(-1)
        x0dum = jnp.zeros(self.sys.xlen)
        self.obj_grad(self.u0, x0dum)
        self.obj_hess(self.u0, x0dum)
        self.con_ineq_jac(self.u0, x0dum)
        self.con_ineq_hessvp(self.u0, jnp.ones(5*self.N), x0dum)
        
        timef = time()
        print(f'Finished setting up MPC controller in {timef - time0} seconds')

    @partial(jit, static_argnums=(0,))
    def rollout_ol_sys_undisturbed (self, u:jax.Array, x0:jax.Array) -> jax.Array :
        def f_euler (xt, ut) :
            xtp1 = xt + self.dt*self.sys.f(0., xt, ut, jnp.array([0.]))
            return (xtp1, xtp1)
        _, x = jax.lax.scan(f_euler, x0, u.reshape(-1,2))
        return x
    
    @partial(jit, static_argnums=(0,))
    def obj (self, u: jax.Array, x0:jax.Array) -> jax.Array :
        x = self.rollout_ol_sys_undisturbed(u, x0)
        return jnp.sum(x**2) + jnp.sum(u**2)

    @partial(jit, static_argnums=(0,))
    def con_ineq (self, u: jax.Array, x0:jax.Array) -> jax.Array :
        x = self.rollout_ol_sys_undisturbed(u, x0)
        r_l2 = jnp.linalg.norm(x[:,:2], axis=1)
        v_l2 = jnp.linalg.norm(x[:,2:], axis=1)
        obs = [
            (x[:,0] - 5)**2 + (x[:,1] - 5)**2 - 3**2,
            (x[:,0] + 5)**2 + (x[:,1] - 5)**2 - 3**2,
            (x[:,0] - 5)**2 + (x[:,1] + 5)**2 - 3**2,
            (x[:,0] + 5)**2 + (x[:,1] + 5)**2 - 3**2,
        ]
        return jnp.concatenate((
            self.sys.nu0 + self.sys.nu1*r_l2 - v_l2,
            *obs
        ))

    # Constraint Hessian-Vector Product
    @partial(jit, static_argnums=(0,))
    def con_ineq_hessvp (self, u, v, x0) :
        def hessvp (u) :
            _, hvp = jax.vjp(partial(self.con_ineq, x0=x0), u)
            return hvp(v)[0] # One tangent, one output. u^T dc_v
        return jax.jacrev(hessvp)(u) 

    def do_mpc (self, x:jax.Array) -> Tuple[jax.Array, jax.Array] :
        cons = [ 
            {'type': 'ineq', 
             'fun': partial(self.con_ineq, x0=x), 
             'jac': partial(self.con_ineq_jac, x0=x),
             'hess': partial(self.con_ineq_hessvp, x0=x)
            }, ]
        ipopt_opts = {
            b'sb': 'yes',
            b'disp': False, 
            b'linear_solver': 'ma57', 
            b'hsllib': 'libcoinhsl.so', 
            b'tol': 1e-5,
            b'max_iter': 1000,
        }

        # Solve the optimization problem
        res = minimize_ipopt(
            partial(self.obj, x0=x), 
            jac=partial(self.obj_grad, x0=x), 
            hess=partial(self.obj_hess, x0=x), 
            x0=self.u0, constraints=cons, options=ipopt_opts)

        uu = res.x
        return uu.reshape(-1,2), self.rollout_ol_sys_undisturbed(uu, x)

    def u(self, t: Union[Integer, Float], x: jax.Array) -> jax.Array :
        uu, _ = self.do_mpc(x)
        return uu[0,:]

if __name__ == '__main__' :
    sys = XYDoubleIntegrator()
    mpc = XYDoubleMPC(sys, 0.05, 5.)

    uu, xx = mpc.do_mpc(jnp.array([-10.,10.,10.,0.]))
    print(uu)
    print(uu.shape)
    print(xx.shape)

