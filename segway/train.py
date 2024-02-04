import jax
import jax.numpy as jnp
import numpy as onp
import immrax as irx
import equinox as eqx
import equinox.nn as nn
from pathlib import Path
import optax
from functools import partial 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from immutabledict import immutabledict
import jax.experimental.compilation_cache.compilation_cache as cc
from segway import Segway
from control import lqr

device = 'gpu'

if device == 'gpu' :
    cc.initialize_cache('cache')

def jit (f, *args, **kwargs) :
    kwargs.setdefault('backend', device)
    return eqx.filter_jit(f, *args, **kwargs)

sys = Segway()
w = irx.icentpert(jnp.zeros(1), 0.0)

## Using closed-loop linearized system to inform nice polytope

A = jax.jacfwd(sys.f, 1)(0., jnp.array([0., 0., 0.]), jnp.array([0.]), jnp.zeros(1))
B = jax.jacfwd(sys.f, 2)(0., jnp.array([0., 0., 0.]), jnp.array([0.]), jnp.zeros(1))
Q = 10*jnp.eye(3)
R = jnp.eye(1)
LQR_K, _, _ = lqr(A, B, Q, R)
LQR_K = -LQR_K
Acl = A + B@LQR_K
L, U = jax.jit(jnp.linalg.eig, backend='cpu')(Acl)

print(A, B, LQR_K, L, U)

L = onp.real_if_close(L); U = onp.real_if_close(U)
reL = onp.real(L); imL = onp.imag(L)

# Convert to Jordan form (real)
Tinv = onp.empty_like(U, dtype=onp.float64)
real_idx = []; polar_tuples = []
skip = False
for i, l in enumerate(L) :
    v = onp.real_if_close(U[:,i])
    if not skip :
        if onp.iscomplex(l) :
            polar_tuples.append((i,i+1))
            rel = onp.real(l); iml = onp.imag(l)
            rev = onp.real(v); imv = onp.imag(v)
            Tinv[:,i] = -rev; Tinv[:,i+1] = imv
            skip = True
        else :
            real_idx.append(i)
            Tinv[:,i] = v
    else :
        skip = False
Tinv[onp.abs(Tinv) < 1e-10] = 0
T = onp.linalg.inv(Tinv); T[onp.abs(T) < 1e-10] = 0


S = jnp.vstack((
    T,
))

safe = irx.icentpert(jnp.zeros(3), 0.05)
Tsafe = irx.interval(T)@safe
s = irx.iconcatenate((Tsafe,))
slwhere = jnp.where(jnp.array([True, True, True]))
suwhere = jnp.where(jnp.array([True, True, True]))

def null_space(A, rcond=None):
    """Taken from scipy, with some modifications to use jax.numpy"""
    u, s, vh = jnp.linalg.svd(A, full_matrices=True)
    M, N = u.shape[0], vh.shape[1]
    if rcond is None:
        rcond = jnp.finfo(s.dtype).eps * max(M, N)
    tol = jnp.amax(s) * rcond
    num = jnp.sum(s > tol, dtype=int)
    Q = vh[num:,:].T.conj()
    return Q

Sdag = jnp.linalg.pinv(S)
NS = null_space(S.T)

@partial(jit)
def I_refine (A:jax.Array, y:irx.Interval) -> irx.Interval :
    A = irx.interval(A)
    def I_r (y) :
        ret = irx.icopy(y)
        for j in range(len(A)) :
            for i in range(len(y)) :
                reti = ((-A[j,:i] @ y[:i] - A[j,i+1:] @ y[i+1:])/A[j,i]) & ret[i]
                retl = ret.lower.at[i].set(reti.lower)
                retu = ret.upper.at[i].set(reti.upper)
                ret = irx.interval(retl, retu)
        return ret

    return I_r(y)

IS = jit(partial(I_refine, NS.T))
s = IS(s)

def build_lifted_embsys (net, eta) :
    Sp = Sdag + eta@NS.T
    lifted_sys = irx.LiftedSystem(sys, S, Sp)
    def lifted_net (y) :
        return net(Sp @ y)
    lifted_net.out_len = net.out_len
    lifted_clsys = irx.NNCSystem(lifted_sys, lifted_net)
    lifted_embsys = irx.NNCEmbeddingSystem(lifted_clsys, 'crown', 'local', 'local')
    return lifted_embsys

net = irx.NeuralNetwork('./models/100r100r2', False)
eta = jnp.zeros((3,len(NS.T)))
nu = jnp.zeros((len(NS.T), len(NS.T)))

def Laccuracy (net:irx.NeuralNetwork, X, U) :
    return jnp.mean((jax.vmap(net)(X) - U)**2)

orderings = irx.standard_ordering(1+len(S)+1+len(w)) 
corners = irx.bot_corner (1+len(S)+1+len(w))

def ES (net:irx.NeuralNetwork, eta:jax.Array, nu:jax.Array) :
    lifted_embsys = build_lifted_embsys(net, eta)
    return lifted_embsys.E(0., irx.i2ut(s), w,
        orderings=orderings, 
        corners=corners,
        refine=IS)

def LS (net:irx.NeuralNetwork, eta:jax.Array, nu:jax.Array, epsl:float=0.05, epsu:float=0.05) :
    E = ES(net, eta, nu)
    def relu_eps (x, eps) :
        return jax.nn.relu(x + eps)
    return jnp.sum(jax.vmap(partial(relu_eps, eps=epsl))(-E[:len(S)][slwhere])) \
         + jnp.sum(jax.vmap(partial(relu_eps, eps=epsu))( E[len(S):][suwhere])) 

@jit
def loss(params, X, U) :
    net, eta, nu = params
    return Laccuracy(net, X, U) + 100.*LS(net, eta, nu) # + 0.0001*jnp.trace(eta.T@eta)

Xtrain = irx.utils.gen_ics(safe, 100000)
Utrain = jax.vmap(lambda x: LQR_K@x)(Xtrain)

print(f"||U|| max = {jnp.linalg.norm(Utrain, axis=1).max()}")

def dataloader (batch_size=1000) :
    while True :
        for i in range(0, len(Xtrain), batch_size) :
            yield Xtrain[i:i+batch_size], Utrain[i:i+batch_size]

def train(params, optim, steps, minsteps, print_every=1) -> irx.NeuralNetwork :
    opt_state = optim.init(eqx.filter(params, eqx.is_array))

    @jit
    def make_step (params, X, U, opt_state) :
        loss_value, grads = eqx.filter_value_and_grad(loss)(params, X, U)
        updates, opt_state = optim.update(grads, opt_state, params)
        params = eqx.apply_updates(params, updates)
        net, eta, nu = params
        return (params, opt_state, loss_value, jnp.sqrt(Laccuracy(net, X, U)), 
                ES(net, eta, nu), jax.vmap(net)(Xtrain))

    for step, (X, U) in zip(range(steps), dataloader(1000)) :
        params, opt_state, train_loss, RMSE, EGval, vfs = make_step(params, X, U, opt_state)
        if (step % print_every) == 0 or (step == steps - 1) :
            net, eta, nu = params
            net.save()
            whole_loss = loss(params, Xtrain, Utrain)
            print(
                f'{step=}, train_loss={whole_loss.item()}, RMSE={jnp.sqrt(whole_loss)},'
                f'\nESl={EGval[:len(S)][slwhere]}, \nESu={EGval[len(S):][suwhere]}'
                f'\neta={eta}'
                f'\nnu={nu} \n'
            )
        if (jnp.all(EGval[:len(S)][slwhere] >= 0) 
            and jnp.all(EGval[len(S):][suwhere] <= 0)
            and step >= minsteps) :
            print('EG constraints satisfied, stopping training')
            print(
                f'{step=}, train_loss={train_loss.item()}, RMSE={RMSE},'
                f'\nESl={EGval[:len(S)][slwhere]}, \nESu={EGval[len(S):][suwhere]}'
                f'\neta={eta} \n'
                f'\nnu={nu} \n'
            )
            return params
    
    return params

optim = optax.adam(0.01)
net, eta, nu = train((net, eta, nu), optim, 100000, 10000, 100)
net.save()


