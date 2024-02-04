import jax
import jax.numpy as jnp
import numpy as onp
import immrax as irx
import equinox as eqx
import equinox.nn as nn
from pathlib import Path
import optax
from functools import partial 
from immrax.utils import file_to_jax
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from immutabledict import immutabledict
import jax.experimental.compilation_cache.compilation_cache as cc
from xydoubleintegrator import XYDoubleIntegrator

device = 'gpu'

if device == 'gpu' :
    cc.initialize_cache('cache')

def jit (f, *args, **kwargs) :
    kwargs.setdefault('backend', device)
    return eqx.filter_jit(f, *args, **kwargs)

FILENAMES = ['fourobs_20240201-042441']

sys = XYDoubleIntegrator()

r2 = jnp.sqrt(2)
S = jnp.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [1., 0., 1., 0.],
    [0., 1., 0., 1.],
 
])
s = irx.icentpert(jnp.zeros(6), [1., 1., 1., 1., 0.9, 0.9])
w = irx.icentpert(jnp.zeros(2), 0.1)

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
                # if jnp.nonzero(A[j,i].lower):
                def b1 () :
                    return ((-A[j,:i] @ ret[:i] - A[j,i+1:] @ ret[i+1:])/A[j,i]) & ret[i]
                def b2 () :
                    return ret[i]
                # reti = ((-A[j,:i] @ ret[:i] - A[j,i+1:] @ ret[i+1:])/A[j,i]) & ret[i]
                reti = jax.lax.cond(jnp.abs(A[j,i].lower) > 1e-10, b1, b2)
                retl = ret.lower.at[i].set(reti.lower)
                retu = ret.upper.at[i].set(reti.upper)
                ret = irx.interval(retl, retu)
        return ret

    return I_r(I_r(y))

IS = jit(partial(I_refine, NS.T))

s = IS(IS(s))

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
eta = jnp.zeros((4,len(NS.T)))
nu = jnp.zeros((len(NS.T), len(NS.T)))

@jit
def Laccuracy (net:irx.NeuralNetwork, X, U) :
    return jnp.mean((jax.vmap(net)(X) - U)**2)

orderings = irx.standard_ordering(1+len(S)+2+2) 
corners = irx.bot_corner(1+len(S)+2+2)

def ES (net:irx.NeuralNetwork, eta:jax.Array=eta, nu:jax.Array=nu) :
    lifted_embsys = build_lifted_embsys(net, eta)
    return lifted_embsys.E(irx.interval([0.]), irx.i2ut(s), w,
        orderings=orderings, 
        corners=corners,
        refine=IS
    )

def LS (net:irx.NeuralNetwork, eta:jax.Array=eta, nu:jax.Array=nu, epsl:float=0.05, epsu:float=0.05) :
    E = ES(net, eta, nu)
    def relu_eps (x, eps) :
        return jax.nn.relu(x + eps)
    return jnp.sum(jax.vmap(partial(relu_eps, eps=epsl))(-E[:len(S)])) \
         + jnp.sum(jax.vmap(partial(relu_eps, eps=epsu))( E[len(S):])) 

def loss(params, X, U) :
    net, eta, nu = params
    return Laccuracy(net, X, U) + 10.*LS(net, eta, nu) 

Xtrain, Utrain = file_to_jax(FILENAMES)
print(f'Loaded data from {FILENAMES}, total of {len(Xtrain)} points')
print(f'mean ||U||={jnp.mean(Utrain**2)}')
print(f'||X||_inf={jnp.linalg.norm(Xtrain, jnp.inf, axis=0)}')
print(f'||U||_inf={jnp.linalg.norm(Utrain, jnp.inf, axis=0)}')

training_loss = jit(partial(loss, X=Xtrain, U=Utrain))
Laccuracy_loss = jit(partial(Laccuracy, X=Xtrain, U=Utrain))

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
        params, opt_state, train_loss, RMSE, ESval, vfs = make_step(params, X, U, opt_state)
        if (step % print_every) == 0 or (step == steps - 1) :
            net, eta, nu = params
            net.save()
            whole_loss = training_loss(params)
            print(
                f'{step=}, train_loss={whole_loss.item()}, RMSE={jnp.sqrt(Laccuracy_loss(net))},'
                f'\nESl={ESval[:len(S)]}, \nESu={ESval[len(S):]}'
                f'\neta={eta}'
                f'\nnu={nu} \n'
            )
        if (jnp.all(ESval[:len(S)] >= 0) 
            and jnp.all(ESval[len(S):] <= 0)
            and step >= minsteps) :
        # if step >= minsteps :
            print('ES constraints satisfied, stopping training')
            whole_loss = training_loss(params)
            print(
                f'{step=}, train_loss={whole_loss.item()}, RMSE={jnp.sqrt(Laccuracy_loss(net))},'
                f'\nESl={ESval[:len(S)]}, \nESu={ESval[len(S):]}'
                f'\neta={eta} \n'
                f'\nnu={nu} \n'
            )
            return params
    
    return params

optim = optax.adam(0.01)
# optim = optax.adamax(0.1)
net, eta, nu = train((net, eta, nu), optim, 100000, 1000, 100)
net.save()

print(Sdag + eta@NS.T)