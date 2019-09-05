import jax
import jax.numpy as np
import numpy as NP
from . import tensor

def gradients(scalar, params):
    assert scalar.shape == ()
    deps = scalar.all_dependencies
    argnums = [i for i, dep in enumerate(deps) if dep in params]
    def fn(*args):
        return scalar.get(dict([(dep,arg) for arg, dep in zip(args, deps)]),
                          force=True)
    shapes = [param.shape for param in params]
    dtypes = [param.dtype for param in params]
    grad_fn = jax.grad(fn, argnums)
    return tensor.List(grad_fn, shapes, dtypes, args=deps)


def function(*args, outputs=[], updates={}):

    # ensure that we only aim at updating variables
    for update in updates.keys():
        assert isinstance(update, tensor.Variable)

    # now create the functin that will take the inputs and return
    # the update values (if any) and the outputs, this function is the one that
    # will be jit compiled for performances
    def fn(*fnargs, _args=args, _hiddens=updates):
        _args = list(args) + list(_hiddens.keys())
        kwargs = dict([(key, value) for key, value in zip(_args, fnargs)])
        # compute the outputs
        fn_outputs = [output.get(kwargs) for output in outputs]
        # compute the values of the updates
        fn_updates = [output.get(kwargs) for output in updates.values()]
        return fn_outputs, fn_updates
    global jitfn
    jitfn = jax.jit(fn)

    # now define the actual function that will be used by the user
    def meta(*fnargs, hiddens=list(updates.keys())):
        assert len(fnargs) == len(args)
        outputs, updates2 = jitfn(*fnargs, *[hidden.value for hidden in hiddens])
        for key, update in zip(updates.keys(), updates2):
            key.value=update
        return outputs
    return meta
