import jax
import jax.numpy as np
import numpy as NP


def NewOp(fn, name=''):
    """function that produces a new Op based on a given function"""

    def init(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.name = 'Tensor: name='+name+', op='+fn.__name__
        all_tensors = list(args) + list(kwargs.values())
        all_dependencies = sum([arg.all_dependencies for arg in all_tensors
                                if hasattr(arg,'all_dependencies')], [])
        self.all_dependencies = list(set(all_tensors+all_dependencies))
        super(self.__class__, self).__init__(fn)

    attributes = {'__init__': init}
    new_class = type('NewOp'+fn.__name__, (Tensor,), attributes)
    return new_class


def gradients(tensor, params):
    assert tensor.shape == ()
    deps = tensor.all_dependencies
    argnums = [i for i, dep in enumerate(deps) if dep in params]
    def fn(*args):
        return tensor.get(dict([(dep,arg) for arg, dep in zip(args, deps)]),
                          force=True)
    shapes = [param.shape for param in params]
    dtypes = [param.dtype for param in params]
    grad_fn = jax.grad(fn, argnums)
    return List(grad_fn, shapes, dtypes, args=deps)


def function(*args, outputs=[], updates={}):

    # ensure that we only aim at updating variables
    for update in updates.keys():
        assert isinstance(update, Variable)

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
