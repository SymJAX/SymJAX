import jax
import jax.numpy as np
import numpy as NP

def isdep(item):
    return isinstance(item, Variable) or isinstance(item, Placeholder)


class Tensor:

    def __init__(self, _eval=None, shape=None, dtype=None):
        if _eval is not None:
            self._eval = _eval
            tree = jax.eval_shape(self._eval, *self.args, **self.kwargs)
            self.shape, self.dtype = tree.shape, tree.dtype
        else:
            assert shape is not None
            assert dtype is not None
            self.shape = shape
            self.dtype = dtype
        if not hasattr(self, 'name'):
            self.name = ''
        self.eval_value = None

    def __repr__(self):
        return '(Tensor' + self.name + ', dtype=' + str(self.dtype) + \
               ', shape='+str(self.shape) + ')'

    def __str__(self):
        return self.__repr__()

    def get(self, ins=dict(), force=False):
        # argument list
        if self.eval_value is not None and not force:
            return self.eval_value
        args = list()
        for arg in self.args:
            if hasattr(arg, 'get'):
                args.append(arg.get(ins, force))
            else:
                args.append(arg)
        # kwarg dictionnary
        kwargs = dict()
        for key, item in zip(self.kwargs.items()):
            if hasattr(item, 'get'):
                kwargs.update({key: item.get(ins, force)})
            else:
                kwargs.update({key: item})
        self.eval_value = self._eval(*args, **kwargs)
        return self.eval_value

    def __add__(self, other):
        return nadd(self, other)
    def __ladd__(self, other):
        return nadd(self.other)
    def __sub__(self, other):
        return nsub(self, other)
    def __lsub__(self, other):
        return nsub(self.other)




class SubTuple(Tensor):

    def __init__(self, shape, dtype, index, parent):
        self.parent = parent
        self.index = index
        self.all_dependencies = parent.all_dependencies
        if not hasattr(self, 'name'):
            self.name = ''
        super().__init__(shape=shape, dtype=dtype)

    def get(self, ins=dict(), force=False):
        return self.parent.get(ins, force)[self.index]



class List(list):

    def __init__(self, _eval, shapes, dtypes, args=[], kwargs={}):

        self.args = args
        self.kwargs = kwargs
        all_tensors = list(args) + list(kwargs.values())
        all_dependencies = sum([arg.all_dependencies for arg in all_tensors
                                if hasattr(arg,'all_dependencies')], [])
        self.all_dependencies = list(set(all_tensors+all_dependencies))

        self._eval = _eval
        self.shapes = shapes
        self.dtypes = dtypes
        self.name = ''
        items = [SubTuple(shape, dtype, i, self)
                 for shape, dtype, i in zip(self.shapes, self.dtypes,
                                            range(len(shapes)))]
        self.eval_value = None
        super().__init__(items)

    def get(self, ins=dict(), force=True):
        if self.eval_value is not None and not force:
            return self.eval_value

        # arg list
        args = list()
        for arg in self.args:
            if hasattr(arg, 'get'):
                args.append(arg.get(ins))
            else:
                args.append(arg)

        # kwarg dictionnary
        kwargs = dict()
        for key, item in zip(self.kwargs.items()):
            if hasattr(item, 'get'):
                kwargs.update({key: item.get(ins)})
            else:
                kwargs.update({key: item})

        self.eval_value = self._eval(*args, **kwargs)
        return self.eval_value


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


class Variable(Tensor):

    def __init__(self, value, name='', trainable=True):
        self.trainable = trainable
        self.value = value
        self.all_dependencies = [self]
        self.name = 'Variable: name='+name+', trainable='+str(trainable)
        super().__init__(shape=value.shape, dtype=value.dtype)

    def get(self, args, force=False):
        if self in args:
            return args[self]
        else:
            return self.value


class Placeholder(Tensor):

    def __init__(self, shape, dtype, name=''):
        self.name = 'Placeholder ' + name
        self.all_dependencies = [self]
        super().__init__(shape=shape, dtype=dtype)

    def get(self, args, force=False):
        assert self in args
        return args[self]


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


ncos = NewOp(np.cos)
nadd = NewOp(np.add)
nsum = NewOp(np.sum)
nsub = NewOp(np.subtract)
nmul = NewOp(np.multiply)

key = jax.random.PRNGKey(1)
z = Variable(jax.random.normal(key, shape=(1, 1)))
w = Placeholder((1, 1), 'float32')
y = ncos(nadd(z,w))
cost = nsum(y)

grad = gradients(cost, [w, z])

train = function(w, outputs=[cost],
                 updates={z:z-nmul(0.01,grad[1])})

for i in range(10):
    print(train(NP.ones((1, 1))))

exit()




def cc(wv, a):
#    a = z.args[0]
    cost2 = grad[1].get({w: wv, z:a}, force=True)
    return a-0.01*cost2, cost2

def geta():
    return z.args[0]

fn2 = jax.jit(cc)
geta = jax.jit(geta)

for i in range(10):
    z.value, error = fn2(NP.ones((1, 1)), z.value)
    print(error)


