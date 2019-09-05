import jax
import jax.numpy as np
import numpy as NP

def isdep(item):
    return isinstance(item, Variable) or isinstance(item, Placeholder)


class Op:

    def __init__(self, fn, name=''):
        """function that produces a new Op based on a given function"""
        self.fn = fn
        self.fn.__name__ = name

    def __call__(self, *args, shape=None, dtype=None, **kwargs):
        return Tensor(_eval=self.fn, args=args, kwargs=kwargs, shape=shape,
                      dtype=dtype)


class Tensor:

    def __init__(self, _eval=None, args=[], kwargs={}, shape=None,
                 dtype=None, name=''):

        self.args, self.kwargs = args, kwargs
        self.name = name
        self._eval = _eval
        self.eval_value = None

        # set shape and dtype
        if shape is None or dtype is None:
            tree = jax.eval_shape(_eval, *self.args, **self.kwargs)
            self.shape, self.dtype = tree.shape, tree.dtype
        else:
            self.shape = shape
            self.dtype = dtype

        # set dependencies
        all_tensors = list(args) + list(kwargs.values())
        all_dependencies = sum([arg.all_dependencies for arg in all_tensors
                                if hasattr(arg,'all_dependencies')], [])
        self.all_dependencies = list(set(all_tensors+all_dependencies))

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


class Variable(Tensor):

    def __init__(self, value, name='', trainable=True):
        self.trainable = trainable
        print(value)
        if not isinstance(value, jax.interpreters.xla.DeviceArray):
            self.value = np.array(value)
            if NP.isscalar(value):
                shape = ()
                dtype = type(value)
            else:
                shape = NP.shape(value)
                dtype = NP.dtype(value)
        else:
            self.value = value
            shape = value.shape
            dtype = value.dtype
        name = 'Variable: name='+name+', trainable='+str(trainable)
        super().__init__(shape=shape, dtype=dtype, name=name)

    def get(self, args, force=False):
        if self in args:
            return args[self]
        else:
            return self.value


class Placeholder(Tensor):

    def __init__(self, shape, dtype, name=''):
        name = 'Placeholder ' + name
        super().__init__(shape=shape, dtype=dtype, name=name)

    def get(self, args, force=False):
        assert self in args
        return args[self]
