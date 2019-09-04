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
