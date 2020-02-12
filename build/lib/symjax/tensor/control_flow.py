import jax.numpy as jnp
import jax.lax as jla
from .base import Op, Tensor, theanofn_to_jaxfn
from .base import Tensor
import numpy



#_scan = Op(jla.scan, name='scan')
def scan(fn, init, *xs):
    newfn = lambda _init, _x, _fn=fn: theanofn_to_jaxfn(_init, *_x, _fn=_fn)
    return _scan(newfn, init, xs)

#_cond = Op(jla.cond, name='cond')

class cond(Op):
    @staticmethod
    def fn(predicate, true_predicate, true_fun, false_predicate, false_fun):
        """ predicate should be a boolean tensor with shape ()
        true_input is the input passed to true_fn that will give the output
        if the predicate evaluates to True, and conversely for False..."""

        # in case the given predicates are not tuples, set them
        if type(true_predicate) != tuple:
            if type(true_predicate) == list:
                true_predicate = tuple(true_predicate)
            else:
                true_predicate = (true_predicate, )
        if type(false_predicate) != tuple:
            if type(false_predicate) == list:
                false_predicate = tuple(false_predicate)
            else:
                false_predicate = (false_predicate, )

        newtruefn = lambda x, _fn=true_fun: theanofn_to_jaxfn(*x, _fn=_fn)
        newfalsefn = lambda x, _fn=false_fun: theanofn_to_jaxfn(*x, _fn=_fn)
        return jla.cond(predicate, true_predicate, newtruefn, false_predicate,
                   newfalsefn)


