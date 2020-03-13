import jax.numpy as jnp
import jax.lax as jla
from .base import Op, Tensor, symjax_to_jax_fn
from .base import Tensor, jax_wrap
import numpy




def _scan(fn, init, xs, constants=(), length=None):
    def new_fn(*args):
        return fn(*args, *constants)
    return jla.scan(new_fn, init, xs, length)
        
_scan2= jax_wrap(_scan)

def scan(fn, init, xs, constants=(), length=None):
    newfn = symjax_to_jax_fn(fn)
    return _scan2(newfn, init, xs, constants, length)

scan.__doc__ = jla.scan.__doc__

#_cond = Op(jla.cond, name='cond')

def _cond(predicate, true_predicate, true_fun, false_predicate, false_fun):
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

        newtruefn = lambda x, _fn=true_fun: symjax_to_jax_fn(*x, _fn=_fn)
        newfalsefn = lambda x, _fn=false_fun: symjax_to_jax_fn(*x, _fn=_fn)
        return jla.cond(predicate, true_predicate, newtruefn, false_predicate,
                   newfalsefn)

cond = jax_wrap(_cond, doc_func=_cond)

##class cond(Op):
##    @staticmethod
##    def fn(predicate, true_predicate, true_fun, false_predicate, false_fun):
##        """ predicate should be a boolean tensor with shape ()
##        true_input is the input passed to true_fn that will give the output
##        if the predicate evaluates to True, and conversely for False..."""
##
##        # in case the given predicates are not tuples, set them
##        if type(true_predicate) != tuple:
##            if type(true_predicate) == list:
##                true_predicate = tuple(true_predicate)
##            else:
##                true_predicate = (true_predicate, )
##        if type(false_predicate) != tuple:
##            if type(false_predicate) == list:
##                false_predicate = tuple(false_predicate)
##            else:
##                false_predicate = (false_predicate, )
##
##        newtruefn = lambda x, _fn=true_fun: symjax_to_jax_fn(*x, _fn=_fn)
##        newfalsefn = lambda x, _fn=false_fun: symjax_to_jax_fn(*x, _fn=_fn)
##        return jla.cond(predicate, true_predicate, newtruefn, false_predicate,
##                   newfalsefn)
##
#
