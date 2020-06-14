import symjax
import symjax.tensor as T

# scope/graph naming and accessing

value1 = T.Variable(T.ones((1,)))
value2 = T.Variable(T.zeros((1,)))

g = symjax.Graph('special')
with g:
    value3 = T.Variable(T.zeros((1,)))
    value4 = T.Variable(T.zeros((1,)))
    result = value3 + value4

    h = symjax.Graph('inversion')
    with h:
        value5 = T.Variable(T.zeros((1,)))
        value6 = T.Variable(T.zeros((1,)))
        value7 = T.Variable(T.zeros((1,)), name='w')


print(g.variables)
# {'unnamed_variable': Variable(name=unnamed_variable, shape=(1,), dtype=float32, trainable=True, scope=/special/),
#  'unnamed_variable_1': Variable(name=unnamed_variable_1, shape=(1,), dtype=float32, trainable=True, scope=/special/)}

print(h.variables)
#{'unnamed_variable': Variable(name=unnamed_variable, shape=(1,), dtype=float32, trainable=True, scope=/special/inversion/),
# 'unnamed_variable_1': Variable(name=unnamed_variable_1, shape=(1,), dtype=float32, trainable=True, scope=/special/inversion/),
# 'w': Variable(name=w, shape=(1,), dtype=float32, trainable=True, scope=/special/inversion/)}

print(h.variable('w'))
# Variable(name=w, shape=(1,), dtype=float32, trainable=True, scope=/special/inversion/)

# now suppose that we did not hold the value for the graph g/h, we can still
# recover a variable based on the name AND the scope

print(symjax.get_variables('/special/inversion/w'))
# Variable(name=w, shape=(1,), dtype=float32, trainable=True, scope=/special/inversion/)

# now if the exact scope name is not know, it is possible to use smart indexing
# for example suppose we do not remember, then we can get all variables named
# 'w' among scopes

print(symjax.get_variables('*/w'))
# Variable(name=w, shape=(1,), dtype=float32, trainable=True, scope=/special/inversion/)

# if only part of the scope is known, all the variables of a given scope can
# be retreived

print(symjax.get_variables('/special/*'))
# [Variable(name=unnamed_variable, shape=(1,), dtype=float32, trainable=True, scope=/special/),
#  Variable(name=unnamed_variable_1, shape=(1,), dtype=float32, trainable=True, scope=/special/),
#  Variable(name=unnamed_variable, shape=(1,), dtype=float32, trainable=True, scope=/special/inversion/),
#  Variable(name=unnamed_variable_1, shape=(1,), dtype=float32, trainable=True, scope=/special/inversion/),
#  Variable(name=w, shape=(1,), dtype=float32, trainable=True, scope=/special/inversion/)]

print(symjax.get_ops('*add'))
# Op(name=add, shape=(1,), dtype=float32, scope=/special/)
