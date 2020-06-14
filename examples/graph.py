import symjax
import symjax.tensor as T


g = symjax.Graph('model1')
with g:
    learning_rate = T.Variable(T.ones((1,)))
    with symjax.Graph('layer1'):
        W1 = T.Variable(T.zeros((1,)), name='W')
        b1 = T.Variable(T.zeros((1,)), name='b')
    with symjax.Graph('layer2'):
        W2 = T.Variable(T.zeros((1,)), name='W')
        b2 = T.Variable(T.zeros((1,)), name='b')
 
# define an irrelevant loss function involving the parameters
loss =  (W1 + b1 + W2 + b2) * learning_rate

# and a train/update function
train = symjax.function(outputs=loss, updates={W1:W1+1, b1:b1+2, W2: W2+2,
                                               b2: b2+3})

# pretend we train for a while
for i in range(4):
    print(train())

# [0.]
# [8.]
# [16.]
# [24.]

# now say we wanted to reset the variables and retrain, we can do
# either with g, as it contains all the variables
g.reset()
# or we can do
symjax.reset_variables('*')
# or if we wanted to only reset say variables from layer2
symjax.reset_variables('*layer2*')

# now that all has been reset, let's retrain for a while
# pretend we train for a while
for i in range(2):
    print(train())

# [0.]
# [8.]

# now resetting is nice, but we might want to save the model parameters, to
# keep training later or do some other analyses. We can do so as follows:
g.save_variables('model1_saved')
# this would save all variables as they are contained in g. Now say we want to
# only save the second layer variables, if we had saved the graph variables as
# say h we could just do ``h.save('layer1_saved')''
# but while we do not have it, we recall the scope of it, we can thus do
symjax.save_variables('*layer1*', 'layer1_saved')
# and for the entire set of variables just do
symjax.save_variables('*', 'model1_saved')

# now suppose that after training or after resetting
symjax.reset_variables('*')

#one wants to recover the saved weights, one can do
symjax.load_variables('*', 'model1_saved')
# in that case all variables will be reloaded as they were in model1_saved,
# if we used symjax.load('*', 'layer1_saved'), an error would occur as not all
# variables are present in this file, one should instead do
# (in this case, this is redundant as we loaded everything up above)
symjax.load_variables('*layer1*', 'layer1_saved')

# we can now pretend to keep training our model form its saved state
for i in range(2):
    print(train())

# [16.]
# [24.]
