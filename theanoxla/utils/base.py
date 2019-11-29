import numpy as np


def train_test_split(X, y, proportion=0.2):
    indices = np.random.permutation(X.shape[0])
    N = int(X.shape[0] * (1 - proportion))
    train_indices = indices[:N]
    test_indices = indices[N:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]



class batchify:
    def __init__(self, *args, batch_size, option='continuous', load_func=None):
        self.args = args
        self.indices = (-batch_size, 0)
        self.option = option
        self.batch_size = batch_size

        if option == 'random_see_all':
            self.permutation = np.random.permutation(args[0].shape[0])
        elif option == 'random':
            self.permutation = np.random.randint(0, args[0].shape[0],
                                                 args[0].shape[0])

        # set up load function
        if load_func is None:
            self.load_func = (None,) * len(args)
        else:
            self.load_func = load_func

        assert np.prod([args[0].shape[0] == arg.shape[0] for arg in args[1:]])
    def __iter__(self):
        return self
    def __next__(self):
        # set current indices
        self.indices = (self.indices[0] + self.batch_size,
                        self.indices[1] + self.batch_size)

        # check if we exhausted the samples
        if self.indices[1] > self.args[0].shape[0]:
            raise StopIteration()
        # else proceed to get the data
        if self.option == 'continuous':
            batch = [arg[self.indices[0]: self.indices[1]] for arg in self.args]
        else:
            perm = self.permutation[self.indices[0]: self.indices[1]]
            batch = [arg[perm] for arg in self.args]
        # now check if needs to apply a load func
        for i, load_func in enumerate(self.load_func):
            if load_func is not None:
                batch[i] = load_func(batch[i])
        if len(self.args) == 1:
            return batch[0]
        else:
            return tuple(batch)



