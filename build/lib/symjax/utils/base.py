import numpy as np
from multiprocessing import Pool, Queue, Lock, Process


def train_test_split(*args, train_size=0.8, stratify=None, seed=None):
    if stratify is not None:
        train_indices = list()
        test_indices = list()
        for c in set(list(stratify)):
            c_indices = np.where(stratify == c)[0]
            np.random.RandomState(seed=seed).shuffle(c_indices)
            if train_size > 1:
                cutoff = train_size
            else:
                cutoff = int(len(c_indices)*train_size)
            train_indices.append(c_indices[:cutoff])
            test_indices.append(c_indices[cutoff:])
        train_indices = np.concatenate(train_indices, 0)
        test_indices = np.concatenate(test_indices, 0)
    else:
        indices = np.random.RandomState(seed=seed).permutation(len(args[0]))
        if train_size > 1:
            cutoff = train_size
        else:
            cutoff = int(len(args[0])*train_size)
        train_indices = indices[:cutoff]
        test_indices = indices[cutoff:]
    output = sum([[arg[train_indices], arg[test_indices]] for arg in args], [])
    return output

class batchify:

    def __init__(self, *args, batch_size, option='continuous', load_func=None,
                 extra_process=0):
        """

        Parameters
        ----------

        load_func: None or list of func
            same length as the number of args. A function is called on a single
            datum and it can be used to apply some normalization but its main
            goal is to load files if the args were list of filenames

        extra_processes: int (optional)
            if there is no load_func then extra process is useless"""

        self.args = args
        self.start_index = 0
        self.option = option
        self.batch_size = batch_size
        self.extra_process = extra_process
        self.terminate = False

        if option == 'random_see_all':
            self.permutation = np.random.permutation(args[0].shape[0])
        elif option == 'random':
            self.permutation = np.random.randint(0, args[0].shape[0],
                                                 args[0].shape[0])
        else:
            self.permutation = np.arange(args[0].shape[0])

        # set up load function
        if load_func is None:
            self.load_func = (None,) * len(args)
        else:
            self.load_func = []
            for i in range(len(load_func)):
                if load_func[i] is None:
                    self.load_func.append(None)
                elif extra_process == 0:
                    def fn(args, queue, f=load_func[i]):
                        result = np.asarray([f(arg) for arg in args])
                        queue.put(result)
                    self.load_func.append(fn)
                else:
                    def fn(lock, data, queue, f=load_func[i]):
                        lock.acquire()
                        with Pool(processes=extra_process) as pool:
                            result = pool.map(f, data)
                        queue.put(np.asarray(result))
                        lock.release()
                    self.load_func.append(fn)
        assert np.prod([args[0].shape[0] == arg.shape[0] for arg in args[1:]])

        self.queues = [Queue() for f in self.load_func]
        self.locks = [Lock() for f in self.load_func]

        # launch the first batch straight away
        batch = self.get_batch()
        self.launch_process(batch)

    def chunk(self, items, n):
        for i in range(0, len(items), n):
            yield items[i:i + n]

    def launch_process(self, batch):

        for b, lock, load_func, queue in zip(batch, self.locks,
                                             self.load_func, self.queues):
            if load_func is None:
                queue.put(b)
            elif load_func is not None and self.extra_process == 0:
                load_func(b, queue)
            elif load_func is not None and self.extra_process > 0:
                # first chunk for each process and launch a process
                p = Process(target=load_func, args=(lock, b, queue))
                p.start()


    def __iter__(self):
        return self

    def get_batch(self):
        indices = (self.start_index, self.start_index + self.batch_size)

        # check if we exhausted the samples
        if indices[1] > self.args[0].shape[0]:
            raise StopIteration()

        # proceed to get the data
        perm = self.permutation[indices[0]: indices[1]]
        batch = [arg[perm] for arg in self.args]
        return batch

    def __next__(self):

        if self.terminate:
            raise StopIteration()

        # we prepare the next batch if possible
        try:
            self.start_index += self.batch_size
            batch = self.get_batch()
            self.launch_process(batch)
        except StopIteration:
            self.terminate = True

        if len(self.queues) == 1:
            return self.queues[0].get()
        else:
            return tuple([queue.get() for queue in self.queues])

    def apply_func(self, batch):
        # now check if needs to apply a load func
        for i, load_func in enumerate(self.load_func):
            if load_func is not None:
                batch[i] = load_func(batch[i])



def vq_to_boundary(states,N,M,duplicate=1):
    #the following should take as input a collection of points
    #in the input space and return a list of binary states, each 
    #element of the list is for 1 specific layer and it is a 2D array
    if states.ndim>1:
        states = vq_to_values(states)
    partitioning  = values_to_boundary(states.reshape((N,M)).astype('float32'),duplicate)
    return partitioning

def values_to_boundary(x,duplicate=0):
    #compute each directional (one step) derivative as a boolean mask
    #representing jump from one region to another and add them (boolean still)
    g_vertical   = np.greater(np.pad(np.abs(x[1:]-x[:-1]),((1,0),(0,0)),'constant'),0)
    g_horizontal = np.greater(np.pad(np.abs(x[:,1:]-x[:,:-1]),[[0,0],[1,0]],'constant'),0)
    g_diagonaldo = np.greater(np.pad(np.abs(x[1:,1:]-x[:-1,:-1]),[[1,0],[1,0]],'constant'),0)
    g_diagonalup = np.greater(np.pad(np.abs(x[:-1:,1:]-x[1:,:-1]),[[1,0],[1,0]],'constant'),0)
    overall      = g_vertical+g_horizontal+g_diagonaldo+g_diagonalup
    if duplicate>0:
        overall                 = np.stack([np.roll(overall,k,1) for k in range(duplicate+1)]\
                                    +[np.roll(overall,k,0) for k in range(duplicate+1)]).sum(0)
        overall[:duplicate]    *= 0
        overall[-duplicate:]   *= 0
        overall[:,:duplicate]  *= 0
        overall[:,-duplicate:] *= 0
    return np.greater(overall,0).astype('float32')



def vq_to_values(states, vq_to_value_dict=None, return_dict=False):
    """Given binary masks obtained for example from the ReLU activation, 
    convert the binary vectors to a single float scalar, the same scalar for
    the same masks. This allows to drastically reduce the memory overhead to
    keep track of a large number of masks. The mapping mask -> real is kept
    inside a dict and thus allow to go from one to another. Thus given a large
    collection of masks, one ends up with a large collection of scalars and
    a mapping mask <-> real only for unique masks and thus allow reduced memory
    requirements.

    Parameters
    ----------

    states : bool matrix
        the matrix of shape (#samples,#binary values). Thus if using a deep net
        the masks of ReLU for all the layers must first be flattened prior
        using this function.

    vq_to_value_dict : dict
        optional dict containing an already built mask <-> real mapping which
        should be used and updated given the states value.

    return_dict : bool
        if the update/created dict should be return as part of the output

    Returns
    -------

    values : scalar vector
        a vector of length #samples in which each entry is the scalar
        representation of the corresponding mask from states

    vq_to_value_dict : dict (optional)
        the newly created or updated dict mapping mask to value and vice-versa.


    """
    if vq_to_value_dict is None:
        vq_to_value_dict = dict()
    if 'count' not in vq_to_value_dict:
        vq_to_value_dict['count']=0
    values = np.zeros(states.shape[0])
    for i,state in enumerate(states):
        str_s = ''.join(state.astype('uint8').astype('str'))
        if(str_s not in vq_to_value_dict):
            vq_to_value_dict[str_s]   = vq_to_value_dict['count']
            vq_to_value_dict['count']+= 0.001
        values[i] = vq_to_value_dict[str_s]
    return values

