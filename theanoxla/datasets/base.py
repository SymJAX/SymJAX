#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from ..utils import case


class BatchPeriod(object):
    def __init__(self, batch_period, offset=0):
        self.batch_period = batch_period
        self.offset = offset
    def __call__(self, batch, epoch):
        if (batch + self.offset)%self.batch_period == 0:
            return True
        return False

class BatchIterator(dict):
    """
    Parameters
    ----------

    options : dict
        A dictionnary describing for each set of a :class:`sknet.Dataset`
        object how to read (sample) batches from it and the batch_size.
        The options are  ``"continuous"``, ``"random"``, ``"random_all"``.
        For example::

            iterator = BatchIterator({'train_set':'random',
                                     'test_set':continuous})
            # returns the array of indices that are looped through
            iterator['train_set']
            # returns the tf.Variable holding the current batch indices
            iterator.train_set


        For specific applications, such as semi-supervised learning, it might
        be useful to simultaneously extract patches from two different sets as
        a single batch. If those two dataset have same number of samples,
        it is straightforward to combine them. But if their length differ or
        if needs be to have a random (but same) batch sampling, then do as
        follows::

            iterator = BatchIterator({'sup_train_set,
                                     unsup_train_set':'random',
                                     'test_set':continuous})
            # returns the array of indices that are looped through
            iterator['train_set']
            # returns the tf.Variable holding the current batch indices
            iterator.train_set


    """
    def __init__(self, options, dataset):

        self.batch_counter = dict()
        self.sets = list(options.keys())
        self.dataset = dataset
        self.options = options
        self.p = dict()
        # extract the sets and then check for joint ones and separate them
        print('BatchIterator initialized with ')
        with tf.variable_scope("iterator"):
            for s, v in options.items():
                print('\t\t{}:{}'.format(s, v))
            self.indices = tf.placeholder_with_default(tf.ones(dataset.batch_size, dtype=tf.int64),
                                          shape=(dataset.batch_size,),
                                          name='indices')

            self.set = tf.placeholder(tf.string, name='set')

    def reset(self, s):
        """reset the indices to loop through for a specific set

        Parameters
        ----------

        s : str
            one of the set, the one to have its indices reset

        """
        self.batch_counter[s] = -1
        N = self.dataset.N(s)
        p = self.p[s] if s in self.p.keys() else None
        if self.options[s] == "continuous":
            indices = np.asarray(range(N)).astype('int64')
        elif self.options[s] == "random":
            indices = np.random.choice(N, (N,), replace=True, p=p)
        elif self.options[s] == "random_no_replace":
            indices = np.random.choice(N, (N,), replace=False, p=p)
        elif self.options[s] == "random_see_all":
            indices = np.random.permutation(N)
        self[s] = indices

    def next(self, s):
        if self.dataset.N_BATCH(s) == (self.batch_counter[s]+1):
            self.reset(s)
            return False
        self.batch_counter[s] += 1
        batch = range((self.batch_counter[s])*self.dataset.batch_size,
                      (self.batch_counter[s]+1)*self.dataset.batch_size)
        return {self.indices: self[s][batch]}


class Dataset(dict):
    def __init__(self, **args):
        super().__init__()
        self.batch_size = None
        self.__dict__.update(args)

    def cast(self, var_name, dtype):
        """cast the dataset variable to a specific type for all the sets
        Args:

        var_name: str
            the name of the variable to cast

        dtype: type or str
            the type to cast the variable into

        """
        sets = self.sets_(var_name)
        for s in sets:
            self[var_name+'/'+s] = self[var_name+'/'+s].astype(dtype)

    @property
    def init_dict(self):
        """return the list of couple with the tensorflow
        variable and the dataset variable to be used to initialized the
        tf one, this has to be run when initializing the graph,
        the TF variables are created before hand by create_placeholders"""
        alls = [(self[v+'/'+s+'/placeholder'], self[v+'/'+s])
                for v in self.variables for s in self.sets_(v)]
        return dict(alls)

    def create_placeholders(self, batch_size, options, device="/cpu:0"):
        # Many settings are put in int64 for GPU compatibility with tf
        self.batch_size = batch_size
        with tf.device(device):
            self.iterator = BatchIterator(options, self)
            with tf.variable_scope("dataset"):
                # create the tensorflow placeholders and variables that
                # will hold the values of the sets and variables as part of
                for varn in list(self.keys()):
                    # ensure that there is not already a member with this name
                    # as a method of the class (dict)
                    assert(varn not in self.__dict__)
                    type_ = str(self[varn].dtype)
                    name1 = varn+'/placeholder'
                    self[name1] = tf.placeholder(type_, shape=self[varn].
                                                 shape, name=varn)
                    name2 = varn+'/Variable'
                    self[name2] = tf.Variable(self[name1], trainable=False,
                                              name=varn)

                for v in self.variables:
                    pairs = list()
                    for s in self.sets_(v):
                        name = v+'/'+s+'/Variable'
                        indices = tf.mod(self.iterator.indices,
                                         self[v+'/'+s].shape[0])
                        #if self[v+'/'+s].dtype == 'int32':
                        #    batch = tf.gather(tf.cast(self[name], tf.float32),
                        #                      indices)
                        #    batch = tf.cast(batch, tf.int32)
                        #else:
                        batch = tf.gather(self[name], indices)
                        # batch = tf.placeholder_with_default(batch, batch.shape)
                        pairs.append((tf.constant(s), batch))
                    self.__dict__[v] = tf.placeholder_with_default(case(self.iterator.set, pairs), batch.shape)

    def split_set(self, set_, new_set_, ratio, stratify=None, seed=None):
        assert new_set_ not in self.sets
        variables = self.variables
        if stratify is not None:
            exit()
            assert(len(y) > 1)
            train_indices = list()
            valid_indices = list()
            for c in set(y):
                c_indices = np.where(y==c)[0]
                np.random.shuffle(c_indices)
                limit = int(len(c_indices)*test_ratio)
                train_indices.append(c_indices[limit:])
                valid_indices.append(c_indices[:limit])
            train_indices = np.concatenate(train_indices, 0)
            valid_indices = np.concatenate(valid_indices, 0)
            #
            self["valid_set"]=[s[valid_indices] if isinstance(s, np.ndarray)
                               else [s[i] for i in valid_indices]
                               for s in self["train_set"]]
            #
            self["train_set"]=[s[train_indices] if isinstance(s, np.ndarray)
                        else [s[i] for i in train_indices]
                        for s in self["train_set"]]
        else:
            indices = np.random.RandomState(seed).permutation(self.N(set_))
            if ratio < 1:
                N = int(self.N(set_)*ratio)
            else:
                N = ratio
            for var in self.variables:
                name = var+'/'+set_
                new_name = var+'/'+new_set_
                self[name], self[new_name]=\
                               self[name][indices[N:]],self[name][indices[:N]]

    @property
    def sets(self):
        return sorted(list(np.unique([k.split('/')[1] for k in self])))

    def sets_(self, varn):
        return sorted(list(np.unique([k.split('/')[1]
			          for k in self if varn in k])))

    @property
    def variables(self):
        return sorted(list(np.unique([k.split('/')[0] for k in self])))

    def N(self, context):
        """returns the length of a given set. If multiple variables
        belong to the given set, the largest length is returned
        """
        length = 0
        for var in self:
            if context in var and len(var.split('/')) == 2:
                length = np.maximum(length, len(self[var]))
        return length

    def N_BATCH(self, context):
        return self.N(context)//self.batch_size

    def shape(self, var):
        set_ = self.sets[0]
        return self[var+'/'+set_].shape

    def datum_shape(self, var):
        return self.shape(var)[1:]

