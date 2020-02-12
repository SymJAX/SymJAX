#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


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


