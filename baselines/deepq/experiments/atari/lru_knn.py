import numpy as np
from sklearn.neighbors import BallTree,KDTree
import os

import gc



#each action -> a lru_knn buffer
class LRU_KNN:
    def __init__(self, capacity, z_dim, env_name):
        self.env_name = env_name
        self.capacity = capacity
        self.states = np.empty((capacity, z_dim), dtype = np.float32)
        self.q_values_decay = np.zeros(capacity)
        self.lru = np.zeros(capacity)
        self.curr_capacity = 0
        self.tm = 0.0
        self.tree = None
        self.addnum = 0
        self.buildnum = 256
        self.buildnum_max = 256
        self.bufpath = './buffer/%s'%self.env_name
        self.build_tree_times = 0
        self.build_tree = False

    def load(self, action):
        try:
            assert(os.path.exists(self.bufpath))
            lru = np.load(os.path.join(self.bufpath, 'lru_%d.npy'%action))
            cap = lru.shape[0]
            self.curr_capacity = cap
            self.tm = np.max(lru) + 0.01
            self.buildnum = self.buildnum_max

            self.states[:cap] = np.load(os.path.join(self.bufpath, 'states_%d.npy'%action))
            self.q_values_decay[:cap] = np.load(os.path.join(self.bufpath, 'q_values_decay_%d.npy'%action))
            self.lru[:cap] = lru
            self.tree = KDTree(self.states[:self.curr_capacity])
            print ("load %d-th buffer success, cap=%d" % (action, cap))
        except:
            print ("load %d-th buffer failed" % action)

    def save(self, action):
        if not os.path.exists('buffer'):
            os.makedirs('buffer')
        if not os.path.exists(self.bufpath):
            os.makedirs(self.bufpath)
        np.save(os.path.join(self.bufpath, 'states_%d'%action), self.states[:self.curr_capacity])
        np.save(os.path.join(self.bufpath, 'q_values_decay_%d'%action), self.q_values_decay[:self.curr_capacity])
        np.save(os.path.join(self.bufpath, 'lru_%d'%action), self.lru[:self.curr_capacity])

    def peek(self, key, value_decay, modify):
        if self.curr_capacity==0 or self.build_tree == False:
            return None


        dist, ind = self.tree.query([key], k=1)
        ind = ind[0][0]

        #if self.states[ind] == key:
        #if np.allclose(self.states[ind], key):
        if np.allclose(self.states[ind], key, atol=1e-08):
            self.lru[ind] = self.tm
            self.tm +=0.01
            if modify:
                if value_decay > self.q_values_decay[ind]:
                    self.q_values_decay[ind] = value_decay
            return self.q_values_decay[ind]
        #print self.states[ind], key

        return None

    def knn_value(self, key, knn):
        knn = min(self.curr_capacity, knn)
        if self.curr_capacity==0 or self.build_tree == False:
            return 0.0, 0.0

        dist, ind = self.tree.query([key], k=knn)

        value = 0.0
        value_decay = 0.0
        for index in ind[0]:
            value_decay += self.q_values_decay[index]
            self.lru[index] = self.tm
            self.tm+=0.01

        q_decay = value_decay / knn

        return q_decay

    def add(self, key, value_decay):
        if self.curr_capacity >= self.capacity:
            # find the LRU entry
            old_index = np.argmin(self.lru)
            self.states[old_index] = key
            self.q_values_decay[old_index] = value_decay
            self.lru[old_index] = self.tm
        else:
            self.states[self.curr_capacity] = key
            self.q_values_decay[self.curr_capacity] = value_decay
            self.lru[self.curr_capacity] = self.tm
            self.curr_capacity+=1
        self.tm += 0.01
        #self.addnum += 1
        #if self.addnum % self.buildnum == 0:
        #    self.addnum = 0
        #    self.buildnum = min(self.buildnum * 2, self.buildnum_max)
        #    del self.tree
        #    self.tree = KDTree(self.states[:self.curr_capacity])
        #    self.build_tree_times += 1
        #if self.curr_capacity < self.buildnum:
        #    del self.tree
        #    self.tree = KDTree(self.states[:self.curr_capacity])
        #    self.build_tree_times += 1

        #if self.build_tree_times == 50:
        #    self.build_tree_times = 0
        #    gc.collect()

    def update_kdtree(self):
        if self.build_tree:
            del self.tree
        self.tree = KDTree(self.states[:self.curr_capacity])
        self.build_tree = True
        self.build_tree_times += 1
        if self.build_tree_times == 50:
            self.build_tree_times = 0
            gc.collect()

