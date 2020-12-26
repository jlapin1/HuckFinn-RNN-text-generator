# -*- coding: utf-8 -*-
"""
Attempting with handwritten GRU cell
#ISSUE#
GradientTape had trouble calculating gradients when working with the o output
from fprop
Must have something to do with the fact that it's values were being set by an
assign operation and NOT A DIRECT CALCULATION
#FIX#
Works when calculating loss on the fly as you propagate through the RNN or
collecting the outputs in a list (not in a preallocated variable)
- Also don't use assign_add with the "loss" variable
"""

import numpy as np
import sys
import time
import tensorflow as tf

# How to work with keras.layers.GRUCell
# gru = tf.keras.layers.GRUCell(hd)
# gru(inp, s_1)
## inp and s_1 need to have a dummy dimension for axis 0

class RNN():
    def __init__(self, vocab_size, embedding_size, hidden_dim, output_size, bptt_trunc=96):
        self.vs = vocab_size
        self.es = embedding_size
        self.hd = hidden_dim
        self.os = output_size
        self.bptt_trunc = bptt_trunc
        
#        self.E = tf.Variable(tf.random_uniform((self.es, self.vs), -self.es**-0.5, self.es**-0.5))
#        self.Uz = tf.Variable(tf.random.uniform((self.hd, self.es), -self.es**-0.5, self.es**-0.5))
#        self.Wz = tf.Variable(tf.random.uniform((self.hd, self.hd), -self.hd**-0.5, self.hd**-0.5))
#        self.bz = tf.Variable(tf.zeros((self.hd, 1), dtype=tf.float32))
#        self.Ur = tf.Variable(tf.random.uniform((self.hd, self.es), -self.es**-0.5, self.es**-0.5))
#        self.Wr = tf.Variable(tf.random.uniform((self.hd, self.hd), -self.hd**-0.5, self.hd**-0.5))
#        self.br = tf.Variable(tf.zeros((self.hd, 1), dtype=tf.float32))
#        self.Uh = tf.Variable(tf.random.uniform((self.hd, self.es), -self.es**-0.5, self.es**-0.5))
#        self.Wh = tf.Variable(tf.random.uniform((self.hd, self.hd), -self.hd**-0.5, self.hd**-0.5))
#        self.bh = tf.Variable(tf.zeros((self.hd, 1), dtype=tf.float32))
#        self.V = tf.Variable(tf.random.uniform((self.os, self.hd), -self.hd**-0.5, self.hd**-0.5))
#        self.c = tf.Variable(tf.zeros((self.os, 1), dtype=tf.float32))
        
        E = np.random.uniform( -self.vs**-0.5, self.vs**-0.5, (self.es, self.vs))
        Uz = np.random.uniform(-self.es**-0.5, self.es**-0.5, (self.hd, self.es))
        Ur = np.random.uniform(-self.es**-0.5, self.es**-0.5, (self.hd, self.es))
        Uh = np.random.uniform(-self.es**-0.5, self.es**-0.5, (self.hd, self.es))
        Wz = np.random.uniform(-self.hd**-0.5, self.hd**-0.5, (self.hd, self.hd))
        Wr = np.random.uniform(-self.hd**-0.5, self.hd**-0.5, (self.hd, self.hd))
        Wh = np.random.uniform(-self.hd**-0.5, self.hd**-0.5, (self.hd, self.hd))
        bz = np.random.uniform(-self.hd**-0.5, self.hd**-0.5, (self.hd,1))
        br = np.random.uniform(-self.hd**-0.5, self.hd**-0.5, (self.hd,1))
        bh = np.random.uniform(-self.hd**-0.5, self.hd**-0.5, (self.hd,1))
        V = np.random.uniform( -self.hd**-0.5, self.hd**-0.5, (self.os, self.hd))
        c = np.random.uniform( -self.hd**-0.5, self.hd**-0.5, (self.os,1))
        self.E = tf.Variable(E, dtype=tf.float32)
        self.Uz = tf.Variable(Uz, dtype=tf.float32)
        self.Wz = tf.Variable(Wz, dtype=tf.float32)
        self.bz = tf.Variable(bz, dtype=tf.float32)
        self.Ur = tf.Variable(Ur, dtype=tf.float32)
        self.Wr = tf.Variable(Wr, dtype=tf.float32)
        self.br = tf.Variable(br, dtype=tf.float32)
        self.Uh = tf.Variable(Uh, dtype=tf.float32)
        self.Wh = tf.Variable(Wh, dtype=tf.float32)
        self.bh = tf.Variable(bh, dtype=tf.float32)
        self.V = tf.Variable(V, dtype=tf.float32)
        self.c = tf.Variable(c, dtype=tf.float32)
        
        self.var = [self.E, self.Uz, self.Wz, self.bz, self.Ur, self.Wr,
                    self.br, self.Uh, self.Wh, self.bh, self.V, self.c]
    def fprop(self, x):
        bs = x.shape[0]
        T = x.shape[1]
        o = []
        s = tf.Variable(tf.zeros((T, self.hd, bs), dtype=tf.float32))
#        xe = tf.Variable(tf.zeros((self.es, bs), dtype=tf.float32))
        for m in range(T):
            # Grab the word vector
#            for n in range(bs):
#                xe[:,n].assign(self.E[:,x[n,m]])
            xe = tf.transpose(tf.nn.embedding_lookup(tf.transpose(self.E), x[:,m]))
            # GRU layer
            S = self.gru(xe, s[m-1])
            s[m].assign(S)
            # Softmax output
            o.append( tf.matmul(self.V, S) + self.c ) 
        return o, s
    
    def gru(self, xe, s):
        
        # GRU layer
        z = tf.nn.sigmoid( tf.matmul(self.Uz, xe) + tf.matmul(self.Wz, s) + self.bz )
        
        r = tf.nn.sigmoid( tf.matmul(self.Ur, xe) + tf.matmul(self.Wr, s) + self.br )
        
        h = tf.nn.tanh( tf.matmul(self.Uh, xe) + tf.matmul(self.Wh, s*r) + self.bh )
        
        st = (1-z)*s + z*h
        
        return st
    
    def calctotloss(self, x, y):
        # Using loss.assign_add(softmax) rendered all gradients as None
        bs = y.shape[0]
        T = y.shape[1]
        o,s = self.fprop(x)
        loss = tf.Variable(tf.zeros(bs))
        for m in range(T):
            yy = tf.one_hot(y[:,m], depth=self.vs, axis=0)
            loss = loss + (tf.nn.softmax_cross_entropy_with_logits(labels=yy, logits=o[m], axis=0))
        return tf.reduce_mean(loss) # taking the mean of the batch size

def randchoice(arr):
    plc = 0
    rnd = np.random.rand()
    for m in range(len(arr)):
        plc += arr[m]
        if plc>rnd:
            return m
def gentext(model, start, SL=100):
    inp = np.array([[char2ind[m] for m in start]])
    o,s = model.fprop(inp)
    inp = randchoice(np.squeeze(tf.nn.softmax(o[-1], axis=0).numpy()))
    st = s[-1]
    txt = [vocab[inp]]
    for m in range(SL):
        xe = tf.expand_dims(model.E[:,inp], axis=-1)
        st = model.gru(xe, st)
        inp = randchoice(np.squeeze(tf.nn.softmax(tf.matmul(model.V, st) + model.c, axis=0).numpy()))
        txt.append(vocab[inp])
    return start+"".join(txt)

def adam(model, inps, targs, epochs, bs, a=0.01, b1=0.9, b2=0.999, lam=0):
    
    # Allocate for Adam variables
    v = []
    vcorr = []
    s = []
    scorr = []
    for m in model.var:
        v.append(tf.Variable(tf.zeros_like(m)))
        vcorr.append(tf.Variable(tf.zeros_like(m)))
        s.append(tf.Variable(tf.zeros_like(m)))
        scorr.append(tf.Variable(tf.zeros_like(m)))
    
    batches = int(inps.shape[0])//bs + 1
    
    updates = 0
    for l in range(epochs):
        
        train_loss = 0
        split = time.time()
        for m in range(batches):
            sys.stdout.write("\rBatch %d/%d %.3f s "%(m+1, batches, time.time()-split))
            split = time.time()
            # Starting and ending batch indices
            first = m*bs
            last = (m+1)*bs if m!=batches-1 else int(inps.shape[0]) # final batch is smaller than others
            # Measure loss and gradients
            with tf.GradientTape() as t:
                loss = model.calctotloss(inps[first:last], targs[first:last])
            train_loss += loss*(last-first)
            grads = t.gradient(loss, model.var)
            # Update variables for batch m
            updates += 1
            for n,o in enumerate(grads):
                v[n] = b1*v[n] + (1-b1)*o # recursive
                vcorr[n] = tf.divide(v[n], 1-b1**updates)
                s[n] = b2*s[n] + (1-b2)*tf.pow(o, 2.0) # recursive
                scorr[n] = tf.divide(s[n], 1-b2**updates)
                model.var[n] = model.var[n].assign_sub( a*(tf.divide(vcorr[n], (tf.pow(scorr[n], 0.5)+1E-8))+lam*model.var[n]) )
        # Print out results for epoch
        print("Epoch: %d; Train Loss: %.3f"%(l,train_loss/int(inps.shape[0])))
        

# Important architecture parameters
sl = 100 # how long a training sentence is
es = 256 # embedding size
hd = 512 # number of nodes in RNN s-layer
#np.random.seed(10)
#tf.random.set_random_seed(10)
############################
# Setting up training data #
############################
f = open("./huckfinn.txt", "r")
story = f.read()
f.close
length = len(story)
vocab = np.unique(list(story))
vocab_size = len(vocab) # this will set input and output size of the RNN 
char2ind = {t:s for s,t in enumerate(vocab)}
# - Training sentences
inps = []
targs = []
for i in range(length//(sl+1)):
    start = i*(sl+1)
    end = i*(sl+1) + sl+1
    slic = story[start:end]
    inps.append([char2ind[j] for j in slic[:sl]])
    targs.append([char2ind[j] for j in slic[1:]])
inps = np.array(inps);targs = np.array(targs)
##############################
# End setting up training ...#
##############################
model = RNN(vocab_size, es, hd, vocab_size)
#with tf.GradientTape() as t:
#    loss = model.calctotloss(inps[:50], targs[:50])
#mg = t.gradient(loss, model.var)
#import sys;sys.exit()
##loss = rnn.calctotloss(inps[0], targs[0])

epochs = 10
batch_size = 100
alpha = 0.01

batches = inps.shape[0] // 100 + 1

#for l in range(epochs):
#    globe = [tf.Variable(tf.zeros_like(m)) for m in model.var]
#    loss_tot = 0
#    for m in range(batches):
#        first = m*batch_size
#        last = (m+1)*batch_size if m<batches-1 else inps.shape[0]
#        with tf.GradientTape() as t:
#            loss = model.calctotloss(inps[first:last], targs[first:last])
#        grads = t.gradient(loss, model.var)
#        for n,o in enumerate(grads):
#            globe[n].assign_add(o)
#        loss_tot += loss/len(inps[0])
#        sys.stdout.write("\rBatch %d/%d "%(m+1, batches))
#    for m,n in enumerate(globe):
#        model.var[m].assign_sub(alpha*(n/batches))
#    
#    print("Epoch: %d; Loss: %f"%(l, loss_tot/batches))

adam(model, inps, targs, epochs, batch_size, lam=0)