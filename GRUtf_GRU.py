# -*- coding: utf-8 -*-
"""
Replaced my handwritten gru cell with tf.keras.layers.GRU, whose function call
evalutates all time steps

# of Dimensions of two matrices in tf.matmul must be the same,
e.g. 2x3 dot 3x2x3 --> Not Allowed
fix  2x3 dot 3x6 --> then reshape to 2x2x3 
"""

import numpy as np
import sys
import time
import tensorflow as tf
#tf.enable_eager_execution()

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
        
        E = np.random.uniform( -self.vs**-0.5, self.vs**-0.5, (self.es, self.vs))
        V = np.random.uniform( -self.hd**-0.5, self.hd**-0.5, (self.hd, self.os))
        c = np.random.uniform( -self.hd**-0.5, self.hd**-0.5, (1,1,self.os))
        self.E = tf.Variable(E, dtype=tf.float32)
        self.V = tf.Variable(V, dtype=tf.float32)
        self.c = tf.Variable(c, dtype=tf.float32)
        
        self.gru = tf.keras.layers.GRU(self.hd, return_sequences=True, return_state=True)
        self.gru(tf.zeros((1,1,self.es)))
        
        self.var = [self.E, self.V, self.c]
        for m in self.gru.weights:
            self.var.append(m)
        
    def fprop(self, x):
        bs,T = x.shape
        # Grab the word vectors
        xe = tf.nn.embedding_lookup(tf.transpose(self.E), x)
        # GRU layer
        O,s = self.gru(xe)
        # Output activation
        o = tf.matmul(   tf.reshape(O, (bs*T, self.hd))   , self.V) + self.c
        o = tf.reshape(o, (bs, T, self.os))
        return o, s
    
    def calctotloss(self, x, y):
        o, s = self.fprop(x)
        yy = tf.one_hot(y, self.os)
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=o, labels=yy)
        return tf.reduce_mean(tf.reduce_sum(loss, axis=1)) # taking the mean of the batch size
    
    def saveweights(self, fn='weights.csv'):
        with open(fn, 'w') as f:
            for m in self.var:
                n = m.numpy()
                for o in n.shape:
                    if o==1:
                        n = np.squeeze(n)
                
                f.write("%d"%(n.shape[0]))
                for o in n.shape[1:]:
                    f.write(",%d"%(o))
                
                f.write("\n")
                
                if len(n.shape)==1:
                    f.write(",".join([str(p) for p in n])+"\n")
                else:
                    for o in range(n.shape[0]):
                        f.write(",".join([str(p) for p in n[o]])+"\n")
    def loadweights(self, fn='weights.csv'):
        buffer = []
        with open(fn, 'r') as f:
            for i in range(len(self.var)):
                shp = [int(m) for m in f.readline().split(",")]
                if len(shp)==1:
                    buffer.append(np.array([(m) for m in f.readline().split(",")]).astype('float32'))
                else:
                    buf = []
                    for m in range(shp[0]):
                        buf.append([float(n) for n in f.readline().split(",")])
                    buffer.append(np.array(buf).astype('float32'))
        self.E = tf.Variable(buffer[0], dtype=tf.float32)
        self.V = tf.Variable(buffer[1], dtype=tf.float32)
        self.c = tf.Variable(buffer[2], dtype=tf.float32)
        for m,n in enumerate(self.gru.weights):
            self.gru.weights[m].assign(tf.Variable(buffer[m+3], dtype=tf.float32))
        self.var = [self.E, self.V, self.c]
        for m,n in enumerate(self.gru.weights):
            self.var.append(n)
        
def randchoice(arr):
    plc = 0
    rnd = np.random.rand()
    for m in range(len(arr)):
        plc += arr[m]
        if plc>rnd:
            return m
def gentext(model, start, SL=100):
    string = []
    inp = np.array([[char2ind[m] for m in start]])
    bs,T = inp.shape
    xe = tf.nn.embedding_lookup(tf.transpose(model.E), inp)
    O,S = model.gru(xe)
    arr = tf.nn.softmax((tf.matmul(tf.squeeze(O), model.V) + tf.squeeze(model.c))[-1]).numpy()
    inp = randchoice(arr)
    string.append(vocab[inp])
    for m in range(SL):
        xe = tf.nn.embedding_lookup(tf.transpose(model.E), np.array([[inp]]))
        O,S = model.gru(xe, initial_state=S)
        arr = tf.nn.softmax((tf.matmul(tf.reshape(O, (1, O.shape[-1])), model.V) + tf.squeeze(model.c))[-1]).numpy()
        inp = randchoice(arr)
        string.append(vocab[inp])
    return start+"".join(string)

def adam(model, inps, targs, epochs, bs, a=0.001, b1=0.9, b2=0.999, lam=0):
    
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
    loss = 0
    updates = 0
    for l in range(epochs):
        
        train_loss = 0
        split = time.time()
        for m in range(batches):
            sys.stdout.write("\rBatch %d/%d %.3f s %.3f "%(m+1, batches, time.time()-split, loss))
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
        model.saveweights()

# Important architecture parameters
sl = 100 # how long a training sentence is
es = 256 # embedding size
hd = 1024 # number of nodes in RNN s-layer

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

epochs = 10
batch_size = 64
alpha = 0.01

batches = inps.shape[0] // 100 + 1
#model.loadweights()
#adam(model, inps, targs, epochs, batch_size, lam=0)
