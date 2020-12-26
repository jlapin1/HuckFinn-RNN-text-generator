# HuckFinn-RNN-text-generator
This is a independent project I undertook in order to better understand Natural Language Processing, specifically Recurrent neural networks and their utilities. I wrote 2 different 
networks, one where I wrote my own GRU cell and forward propagation subroutines (GRUtf.py), and then another using Keras's GRU layer (GRUtf_3.py), which automatically outputs all 
outputs in the sequence. Althought the latter implementation is faster, it was a valuable exercise writing my own subroutines in order to understand the processes that are going on
under the hood of a GRU layer.

The model is training on Mark Twain's classic novel "The adventures of Huckleberry Finn". I chose this work because of Twain's unique voice and rhetorical style, a style worth
capturing and emulating using a RNN. After training for 10 epochs, examples of the networks output on various "seed" words is shown in the picture titled "Text generator
examples.png".
