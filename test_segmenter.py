#!/usr/bin/env python

from segmenter import *

from collocate import *
from chunker import BigramChunker

from nltk.corpus import conll2000

import urllib2

if __name__ == '__main__':
    
    # read the current punch news headline and use it for testing
    uri = ''
    content = ''
    
    
    with open('sample.txt','r') as f:
        content = f.readlines()
       
    
    content = ''.join(content)
    
    seg = TextSegmentation(content)
    sentences = seg.segment()
    #print sentences
    train_sents = conll2000.chunked_sents('train.txt',chunk_types=['NP'])
    
    test_chunker = BigramChunker(train_sents)
    
    # Testing the bigram chunker
    tests = []
    for sent in sentences:
        tree = test_chunker.parse(sent)
        for subtree in tree.subtrees():
            if subtree.node =='NP':
                tests.append(subtree.leaves())
    
    # Testing the collocationFuture
    #print 'The list of sentences with pos-tag'
    #print tests
    #print "\n"
    
    future1 = CollocationFuture()
    ctk = []
    for km in tests:
        # convert this to a tuple
        ctk.append(tuple(km))
    
    future1.setChunks(ctk)
    future1.willParse()
    resp = future1.parse()
    
    # Convert the tuple entries to a list so we can parse it using the
    # phrase engine future
    phrases = []
    for item in resp:
        #phrases.append(list(k))
        for k in item:
            phrases.append(list(k))
        
    #print phrases    
    future = PhraseFuture()
    future.setChunks(phrases)
    resp = future.parse()
    
    print resp
