#!/usr/bin/env python

from nltk.corpus import conll2000

from chunker import *

if __name__ == '__main__':
    # Test the unigram chunker using the coll2000 dataset
    test_sents = conll2000.chunked_sents('test.txt',chunk_types=['NP'])
    train_sents = conll2000.chunked_sents('train.txt',chunk_types=['NP'])
    
    # test the unigram chunker
    #uni_chunker = UnigramChunker(train_sents)
    #print uni_chunker.evaluate(test_sents)
    
    # test the bigram chunkers
    #big_chunker = BigramChunker(train_sents)
    #print big_chunker.evaluate(test_sents)
    
    # Testing the classifier based chunker
    class_chunker = ConsecutiveNPChunker(train_sents)
    
    print class_chunker.evaluate(test_sents)
    
    # Testing trigram chunker
    #tri_chunker = TrigramChunker(train_sents)
    #print tri_chunker.evaluate(test_sents)