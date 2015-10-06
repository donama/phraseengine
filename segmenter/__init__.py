#!/usr/bin/env python
"""
The segmentation of raw text involves performing a lot of work
this includes sentence segmenting of raw text to produce lists of
sentences, tokenization of the sentences to and finally pos tagging the tokenized
sentences.

This step prepares the raw input texts for chunking to extract phrases
"""
import nltk

class TextSegmentation(object):
    
    def __init__(self, raw_text):
        self.document = raw_text
    
    def segment(self ):
        
        # First we need to clean the document of html tags etc
        if not self.document:
            return None
        self.document = nltk.clean_html(self.document)
        # First step build sentence segments from the document
        sentences = nltk.sent_tokenize(self.document)
        # build word tokenize
        sentences = [nltk.word_tokenize(sent) for sent in sentences]
        # finally generate pos-tags
        sentences = [nltk.pos_tag(sent) for sent in sentences]
        #we feed this sentences to a chunker for processing
        return sentences
        
    
    

