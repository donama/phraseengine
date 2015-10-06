#!/usr/bin/env python
"""
The PhraseEngine for extracting phrases from discussions
"""
from chunker import BigramChunker, TrigramChunker
from segmenter import *
from collocate import *
from exceptions import PhraseEngineException, PhraseEngineServiceException
from nltk.corpus import conll2000, conll2002
import random

class PhraseEngine(object):
    def __init__(self, ):
        self.document = None
    
    def set_document(self,document=None):
        self.document = document
        return self
    def _build_training_sents(self ):
        # This method randomly select a corpus from the provided lists and then
        # build and return a train sentences that the chunkers will use
        corpuses = [(conll2000,'train.txt'),(conll2002,'esp.train')]
        #trainer = random.choice(corpuses)
        #train_sents = trainer[0].chunked_sents(trainer[1],chunk_types=['NP'])
        train_sents = conll2000.chunked_sents('train.txt',chunk_types=['NP'])
        return train_sents
        
    
    def _build_chunkers(self):
        # Build a map of chunkers that will be used for chunking the
        # sentences
        chunkers = []
        # select a trainer that will be used for training the chunkers
        train_sents = self._build_training_sents()
        if not train_sents:
            raise PhraseEngineException('Could not build train sents for chunkers')
        chunkers.append(BigramChunker(train_sents))
        chunkers.append(TrigramChunker(train_sents))
        return chunkers
    def _generate_sentences(self):
        # From the given document generate sentences that we need to POS tag
        if not self.document or not len(self.document):
            return None
        segmenter = TextSegmentation(self.document)
        sentences = segmenter.segment()
        if not sentences:
            raise PhraseEngineException('Attempting to generate sentences from document but failed')
        return sentences
    def _generat_word_phrase(self,word_tags=[]):
        # This uses the PhraseFuture engine to generate word phrases
        # that would forma lists of dict with frequency,word mappings
        words = []
        if not word_tags:
            return words
        word_future = PhraseFuture()
        try:
            word_future.setChunks(word_tags)
            words = word_future.parse()
            
        except Exception as e:
            raise PhraseEngineException(str(e))
        return words
    
    def build_phrases(self):
        # This is the main entry point from which the document is processed to
        # generate the phrases and then returns a list of dicts
        words = []
        if not self.document:
            raise PhraseEngineServiceException('No document is set for processing')
        # We have a document for processing
        try:
            # We first convert the document into sentences
            sentences = self._generate_sentences()
            # Build the chunkers that will find the phrases int he sentences
            chunkers = self._build_chunkers()
            word_tags = []
            # Iterate through the chunkers and feed it with each of the sentences
            # to build a tree, then from the tree find out all leaf that have node type 'NP'
            # track these nodes for the contain what we are looking for
            for chunker in chunkers:
                for sent in sentences:
                    tree = chunker.parse(sent)
                    for subtree in tree.subtrees():
                        if subtree.node == 'NP':
                            word_tags.append(subtree.leaves())
            
            if not word_tags:
                print word_tags
                raise PhraseEngineException('Parser could not generate sentences from document')
            # Ok now that we have a word_tags we need to build collocations from it using
            # the collocation future. The essence of building a collocation map is to find commonly
            # used phrases and even new phrases that can stand or express an idea
            
            collocation_future = CollocationFuture()
            future_words = []
            for wordtag in word_tags:
                # convert to tuples
                future_words.append(tuple(wordtag))
                
            collocation_future.setChunks(future_words)
            collocation_future.willParse()
            collocations = collocation_future.parse()
            
            if not collocations:
                raise PhraseEngineException('Building collocations from tagged words failed')
            # Ok finally we prepare the collocations for phrase extractions using the
            # pos-tag as a guide to determining what a noun phrase should be. Occasionally the engine
            # generates a very long phrase which means that the finall generated noun phrase still
            # requires human to post-process the outcome
            phrases = []
            for collo in collocations:
                for item in collo:
                    phrases.append(list(item))
            
            phrase_future = PhraseFuture()
            phrase_future.setChunks(phrases)
            words = phrase_future.parse()
            
        except (Exception, PhraseEngineException) as e:
            raise PhraseEngineServiceException(e)
        
        return words
    
        
    
        
    
    
    
    

