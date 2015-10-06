#!/usr/bin/env python
"""
Utility for finding collocations from tagged chunked senetences
"""

import nltk

from nltk.collocations import *
from nltk.corpus import stopwords

MODE_SEARCH = 0
MODE_NOUN = 1

import re

CAPS_REGEXP = re.compile(r'[A-Z]')

class PhraseFuture(object):
    
    def __init__(self):
        
        self.multiterms = []
        self.terms = {}
        self.stopset = set(stopwords.words('english'))
        self.chunks = [];
    
    def addTerms(self,term):
        # We add the term to the multiterm map as well
        # as the term dictionary this is to ensure that for every
        # possible multiple terms, we also keep track of their component
        # words
        self.multiterms.append(term)
        self.terms[term] = self.terms.get(term,0)+1
        
    def setChunks(self,chunks):
        self.chunks = chunks
    
    def filterStopwords(self,wordmaps):
        terms = []
        CHAR_SPECS=re.compile(r'[\(\:\)\_\!\[\]\+\/\*\;\,\?\@0-9]+')
        
        for term,freq in wordmaps.items():
            # If the length of the word is more than 36 discard it
            if len(term) >= 42:
                continue
            if term and not CHAR_SPECS.match(term):        
                word_group = []
                for word in term.split():
                    word = word.lower()
                    if not word in self.stopset:
                        if word.islower():
                            word = word.title()
                        word_group.append(word)
                        
                if word_group and len(word_group):
                    terms.append(dict(word=' '.join(word_group),frequency=freq))
        
        return terms
                
    def parse(self):
        
        global MODE_SEARCH
        global MODE_NOUN
        global CAPS_REGEXP
        
        state = MODE_SEARCH
        noun = MODE_NOUN
        
        if not self.chunks:
            return list()
        # Ok we parse each item of the chunks which is a list of tuples
        # consisting of entries of this type (word,pos-tag)
        for item in self.chunks:
            for tags in item:
                # Get hold of the word
                term = str(tags[0])
                pos = tags[1]
                term = term.strip()
                if term == '':
                    continue
                if state == MODE_SEARCH and pos.startswith('N'):
                    # Noun component
                    state == noun
                    if len(term) > 2:
                        self.addTerms(term)
                        
                #what state is our state machine one
                elif state == MODE_SEARCH and pos == 'JJ' and CAPS_REGEXP.match(term[:1]) is not None:
                    # we transition to a noun state
                    state = noun
                    if len(term) > 2:
                        self.addTerms(term)
                #if we are already in noun state
                elif state == noun and pos.startswith('N'):
                    if len(term) > 2:
                        self.addTerms(term)
                elif state == noun and not pos.startswith('N'):
                    
                    state = MODE_SEARCH
            
            # package the word entries and store them
            if len(self.multiterms) > 1:
                word = ' '.join(self.multiterms)
                self.terms[word] = self.terms.get(word,0)+1
            elif len(self.multiterms) == 1:
                word = self.multiterms.pop()
                self.terms[word] = self.terms.get(word,0)+1
            self.multiterms = []
        
        # at this stage we return the terms with their occurencies
        if self.terms and len(self.terms):
            return self.filterStopwords(self.terms)
        else:
            return list()
            
    

class CollocationFuture(object):
    def __init__(self ):
        
        self.collocations = []
        self.chunks = []
        self.bigram_measures = None
        self.trigram_measure = None
        self.bfinder = None
        self.tfinder = None
        self.with_filter = None
        self.filter_limit = 0;
    
    def setChunks(self, chunks):
        self.chunks = chunks
    
    def withFilter(self):
        self.with_filter = True
    
    def filterLimit(self,num):
        self.filterLimit = num
    
    def willParse(self):
        self.bigram_measures = nltk.collocations.BigramAssocMeasures()
        self.trigram_measure = nltk.collocations.TrigramAssocMeasures()
        
    def parse(self):
        if not self.chunks:
            return None
        # build the finder
        dataset = []
        self.bfinder = BigramCollocationFinder.from_words(self.chunks)
        self.tfinder = TrigramCollocationFinder.from_words(self.chunks)
        
        # if we are applying filter set it here
        if self.with_filter and self.filterLimit:
            self.bfinder.apply_freq_filter(self.filterLimit)
            self.tfinder.apply_freq_filter(self.filterLimit)
            
        parsed = self.bfinder.nbest(self.bigram_measures.pmi,50)
        dataset.extend(parsed)
        # The trigram collocation
        tparsed = self.tfinder.nbest(self.trigram_measure.pmi,50)
        dataset.extend(tparsed)
        return dataset
