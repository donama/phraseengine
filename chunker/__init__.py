#!/usr/bin/env python
"""
This is a sentence chunker based on classifier , to assertian the
worth of this method we can always evaluate the performance of the classifier by passing  a trained
tagged sentence like this
>> chunker = ConsecutiveNPChunker(train_sents)
>> print chunker.evaluate(test_sents)

For training the classifier, we use the coll2000 collections
which has over 750k words in it. We split this in to parts
train and test, and with that we train all the classifiers

"""

import nltk
import re

def npchunk_features(sentence,i,history):
    word, pos = sentence[i]
    if i == 0:
        prevword,prevpos = "<START>","<START>"
    else:
        prevword,prevpos = sentence[i-1]

    if i == len(sentence) - 1:
        nextword,nextpos = "<END>","<END>"
    else:
        nextword,nextpos = sentence[i+1]

    return {
        "pos": pos,
        "word":word,
        "prevpos":prevpos,
        "nextpos":nextpos,
        "prevpos+pos":"%s+%s" %(prevpos,pos),
        "pos+nextpos":"%s+%s" %(pos,nextpos),
        "tags-since-dt": tags_since_dt(sentence,i)
    }

def tags_since_dt(sentence,i):
    tags = set()
    for word,pos in sentence[:i]:
        if pos == 'DT':
            tags = set()
        else:
            tags.add(pos)
    return '+'.join(sorted(tags))


"""
The classifier based chunker
"""
# The classifier based tagger

class ConsecutiveNPChunkTagger(nltk.TaggerI):

    def __init__(self, train_sents):
        train_set = []
        for tagged_sent in train_sents:
            untagged_sent = nltk.tag.untag(tagged_sent)
            history = []
            for i, (word, tag) in enumerate(tagged_sent):
                featureset = npchunk_features(untagged_sent, i, history)
                train_set.append( (featureset, tag) )
                history.append(tag)

        from nltk.classify import maxent
        self.classifier = maxent.MaxentClassifier.train(
            train_set, algorithm='iis', trace=0)

    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            featureset = npchunk_features(sentence, i, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence, history)

# The classifier based chunker
class ConsecutiveNPChunker(nltk.ChunkParserI):
    def __init__(self, train_sents):
        tagged_sents = [[((w,t),c) for (w,t,c) in
                         nltk.chunk.util.tree2conlltags(sent)]
                        for sent in train_sents]

        self.tagger = ConsecutiveNPChunkTagger(tagged_sents)

    def parse(self, sentence):
        tagged_sents = self.tagger.tag(sentence)
        conlltags = [(w,t,c) for ((w,t),c) in tagged_sents]
        return nltk.chunk.util.conlltags2tree(conlltags)


"""
The unigram NP chunker
"""

class UnigramChunker(nltk.ChunkParserI):

    def __init__(self,train_sents):
        train_data=[[(t,c) for w,t,c in nltk.chunk.util.tree2conlltags(sent)]
                    for sent in train_sents]
        self.tagger=nltk.UnigramTagger(train_data)

    def parse(self,sentence):
        pos_tags=[pos for (word,pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos,chunktag) in tagged_pos_tags]
        conlltags = [(word,pos,chunktag) for ((word,pos),chunktag) in zip(sentence,chunktags)]
        return nltk.chunk.util.conlltags2tree(conlltags)


# The bigram NP chunker

class BigramChunker(nltk.ChunkParserI):

    def __init__(self,train_sents):
        train_data = [[(t,c) for w,t,c in nltk.chunk.util.tree2conlltags(sent)]
                    for sent in train_sents]
        self.tagger = nltk.BigramTagger(train_data)

    def parse(self,sentence):
        pos_tags=[pos for (word,pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos,chunktag) in tagged_pos_tags]
        conlltags=[(word,pos,chunktag) for ((word,pos),chunktag) in zip(sentence,chunktags)]
        return nltk.chunk.util.conlltags2tree(conlltags)


# The trigram NP chunker

class TrigramChunker(nltk.ChunkParserI):

    def __init__(self, train_sents):
        train_data = [[(t,c) for w,t,c in nltk.chunk.util.tree2conlltags(sent)]
                    for sent in train_sents]
        self.tagger = nltk.TrigramTagger(train_data)

    def parse(self,sentence):
        pos_tags=[pos for (word,pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos,chunktag) in tagged_pos_tags]
        conlltags=[(word,pos,chunktag) for ((word,pos),chunktag) in zip(sentence,chunktags)]
        return nltk.chunk.util.conlltags2tree(conlltags)


TWO_RULE_GRAMMER = r"""
    NP: {<DT|PP\$>?<JJ>*<NN>}   # chunk determiner/posessive, adjective and nouns
        {<NNP>+}                # chunk sequences of proper nouns
        {<NN>+}
        """

class RegexChunker(object):
    def __init__(self):

        # Compile the rule
        global TWO_RULE_GRAMMER
        self.chunk_rule = nltk.RegexpParser(TWO_RULE_GRAMMER)
    def parse(self,sentences):
        parsed = []

        if sentences and len(sentences):
            for sent in sentences:
                tree = self.chunk_rule.parse(sent)
                for subtree in tree.subtrees():
                    if subtree.node == 'NP':
                        parsed.append(subtree.leaves())

        return parsed
                
