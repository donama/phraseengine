#!/usr/bin/env python
"""
This scripts uses the phraseengine to process and generate
noun phrases from documents
"""
from phraseengine import PhraseEngine
from phraseengine.exceptions import PhraseEngineServiceException

def run_engine(document):
    response = {}
    engine = PhraseEngine()
    try:
        engine.set_document(document)
        words = engine.build_phrases()
        response ={'response':words}
        
    except PhraseEngineServiceException as e:
        response ={'error':'parser_error'
                   ,'error_description':str(e)}
    return response


if __name__ == '__main__':
    
    # For testing purpuse we run it this way, however this scripts will
    # be called by an extrnal program
    content = ''
    with open('sample3.txt','rb') as f:
        content = ''.join(f.readlines())
    
    if content:
        print run_engine(content)
        

