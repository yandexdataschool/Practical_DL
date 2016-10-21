# -*- coding: utf-8 -*-


from operator import add
import re
import numpy as np
def get_metrics(gen_sequences,alphabet):    
    strings = map(lambda v:reduce(add,v), map(alphabet.__getitem__,gen_sequences))
    
    #at least one complete string
    strings = filter(lambda v: '|' in v,strings)
    
    #cut off last unfinished string, if any
    
    strings = map(lambda v: v[:-1-v[::-1].index('|')],strings)
    
    all_strings = '|'.join(strings)
    
    
    
    seqs = filter(len,all_strings.split('|'))
    
    matches = map(lambda seq:re.match(r"^a+b+c+",seq),seqs)
    
    is_correct = map( lambda seq,m: m is not None and m.pos==0 and m.endpos == len(seq), 
                        seqs, matches)
    
    
    
    
    correct_seqs = np.array(seqs)[np.array(is_correct,dtype='bool')]
    
    seqs_error = map(lambda s: s.count('a')+ s.count('b') - s.count('c'),correct_seqs)
    
    return np.mean(is_correct), np.mean(np.abs(seqs_error))

