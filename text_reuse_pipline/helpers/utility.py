from scipy.sparse import csc_matrix
from scipy.spatial import distance
import numpy as np
import hashlib
import sys
import logging
 
def getlogger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:
        pass
    else:
        ch = logging.StreamHandler(sys.stderr)
        ch.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        return logger


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def make_part_filter(index, nb_parts):
        def part_filter(split_index, iterator):
            if split_index % nb_parts == index:
                for el in iterator:
                    yield el
        return part_filter

def cosine_sim(a, b):
   
    #return a.dot(b)/(a.norm(2)*b.norm(2))
    return distance.cosine(a.toArray(), b.toArray())

def cosine_norm(a, a_norm, b, b_norm):
    # return function for csc_matrix dot
    #return a.T.dot(b)/(a_norm*b_norm)
    return a.dot(b)/(a_norm*b_norm)

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def int_to_bytes(x):
    return x.to_bytes((x.bit_length() + 7) // 8, 'big')

def int_from_bytes(xbytes):
    return int.from_bytes(xbytes, 'big')

def sort_entries(element, minimize = False):
    dtype = [('wiki_id', int), ('cosine', float)]
    arr = np.array(element[1], dtype)
    arr.sort(order='cosine')
    #Take only the ids
    if minimize:
        return (element[0], list(map(lambda p: (p[0], int(p[1] * 255)), arr[::-1].tolist())))
    else:
        arr[:] = arr[::-1].tolist()
        return (element[0], arr)
 
def to_outputline(element):
    w_id = str(element[0])
    for e in element[1]:
        w_id+=","+str(e[0])+","+str(e[1])
    return w_id

def join_paragraphs_to_text(paragraphs):
    if paragraphs:
        doc_length = sum(map(lambda p: len(p), paragraphs))
        if doc_length > 10000:
            return 'too long'
        else:
            p = map(lambda p: p[1], paragraphs)
            text = '\n'.join(p)
            return text
    else:
        return ''

#hashing string s into 8 digits integer
def hash_string(s):
   hash_string = hashlib.sha1(s.encode()).hexdigest()
   hash_string = hash_string[0:8]
   return int(hash_string, 16)