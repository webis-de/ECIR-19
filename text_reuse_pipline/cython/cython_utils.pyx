# cython: language_level=3

import numpy as np
cimport numpy as np

ctypedef np.float64_t dtype_float
ctypedef np.int32_t dtype_int

cdef double sparse_cosine_score(np.ndarray[np.int32_t, ndim=1] x_i, np.ndarray[double, ndim=1] x_v, np.ndarray[np.int32_t, ndim=1] y_i, np.ndarray[double, ndim=1] y_v,double a_norm,double b_norm):
    cdef int n_x = x_i.shape[0]
    cdef int n_y = y_i.shape[0]
    
    cdef double a_n = a_norm
    cdef double b_n = b_norm

    cdef double sim = 0.0
    cdef int i = 0
    cdef int j = 0
    while (i<n_x and j<n_y):
        if (x_i[i] == y_i[j]):
            sim += x_v[i]*y_v[j]
            i+=1
            j+=1
        elif (x_i[i]<y_i[j]):
            i+=1
        else:
            j+=1
    return sim/(a_n*b_n)

cdef cosine_score(np.ndarray[double, ndim=1] v1, np.ndarray[double, ndim=1] v2, double norm1, double norm2):
    
    cdef double s = 0.0
    cdef int i = 0
    
    while(i < 100):
        s += v1[i] * v2[i]
        i += 1
    
    return s / (norm1 * norm2)

cdef double ngram_score(set ngrams1, set ngrams2):
    cdef double score = 0.0
    #compute how many stopword-ngrams shared between two items
    cdef set shared_ngrams
    cdef int min_ngram = 0

    shared_ngrams = ngrams1 & ngrams2
    #divide over minimum number of stopowrd-ngram
    if len(ngrams1) < len(ngrams2):
        min_ngram = len(ngrams1)
    else:
        min_ngram = len(ngrams2)
    #check if there is no stopword nagram at all.
    score  = len(shared_ngrams)/min_ngram if min_ngram > 0 else 0

    return score

cpdef double max_doc_score(list paras1, list paras2, double threshold):
    cdef double final_score=0.0
    cdef double curr_score=0.0
    for p1 in paras1:
        for p2 in paras2:
            curr_score = sparse_cosine_score(np.asarray(p1[0].indices), np.asarray(p1[0].values), np.asarray(p2[0].indices), np.asarray(p2[0].values), p1[1], p2[1])
            if curr_score > final_score:
                final_score = curr_score

    return final_score


cpdef project_vector(vec, normals):
    cdef np.ndarray[np.int32_t, ndim=1] v_i = np.asarray(vec.indices)
    cdef np.ndarray[double, ndim=1] v_v =  np.asarray(vec.values)

    cdef int n_v = v_i.shape[0]

    cdef np.ndarray[double, ndim=1] proj = np.zeros(16)
    cdef int i = 0


cpdef cosine_mapper( partition,  bc, threshold):
    
    cdef list result = []
    for item1 in partition:
        for item2 in bc:        
            if (item2[0]!=item1[0]):
                s = sparse_cosine_score(np.asarray(item2[1].indices), np.asarray(item2[1].values), np.asarray(item1[1].indices), np.asarray(item1[1].values), item2[2], item1[2])
                if s >= threshold:
                    result.append((str(item2[0])+"-"+str(item1[0]), s))
                    
    return result

cpdef cosine_mapper_symmetric( partition,  bc, threshold):
    cdef list result = []
    cdef double score = 0.0
    for doc1 in partition:
        for doc2 in bc:        
            if (doc1[0] < doc2[0]):
                score = max_doc_score(doc1[1][1], doc2[1][1], threshold)
                if score >= threshold:
                    result.append((str(doc2[1][0])+"-"+str(doc1[1][0]), score))
                    
    return result

cpdef collect_doc_ids(list hashes, dict index):
    cdef set result = set()
    for h in hashes:
        if h in index:
            result = result.union(index[h])
    return result

cpdef index_based_cosine_mapper(web, wiki_idx, wiki_tfidfs, threshold):
    import pickle
    import gzip
    
    cdef list result  = []
    cdef double score = 0.0

    for page in web:
        wiki_doc_ids = collect_doc_ids(page[2], wiki_idx.value)
        for wiki_id in wiki_doc_ids:
            if wiki_id in wiki_tfidfs.value:
                doc_tfidfs = pickle.loads(gzip.decompress(wiki_tfidfs.value[wiki_id]))
                #compute cosine similarity
                sim = max_doc_score(doc_tfidfs, page[1], threshold)
                if sim > threshold:
                    result.append((page[0], (wiki_id, sim)))

    return result

cpdef w2v_mapper( partition,  bc):
    
    cdef list result = []
    for item1 in partition:
        for item2 in bc:        
            if (item2[0]!=item1[0]):
                s  = cosine_score(np.asarray(item1[1]), np.asarray(item2[1]), item1[2], item2[2])
                result.append((str(item2[0])+"-"+str(item1[0]), s))
                    
    return result
 
cpdef w2v_ngram_mapper( partition,  bc, double w1, double w2):
    
    cdef list result = []
    cdef double tfidf_s
    cdef double ngram_s
    cdef double total_s
    for item1 in partition:
        for item2 in bc:        
            if (item2[0]!=item1[0]):
                #p2v score
                p2v_s   = cosine_score(np.asarray(item1[1]), np.asarray(item2[1]), item1[2], item2[2])
                #ngram score
                ngram_s = ngram_score(set(item1[3]), set(item2[3]))
                #total score
                total_s = (w1 * p2v_s + w2 * ngram_s)/2
                result.append((str(item2[0])+"-"+str(item1[0]), total_s))
                    
    return result

cpdef tfidf_ngram_mapper( partition,  bc, double w1, double w2):
    
    cdef list result = []
    cdef double tfidf_s
    cdef double ngram_s
    cdef double total_s

    for item1 in partition:
        for item2 in bc:        
            if (item2[0]!=item1[0]):
                #p2v score
                tfidf_s = sparse_cosine_score(np.asarray(item2[1].indices), np.asarray(item2[1].values), np.asarray(item1[1].indices), np.asarray(item1[1].values), item2[2], item1[2])
                #ngram score
                ngram_s = ngram_score(set(item1[3]), set(item2[3]))
                #total score
                total_s = (w1 * tfidf_s + w2 * ngram_s)/2
                result.append((str(item2[0])+"-"+str(item1[0]), total_s))
                    
    return result

cpdef ngram_mapper( partition,  bc):
    
    cdef list result = []
    for item1 in partition:
        for item2 in bc:        
            if (item2[0]!=item1[0]):
                #ngram score
                ngram_s = ngram_score(set(item1[1]), set(item2[1]))
                result.append((str(item2[0])+"-"+str(item1[0]), ngram_s))
                    
    return result