from pyspark.sql.types import *
from pyspark.sql.functions import udf, col, when, size
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.mllib.linalg.distributed import RowMatrix, IndexedRowMatrix, IndexedRow, CoordinateMatrix, MatrixEntry
from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.linalg.distributed import RowMatrix, IndexedRowMatrix, IndexedRow
from scipy import spatial
from pyspark.mllib.linalg import Matrix, Matrices
from scipy.sparse import csc_matrix
from scipy.spatial import distance

import numpy as np
import time
import sys
import os
import re
import json
import argparse
import gzip
import pickle

from text_reuse_pipline.helpers import utility


def compute_ranks(sc, rdd1, rdd2, output_path, total_no_blocks, threshold, blocks_range=None):
    import cython_utils
    no_docs = rdd1.count()
    no_docs_per_block = int(no_docs/total_no_blocks)

    print(no_docs_per_block)
    rdd1 = rdd1.partitionBy(4000,lambda k: int(k) % 4000).cache()
    rdd2 = rdd2.partitionBy(total_no_blocks, lambda k: int(k) % no_docs_per_block).cache()

    #If no range specified, we loop over the whole range
    if blocks_range == None:
        blocks_range = range(total_no_blocks)
    else:
        blocks_range = blocks_range.split('-')
        blocks_range = range( int(blocks_range[0]), int(blocks_range[1]))

    #Iterate over all data parts
    for p_idx in blocks_range:
        print('Processing block number: ' + str(p_idx + 1))
        
        # get the right part
        # collect it to the driver
        # broadcast that part to all nodes
        part      = rdd2.mapPartitionsWithIndex(utility.make_part_filter(p_idx, total_no_blocks), preservesPartitioning=True)
        local_part= part.collect()
        bc_block  = sc.broadcast(local_part)
        print('Broadcasting block of size: ' + str(len(bc_block.value)))

        similarity_pairs = rdd1.mapPartitions(lambda p: cython_utils.cosine_mapper_symmetric(p, bc_block.value, threshold), preservesPartitioning=True)

        similarity_pairs = similarity_pairs.map(lambda x: (int(x[0].split('-')[0]), (int(x[0].split('-')[1]), x[1])))
        # group similarities by first article id
        similarity = similarity_pairs.groupByKey().mapValues(lambda x: list(x))
        
        ordered_similarity = similarity.map(lambda x: utility.sort_entries(x, True))
        #map into json and save as bzip2
        ordered_similarity = ordered_similarity.map(json.dumps)
        ordered_similarity.saveAsTextFile(output_path +'/'+str(p_idx),'org.apache.hadoop.io.compress.BZip2Codec')
        bc_block.unpersist()

def hash_based_ranks(sc, rdd1, rdd2, output_path, threshold):
    import cython_utils

    rdd1_hashes = rdd1.map(lambda x: (x[0], x[2]))
    rdd2_hashes = rdd2.map(lambda x: (x[0], x[2]))

    rdd1_idx = rdd1_hashes.flatMap(lambda x: ((h, x[0]) for h in x[1]))
    rdd1_idx = rdd1_idx.map(lambda x: (x[0], [x[1]])).reduceByKey(lambda a, b: a+b)

    rdd2_idx = rdd2_hashes.flatMap(lambda x: ((h, x[0]) for h in x[1]))
    rdd2_idx = rdd2_idx.map(lambda x: (x[0], [x[1]])).reduceByKey(lambda a, b: a+b)

    def suspicous_pages(item, wiki_idx_bc):
        for h in item[2]:
            if h in wiki_idx_bc.value:
                return True
        return False
        
    #broadcast the index
    shared_idx = rdd2_idx.join(rdd1_idx).map(lambda x: (x[0], x[1][0]))
    shared_idx_local = shared_idx.collectAsMap()
    print('Number of hashes: ' + str(len(shared_idx_local)))
    print('size of shared idx bc: ' + str(sys.getsizeof(shared_idx_local)))
    shared_idx_bc  = sc.broadcast(shared_idx_local)

    #broadcast wiki tfidfs
    rdd2_compressed       = rdd2.map(lambda x: (x[0], gzip.compress(pickle.dumps(x[1]))))
    rdd2_compressed_local = rdd2_compressed.collectAsMap()
    print('Number of tfidfs: ' + str(len(rdd2_compressed_local)))
    print('size of tfidf bc: ' + str(sys.getsizeof(rdd2_compressed_local)))
    rdd2_compressed_local_bc  = sc.broadcast(rdd2_compressed_local)

    #keep only items that has hashes in wiki_idx
    rdd1_filtered    = rdd1.filter(lambda x: suspicous_pages(x, shared_idx_bc))
    similarity_pairs = rdd1_filtered.mapPartitions(lambda p: cython_utils.index_based_cosine_mapper(p, shared_idx_bc, rdd2_compressed_local_bc, threshold))
    

    # group similarities by first article id
    similarity = similarity_pairs.groupByKey().mapValues(lambda x: list(x))
    ordered_similarity = similarity.map(lambda x: utility.sort_entries(x, True))
    ordered_similarity = ordered_similarity.map(json.dumps)
    ordered_similarity.saveAsTextFile(output_path, 'org.apache.hadoop.io.compress.BZip2Codec')

    shared_idx_bc.unpersist()
    rdd2_compressed_local_bc.unpersist()
