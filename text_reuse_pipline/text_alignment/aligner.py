# this script reads the wiki xml files, creates tf-idf vectors and saves them in equal parts
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
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
import gzip
import argparse
import logging
import gc


logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

def compress_paragraphs(row) :
    from text_reuse_pipline.helpers import utility
    #join paragraphs into one string
    text = utility.join_paragraphs_to_text(row[1])
    #return a tuple of (wiki_id, compressed text)
    return (row[0], gzip.compress(bytes(text, 'utf-8')))


def convert_row_to_docsimilarity(row, max_size=None):
    from text_reuse_pipline.text_alignment.models import DocSimilarity
    from text_reuse_pipline.helpers import utility

    wiki_id = row[0]
    other_wiki_ids = row[1]
    wiki_body = row[2] if len(row) > 2 else None

    if max_size != None:
        for c in utility.chunks(other_wiki_ids, max_size):
            yield DocSimilarity(wiki_id, list(c), wiki_body)
    else:
        yield DocSimilarity(wiki_id, other_wiki_ids, wiki_body)

def compute_alignments(doc_similarities, wiki_broadcast, web_broadcast, min_similarity=0.01, no_misses_threshold=None, no_alignments_threshold=None):
    from text_reuse_pipline.helpers import utility
    from text_reuse_pipline.text_alignment.models import DocSimilarity, DocAlignment, PicaPicaAligner
    logger = utility.getlogger('PYSPARK-WORKER')

    print('Starting alignments..')
    partition_doc_alignments = []

    #Initialize an instance of PicaPicaAligner that open java subprocess of (picapicalianger.jar)
    #and communicate with it to align documents
    aligner = PicaPicaAligner()

    #retrieve the collected wikipedia from the broadcasted variable
    #wikipedia = wiki_broadcast.value

    #Loop over all doc similarities objects and compute alignments
    for doc_sim in doc_similarities:
        current_wiki_id = doc_sim.wiki_id
        if current_wiki_id in wiki_broadcast.value:
            wiki_article_text = gzip.decompress(wiki_broadcast.value[current_wiki_id]).decode('utf-8')
            doc_alignment     = DocAlignment(current_wiki_id, len(wiki_article_text))
            print('Aligner: processing id: ' + str(current_wiki_id) + ' size: ' + str(len(wiki_article_text)) + ' against ' + str(len(doc_sim.similarity_map)))
            no_misses = 0
            no_alignments = 0
            for i, other_wiki_id in enumerate(doc_sim.similarity_map):
                score = doc_sim.similarity_map[other_wiki_id]
                if score < min_similarity:
                    print('Went under similarity threshold: ' + str(score))
                    break
                if other_wiki_id in web_broadcast.value:
                    other_wiki_article_text = gzip.decompress(web_broadcast.value[other_wiki_id]).decode('utf-8')
                    alignments = aligner.align(wiki_article_text, other_wiki_article_text)
                
                    if len(alignments) > 0:
                        doc_alignment.add_alignment(other_wiki_id,len(other_wiki_article_text), alignments)
                        no_misses = 0
                        no_alignments +=1
                    else:
                        #no alignments found
                        no_misses +=1
                    del other_wiki_article_text

                #Check if we examined no_misses_threshold number of documents and we didn't find any alignments
                if (no_misses_threshold is not None) and (no_misses > no_misses_threshold):
                    break

                #Check if the document has been found to be aligned with more than x other docs
                if (no_alignments_threshold is not None) and (no_alignments > no_alignments_threshold):
                    break

            del wiki_article_text
            gc.collect()
            partition_doc_alignments.append(doc_alignment)

    aligner.finish()
    return partition_doc_alignments


def compute_alignments_between2ds(doc_similarities, ds2_bc, min_similarity=0.01, no_misses_threshold=None):
    from text_reuse_pipline.helpers import utility
    from text_reuse_pipline.text_alignment.models import DocSimilarity, DocAlignment, PicaPicaAligner
    logger = utility.getlogger('PYSPARK-WORKER')

    print('Starting alignments..')
    partition_doc_alignments = []

    #Initialize an instance of PicaPicaAligner that open java subprocess of (picapicalianger.jar)
    #and communicate with it to align documents
    aligner = PicaPicaAligner()

    #Loop over all doc similarities objects and compute alignments
    for doc_sim in doc_similarities:
        d1_id   = doc_sim.wiki_id
        d1_text = gzip.decompress(doc_sim.wiki_body).decode('utf-8')

        doc_alignment     = DocAlignment(d1_id, len(d1_text))
        print('Aligner: processing id: ' + str(d1_id) + ' size: ' + str(len(d1_text)) + ' against ' + str(len(doc_sim.similarity_map)))
        no_misses = 0
        no_alignments = 0
        for i, ds2_id in enumerate(doc_sim.similarity_map):
            score = doc_sim.similarity_map[ds2_id]
            if score < min_similarity:
                print('Went under similarity threshold: ' + str(score))
                break
            if ds2_id in ds2_bc.value:
                ds2_text = gzip.decompress(ds2_bc.value[ds2_id]).decode('utf-8')
                alignments = aligner.align(d1_text, ds2_text)
            
                if len(alignments) > 0:
                    doc_alignment.add_alignment(ds2_id,len(ds2_text), alignments)
                    no_misses = 0
                else:
                    #no alignments found
                    no_misses +=1
                del ds2_text
                no_alignments +=1
            else:
                print('id not found')
            #Check if we examined no_misses_threshold number of documents and we didn't find any alignments
            if (no_misses_threshold is not None) and (no_misses > no_misses_threshold):
                print('break due no_misses setting')
                break

        print('aligned against: ' + str(no_alignments))
        print('found :' + str(len(doc_alignment.alignments)))
        del d1_text
        gc.collect()
        partition_doc_alignments.append(doc_alignment)

    aligner.finish()
    return partition_doc_alignments