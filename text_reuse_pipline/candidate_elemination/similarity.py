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
import argparse

from text_reuse_pipline.helpers import utility

#'/user/waje8705/wiki-text/wiki_00'
conf = (SparkConf().setAppName("all_sim"))
sc = SparkContext(conf = conf)
sqlC = SQLContext(sc)

sc.addPyFile('hdfs://betaweb020:8020/user/waje8705/cython/spark_c.so')
import spark_c


def load_data(data_path):
    wiki = sc.pickleFile(data_path)
    df = wiki.toDF()
    # filter empty fields, they cause a the script to stop
    df = df.filter(col("paragraph").isNotNull())
    df = df.filter(col("title").isNotNull())

    return df

# Divide input_clm into paragraphs and add it to a new
#TODO use wikipedia represention of pargraphs with normalization and then flattening
def split_text_to_paragraphs(df, input_clm, output_clm):
    #Split text into paragraphs
    udf = udf(lambda x: re.split('\s{4,}',x), ArrayType(StringType()))
    df  = df.withColumn(output_clm, udf(getattr(df, input_clm)))
    return df


# Flatten the input column and Tokenize it's text of input_clm
def flatten(df,input_clm, output_clm):
    para_rdd = df.rdd.flatMap(lambda xs: [(xs.title, xs.id,  x) for x in getattr(xs, input_clm)])
    result = para_rdd.toDF(["title", "id", output_clm])
    return result

# Generating tfidf model features from input_clm and
# and write it to output_clm
def paragraph_representation(df, input_clm, output_clm, nb_features):
    #tokenize
    tokenizer = Tokenizer(inputCol=input_clm, outputCol="tokens")
    tokenized_df = tokenizer.transform(df)

    # use mllib to create the tf-idf vectors
    para_hashingTF = HashingTF(inputCol= "tokens", outputCol="rawfeatures", numFeatures=nb_features)
    para_featurized_df = para_hashingTF.transform(tokenized_df)

    para_idf = IDF(inputCol="rawfeatures", outputCol=output_clm)
    para_idf_model = para_idf.fit(para_featurized_df)

    tfidf_df = para_idf_model.transform(para_featurized_df)
    
    row_rdd  = tfidf_df.select("features", "id").rdd.map(lambda x: (x.features, x.id)).cache()
    row_rdd  = row_rdd.zipWithIndex().map(lambda x: (x[0][0], x[0][1], x[1]))
    #TODO save previous df column names and use them...
    return row_rdd.toDF([output_clm, "id", "index"])

# Save tfidf model as (index, (features, id, features_norm))
# Save tfidf into all_vectors  and then split it into num_parts and save each part
def save_representation(rdd, output_path, num_parts):
    #partition_count = 10
    #calc norm for every vector, remap value (index, (sparse vector, wiki_id, norm))
    #TODO instead of repartition we could either partitionby index or never partition because
    #its being done when loading the tfidf in calc_similarity method
    #cluster_rdd = rdd.map(lambda x : (x[2], (x[0], x[1], x[0].norm(2)))).repartition(partition_count).cache()
    cluster_rdd = rdd.map(lambda x : (x[2], (x[0], x[1], x[0].norm(2)))).cache()
   
    # save whole rdd 
    cluster_rdd.saveAsPickleFile(output_path + "/all_vectors")

    size = cluster_rdd.count()

    # adapt the num_parts value with a prime number of your choice
    splitsize = divmod(size, num_parts)
    split_rdd = cluster_rdd.partitionBy(num_parts,lambda k: int(k) % num_parts).cache()

    # save all parts for further calculation
    for it in range(num_parts):
        part = split_rdd.mapPartitionsWithIndex(utility.make_part_filter(it, num_parts)).cache()
        part.saveAsPickleFile(output_path + "/part_vectors"+str(it))


# Load all_tfidf vectors, and the part_idx of tfidf. Then split part_idx into 
# blocks_no parts. broadcast each part in turn to all machines and then calculate 
# similarity between all_tfidf and the specified block.
def calc_similarity(part_idx, blocks_range=None, save_as='csv'):

    input_path    = conf['docs_representation_path']
    output_path   = conf['candidate_docs_path']
    sim_threshold = conf['similarity_threshold']
    blocks_no     = conf['broadcasted_blocks_no']

    # each node gets 1/2000 of the data, partition by index
    cluster_rdd = sc.pickleFile(input_path +"/all_vectors", 2000).partitionBy(2000,lambda k: int(k) % 2000).cache()
    part_rdd    = sc.pickleFile(input_path + "/part_vectors" +  str(part_idx)).partitionBy(blocks_no, lambda k: int(k) % blocks_no).cache()

    #If no range specified, we loop over the whole range
    if blocks_range == None:
        blocks_range = range(blocks_no)
    else:
        blocks_range = blocks_range.split('-')
        blocks_range = range( int(blocks_range[0]), int(blocks_range[1]))

    # iterate over all data parts
    for single_partition in blocks_range:
        # get the right part
        part = part_rdd.mapPartitionsWithIndex(utility.make_part_filter(single_partition, blocks_no), preservesPartitioning=True)
        # collect it to the driver
        local_part = part.collect()
        # broadcast that part to all nodes    
        broad_rdd = sc.broadcast(local_part)
        
        similarity_pairs = cluster_rdd.mapPartitions(lambda p: spark_c.cosine_mapper(p, broad_rdd.value, sim_threshold), 
            preservesPartitioning=True)
        
        #Keep just the highest similarity of all article-article similarities 
        #TODO is this reduceByKey really helpful?? 
        similarity_pairs = similarity_pairs.reduceByKey(max) 
        similarity_pairs = similarity_pairs.map(lambda x: (int(x[0].split('-')[0]), (int(x[0].split('-')[1]), x[1])))
        # group similarities by first article id
        similarity = similarity_pairs.groupByKey().mapValues(lambda x: list(x))
        ordered_similarity = similarity.map(lambda x: utility.sort_entries(x))

        if save_as == 'csv':
            output = ordered_similarity.map(utility.to_outputline).coalesce(10)
            output.saveAsTextFile(output_path +'/similarity_rdd/'+str(single_partition)+'.csv')
        elif save_as == 'pickle':
            ordered_similarity = ordered_similarity.coalesce(10)
            ordered_similarity.saveAsPickleFile(output_path +'/similarity_rdd_pickled/'+str(single_partition))

        broad_rdd.unpersist()


def building_tfidf(conf, inpt_clm='paragraph'):
    print("Loading Data....")
    #1.Load data (id, title, array of paragraphs)
    data = load_data(conf['data_path'])

    print("Flatten Paragraphs column ....")
    #2.Flatten the paragraphs ( id, title, paragraph)
    data  = flatten(data, inpt_clm, inpt_clm)

    print("Building tfidf model....")
    #3.Build tfidf model over paragraphs ("features", "id", "index")
    #TODO we should keep the same schema of the data df and only add features column
    tfidf = paragraph_representation(data, inpt_clm, 'features', conf['nb_features'])

    print("Saving tfidf model...")
    #4.Save tfidf model
    save_representation(tfidf.rdd, conf['docs_representation_path'], conf['tf_idf_num_parts'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='align similar documents')
    parser.add_argument('task', help='Task to perform tfidf/cosine_similarity')
    parser.add_argument('--config', help='Path to the config file')

    #tfidf task arguments
    parser.add_argument('--clm_name', help='Column name to run the tfidf over, default= "paragraph" ')

    #similarity task arguments
    parser.add_argument('--part_no', help='Part number on what to perform similarity')
    parser.add_argument('--blocks_range', help='Specified blocks range, eg: 2-5')
    parser.add_argument('--sim_output', help='Similarity output as csv or pickle')

    args = parser.parse_args()

    print("Loading config....")
    #loading config
    conf_str = open(args.config).read()
    conf     = json.loads(conf_str)

    if args.task == 'tfidf':
        #======= Generating tfidf vectors 
        print("generating tfidf")
        building_tfidf(conf, args.clm_name)
    elif args.task == 'cosine_similarity':
        #======= Calculating Similarity
        print("calculating similarity....")
        calc_similarity(args.part_no, args.blocks_range, args.sim_output)