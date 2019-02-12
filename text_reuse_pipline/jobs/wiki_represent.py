from text_reuse_pipline.candidate_elemination.representation import tfidf, para2vec
from pyspark import SparkConf, SparkContext
from pyspark.sql.context import SQLContext
from pyspark.sql.functions import udf, col, when, size

def tfidf_rep(sc):
    method_name = 'tfidf'
    output_path = 'text-reuse/pipeline/wiki_rep_' + method_name
    input_file  = 'text-reuse/pipeline/wiki_preprocessed'

    df = sc.pickleFile(input_file).toDF()
    #Create tfidf representation from paragraph column
    #and save the new df
    tfidf_df = tfidf(df, 'paragraph', 'tfidf')

    tfidf_rdd = tfidf_df.rdd.map(lambda r: (r.d_id, [(r.tfidf, r.norm)])).reduceByKey(lambda a, b: a+b)
    tfidf_rdd = tfidf_rdd.zipWithIndex()
    tfidf_rdd = tfidf_rdd.map(lambda r: (r[1], r[0]))

    tfidf_rdd.saveAsPickleFile(output_path)

def p2v_rep(sc):
    method_name = 'p2v'
    output_path = 'text-reuse/pipeline/wiki_rep_' + method_name
    input_file  = 'text-reuse/pipeline/wiki_preprocessed'
    w2v_model_path = 'text-reuse/experiment/w2v_model/data'


    df = sc.pickleFile(input_file).toDF()
    w2vec_lookup = sqlC.read.parquet(w2v_model_path)
    result = para2vec(sc, df, 'paragraph', w2vec_lookup)
    result.rdd.saveAsPickleFile(output_path)

def run(sc, args):
    rep_method = args[0]
    if rep_method == 'tfidf':
        tfidf_rep(sc)
    elif rep_method == 'p2vec':
        p2v_rep(sc)