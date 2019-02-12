from text_reuse_pipline.candidate_elemination.representation import tfidf, para2vec, joint_tfidf
from pyspark import SparkConf, SparkContext
from pyspark.sql.context import SQLContext
from pyspark.sql.functions import udf, col, when, size

def run(sc, args):
    ds1_path = args[0]
    ds2_path = args[1]
    output_path  = args[2]
    

    ds1  = sc.pickleFile(ds1_path).toDF()
    ds2  = sc.pickleFile(ds2_path).toDF()
    
    #Keep only the following column: p_id, d_id, paragraph
    ds1  = ds1.select('p_id', 'd_id', 'paragraph')
    ds2  = ds2.select('p_id', 'd_id', 'paragraph')

    #fit a tfidf model on ds1 and use to transofmr ds2
    ds1_df, ds2_df = joint_tfidf(ds1, ds2, 'paragraph', 'tfidf')

    ds2 = ds2_df.rdd.map(lambda row: (row.d_id, row.tfidf, row.norm))
    ds1 = ds1_df.rdd.map(lambda row: (row.d_id, row.tfidf, row.norm))
    ds2.saveAsPickleFile(output_path + '/' + ds2_path.split('/')[-1] + '_tfidf')
    ds1.saveAsPickleFile(output_path + '/' + ds1_path.split('/')[-1] + '_tfidf')
