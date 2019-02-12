from pyspark import SparkConf, SparkContext
from pyspark.sql.context import SQLContext
from text_reuse_pipline.helpers import LSH

def run(sc, args):
    code_length = args[0]
    no_codes    = args[1]
    #TODO load tfidfs from the wikitfidf in pipeline
    wiki_path   = 'text-reuse/reps/wiki_tfidf'
    output_path = 'text-reuse/pipeline/hashes/'

    wiki  = sc.pickleFile(wiki_path).repartition(4000).toDF()
    wiki  = LSH.hash_df(wiki, 'tfidf', 'hashes', 260000, int(code_length), int(no_codes))

    #reshape the rdd into form of (id, ([hashes], [(tfidf, norm)]))
    output = wiki.rdd.map(lambda x: (x.doc_id, [(x.doc_id, x.tfidf, x.norm, x.hashes)])).reduceByKey(lambda x, b: a+b)

    output.saveAsPickleFile(output_path + 'lsh_' + code_length + '_' + no_codes)