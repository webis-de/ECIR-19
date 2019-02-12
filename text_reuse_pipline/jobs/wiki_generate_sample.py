from text_reuse_pipline.candidate_elemination.representation import tfidf
from pyspark import SparkConf, SparkContext
from pyspark.sql.context import SQLContext
from text_reuse_pipline.helpers import VSH
from text_reuse_pipline.candidate_elemination.representation import tfidf, para2vec
import numpy as np
import pandas as pd

def run(sc, args):
    original_dim = int(args[0])
    sample_frac = float(args[1])

    #TODO load tfidfs from the wikitfidf in pipeline
    wiki_path   = 'text-reuse/pipeline/wiki_preprocessed'
    output_path = 'text-reuse/pipeline/sample/articles_sample_' + str(original_dim)


    rdd = sc.pickleFile(wiki_path)
    rdd = rdd.map(lambda x: (x.d_id, [x.paragraph])).reduceByKey(lambda a, b: a+b).map(lambda x: (x[0], ' \n '.join(x[1])))
    rdd = rdd.map(lambda x: (x[0], x[1], len(x[1].split())))
    df  = rdd.toDF(['d_id', 'content', 'd_len'])
    
    #Sample fraction of the rdd and save it as csv
    wiki_sample = df.rdd.filter(lambda r: r.d_len > 2000).sample(False, sample_frac, 123)
    wiki_sample_local = wiki_sample.map(lambda x: (x.d_id, x.content) ).collect()

    wiki_df = pd.DataFrame(wiki_sample_local, columns=['d_id', 'content'])
    wiki_df.to_pickle('wiki_sample_content.dat', protocol=2)