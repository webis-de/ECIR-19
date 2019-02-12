from text_reuse_pipline.text_alignment import aligner
from pyspark.sql.functions import udf, col, when, size
from pyspark.sql.types import *
from text_reuse_pipline.helpers import utility

import gzip
import json
import sys

def run(sc, args):
    
    wiki_text_path = 'text-reuse/wiki_00_text_minified'
    blocks_range = args[0]
    top_k = int(args[1])

    p_merge_udf = udf(lambda a: ' ' if a is None else ' '.join(a), StringType())
    wiki_df  = sc.pickleFile(wiki_text_path).toDF()
    wiki_df  = wiki_df.select('id', 'title', p_merge_udf('paragraphs_n').alias('body'))
    wiki_rdd = wiki_df.rdd.map(lambda x: (x.id, len(x.body.split())))

    top_k_docs = wiki_rdd.takeOrdered(top_k, key=lambda x: -x[1])
    top_k_ids  = list(map(lambda x: x[0], top_k_docs))

    candidates_path = 'text-reuse/pipeline/candidates/[' + str(blocks_range) + ']'
    output_path     = 'text-reuse/pipeline/shuffled_candidates/' + str(blocks_range) + '_' + str(top_k)
    
    similar_docs_rdd = sc.textFile(candidates_path)
    sim_pairs = similar_docs_rdd.map(json.loads).flatMap(lambda x: ((x[0], s[0], s[1]/255) for s in x[1]))
    sim_pairs = sim_pairs.repartition(1000)


    def extract_top_k(p, top_k_ids):
        result = []
        for d in p:
            if d[0] in top_k_ids:
                result.append((d[0], [(d[1], d[2])]))
            elif d[1] in top_k_ids:
                result.append((d[1], [(d[0], d[2])]))

        return result

    final_result = sim_pairs.mapPartitions(lambda p: extract_top_k(p, top_k_ids)).reduceByKey(lambda a,b: a+b)

    ordered_similarity = final_result.map(lambda x: utility.sort_entries(x, True))
    #map into json and save as bzip2
    ordered_similarity = ordered_similarity.map(json.dumps)
    ordered_similarity.saveAsTextFile(output_path,'org.apache.hadoop.io.compress.BZip2Codec')