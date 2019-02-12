from text_reuse_pipline.text_alignment import aligner
from pyspark.sql.functions import udf, col, when, size
from pyspark.sql.types import *
import gzip
import json
import sys

def run(sc, args):
    
    wiki_text_path = 'text-reuse/wiki_00_text_minified'

    input_path     = args[0]
    web_text_path  = args[1]
    output_path    = args[2]
    min_similarity = float(args[3]) if len(args) > 3 else 0.01
    no_misses_threshold = int(args[4]) if len(args) > 4 else None
    max_size_of_similar_docs = int(args[5]) if len(args) > 5 else None

    p_merge_udf = udf(lambda a: ' ' if a is None else ' '.join(a), StringType())

    similar_docs_rdd = sc.textFile(input_path)
    #Convert int scores into the according float value
    similar_docs_rdd = similar_docs_rdd.map(json.loads).map(lambda x: (x[0], list(map(lambda d: (d[0], d[1]/255), x[1]))))

    #Load wikipedia text, collect it and broadcast it
    wiki_df  = sc.pickleFile(wiki_text_path).toDF()
    wiki_df  = wiki_df.select('id', p_merge_udf('paragraphs_n').alias('body'))
    wiki_rdd = wiki_df.rdd.map(lambda row: (row.id, gzip.compress(bytes(row.body, 'utf-8'))))
    wiki_rdd_local = wiki_rdd.collectAsMap() #402MB
    wiki_rdd_bc = sc.broadcast(wiki_rdd_local)

    #Load web text
    web_rdd = sc.pickleFile(web_text_path)
    web_df  = web_rdd.map(lambda row: (row.d_id, row.paragraph)).groupByKey().mapValues(list).toDF(['id', 'paragraphs'])
    web_df  = web_df.select('id', p_merge_udf('paragraphs').alias('body'))
    web_rdd = web_df.rdd.map(lambda row: (row.id, gzip.compress(bytes(row.body, 'utf-8'))))


    similar_docs_rdd = similar_docs_rdd.join(web_rdd).map(lambda x: (x[0], x[1][0], x[1][1]))    

    #convert into DocSimilarity
    if max_size_of_similar_docs == None:
        similar_docs_rdd = similar_docs_rdd.flatMap(lambda row: aligner.convert_row_to_docsimilarity(row))
        similar_docs_rdd = similar_docs_rdd.repartition(4000)
    else:
        #print('number of docsimilarity to process: ' + str(similar_docs_rdd.count()))
        similar_docs_rdd = similar_docs_rdd.flatMap(lambda row: aligner.convert_row_to_docsimilarity(row, max_size_of_similar_docs))
        similar_docs_rdd = similar_docs_rdd.repartition(4000)

    #Compute alignments
    aligned_documents = similar_docs_rdd.mapPartitions(lambda p: aligner.compute_alignments_between2ds(p, wiki_rdd_bc, min_similarity, no_misses_threshold))

    #Keep only alignments with length > 0
    aligned_documents = aligned_documents.filter(lambda doc_alignment: len(doc_alignment.alignments) > 0)
    #Convert to json
    aligned_documents = aligned_documents.map(lambda doc_alignment: json.dumps(doc_alignment, default=lambda o:o.__dict__))

    #Save alignments
    aligned_documents.saveAsTextFile(output_path)