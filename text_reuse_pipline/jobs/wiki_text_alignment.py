from text_reuse_pipline.text_alignment import aligner
from pyspark.sql.functions import udf, col, when, size
from pyspark.sql.types import *
import gzip
import json
import sys

def run(sc, args):
    
    wiki_text_path = 'text-reuse/wiki_00_text_minified'

    input_path     = args[0]
    output_path    = args[1]
    min_similarity = float(args[2]) if len(args) > 2 else 0.01
    no_misses_threshold = int(args[3]) if len(args) > 3 else None
    no_alignments_threshold = int(args[4]) if len(args) > 4 else None
    max_size_of_similar_docs = int(args[5]) if len(args) > 5 else None

    #blocks_range = blocks_range.split('-')
    #blocks_range = range(int(blocks_range[0]), int(blocks_range[1]))

    p_merge_udf = udf(lambda a: ' ' if a is None else ' '.join(a), StringType())
    wiki_df  = sc.pickleFile(wiki_text_path).toDF()
    wiki_df  = wiki_df.select('id', 'title', p_merge_udf('paragraphs_n').alias('body'))
    wiki_rdd = wiki_df.rdd.map(lambda row: (row.id, gzip.compress(bytes(row.body, 'utf-8'))))
    wiki_rdd_local = wiki_rdd.collectAsMap()
    block_text_broadcast = sc.broadcast(wiki_rdd_local)    
    print('broadcast size: ' + str(sys.getsizeof(wiki_rdd_local)))
    
    similar_docs_rdd = sc.textFile(input_path)
    #Convert int scores into the according float value
    similar_docs_rdd = similar_docs_rdd.map(json.loads).map(lambda x: (x[0], list(map(lambda d: (d[0], d[1]/255), x[1]))))


    #convert into DocSimilarity
    if max_size_of_similar_docs == None:
        similar_docs_rdd = similar_docs_rdd.flatMap(lambda row: aligner.convert_row_to_docsimilarity(row))
        similar_docs_rdd = similar_docs_rdd.repartition(4000)
    else:
        #print('number of docsimilarity to process: ' + str(similar_docs_rdd.count()))
        similar_docs_rdd = similar_docs_rdd.flatMap(lambda row: aligner.convert_row_to_docsimilarity(row, max_size_of_similar_docs))
        similar_docs_rdd = similar_docs_rdd.repartition(4000)

    
    #Compute alignments
    aligned_documents = similar_docs_rdd.mapPartitions(lambda p: aligner.compute_alignments(p, block_text_broadcast, block_text_broadcast, min_similarity, no_misses_threshold, no_alignments_threshold))

    #Keep only alignments with length > 0
    aligned_documents = aligned_documents.filter(lambda doc_alignment: len(doc_alignment.alignments) > 0)
    #Convert to json
    aligned_documents = aligned_documents.map(lambda doc_alignment: json.dumps(doc_alignment, default=lambda o:o.__dict__))

    #Save alignments
    aligned_documents.saveAsTextFile(output_path)