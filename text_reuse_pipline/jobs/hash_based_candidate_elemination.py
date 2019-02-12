from text_reuse_pipline.candidate_elemination import ranking
from text_reuse_pipline.helpers import hashing
from text_reuse_pipline.helpers import utility

def run(sc, args):
    #'hdfs://betaweb020:8020/user/sile2804/cython_utils.so'
    rdd1_tfidf_path   = args[0]
    rdd1_hashes_path  = args[1]

    rdd2_tfidf_path   = args[2]
    rdd2_hashes_path  = args[3]

    output_path  = args[4]
    cython_path  = args[5]
    blocks_range = args[6]
    total_no_blocks = int(args[7])
    threshold       = float(args[8])

    hamming_dist    = 0
    #If no range specified, we loop over the whole range
    if blocks_range == None:
        blocks_range = range(total_no_blocks)
    else:
        blocks_range = blocks_range.split('-')
        blocks_range = range( int(blocks_range[0]), int(blocks_range[1]))

    #import cython utils
    sc.addPyFile(cython_path)

    rdd1_tfidf  = sc.pickleFile(rdd1_tfidf_path).map(lambda x:  (x[0], [(x[1], x[2])])).reduceByKey(lambda a, b: a+b)
    rdd2_tfidf  = sc.pickleFile(rdd2_tfidf_path).map(lambda x:  (x[0], [(x[1], x[2])])).reduceByKey(lambda a, b: a+b)

    #Load hashes, convert them to int, extend if required, and collect the hashes per doc_id
    rdd1_hashes = sc.pickleFile(rdd1_hashes_path).map(lambda x: (x[0], hashing.binlist_to_bstring(x[1])))
    rdd1_hashes = rdd1_hashes.flatMap(lambda x: list((x[0], [d]) for d in hashing.expand_hash(x[1], hamming_dist)))
    rdd1_hashes = rdd1_hashes.reduceByKey(lambda a, b: a+b)

    rdd2_hashes = sc.pickleFile(rdd2_hashes_path).map(lambda x: (x[0], hashing.binlist_to_bstring(x[1])))
    rdd2_hashes = rdd2_hashes.flatMap(lambda x: list((x[0], [d]) for d in hashing.expand_hash(x[1], hamming_dist)))
    rdd2_hashes = rdd2_hashes.reduceByKey(lambda a, b: a+b)

    #create dataset containing hashes and tfidfs
    rdd1 = rdd1_tfidf.join(rdd1_hashes).map(lambda x: (x[0], x[1][0], x[1][1]))
    rdd1 = rdd1.repartition(4000)

    rdd2 = rdd2_tfidf.join(rdd2_hashes).map(lambda x: (x[0], x[1][0], x[1][1]))
    rdd2 = rdd2.repartition(total_no_blocks).cache()

    #Iterate over all data parts
    for p_idx in blocks_range:
        print('Processing block number: ' + str(p_idx + 1))

        rdd2_part = rdd2.mapPartitionsWithIndex(utility.make_part_filter(p_idx, total_no_blocks), preservesPartitioning=True)
        
        partition_path = output_path + '/' + str(p_idx)

        ranking.hash_based_ranks(sc, rdd1, rdd2_part, partition_path, threshold)