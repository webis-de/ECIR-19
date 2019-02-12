from text_reuse_pipline.candidate_elemination import ranking


def run(sc, args):
    #'hdfs://betaweb020:8020/user/sile2804/cython_utils.so'
    cython_path  = args[0]
    blocks_range = args[1]
    threshold    = float(args[2])

    rdd1_path = 'text-reuse/pipeline/wiki_rep_tfidf'
    rdd2_path = 'text-reuse/pipeline/wiki_rep_tfidf'
    output_path = 'text-reuse/pipeline/candidates'
    total_no_blocks = 100
    

    #import cython utils
    sc.addPyFile(cython_path)

    rdd1 = sc.pickleFile(rdd1_path)
    rdd2 = sc.pickleFile(rdd2_path)

    ranking.compute_ranks(sc, rdd1, rdd2, output_path, total_no_blocks, threshold, blocks_range)