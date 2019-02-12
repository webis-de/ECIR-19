
def run(sc, args):
    input_path   = args[0]
    fraction     = float(args[1])
    output_path  = args[2]
    seed = 1221

    prep_web = sc.pickleFile(input_path)
    web_articles = prep_web.map(lambda x: (x.d_id, [(x.p_id, x.d_url, x.paragraph)])).reduceByKey(lambda a,b: a+b)
    web_sample   = web_articles.sample(False, fraction, 1221)
    web_paras    = web_sample.flatMap(lambda x: list((p[0], x[0], x[1][0][1], p[2]) for p in x[1])).toDF(['p_id', 'd_id', 'd_url', 'paragraph'])
    web_paras.rdd.saveAsPickleFile(output_path)