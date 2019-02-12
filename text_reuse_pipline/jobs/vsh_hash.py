from text_reuse_pipline.helpers import VSH

from pyspark import SparkConf, SparkContext
from pyspark.sql.context import SQLContext
from pyspark.sql.functions import udf, col, when, size
from pyspark.sql.types import *
from pyspark.ml.feature import Tokenizer, CountVectorizer, IDF, StopWordsRemover, CountVectorizerModel
import numpy as np

def hash_partition(p, model):
    import tensorflow as tf
    import numpy as np
    import os, zipfile
    
    l = list(p)
    
    d_ids  = list(map(lambda x: x.d_id, l))
    
    #computing codes from tfidf vectors
    print('trying to transform feature vector..')
    codes  = model.transform(list(map(lambda x: x.tfidf.toArray(), l)))
    result = zip(d_ids, codes)
    
    model.cleanup()

    return result

def tfidf(df, cv_model, input_clm, output_clm):
    tokenizer = Tokenizer(inputCol=input_clm, outputCol="tokens")
    tokenized_df = tokenizer.transform(df)

    #remove stopwords
    remover      = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
    tokenized_df = remover.transform(tokenized_df)

    
    def tokens_filter(tokens):
        from nltk.stem import SnowballStemmer
        snowball_stemmer = SnowballStemmer("english")
        
        filtered_tokens = list(filter(lambda x: len(x) > 2 and x.isalpha(), tokens))
        stemmed_tokens = list(map(lambda x: snowball_stemmer.stem(x), filtered_tokens))
        return stemmed_tokens
    
    token_filter_udf = udf(lambda x: tokens_filter(x), ArrayType(StringType()))
    tokenized_df     = tokenized_df.withColumn('filtered_tokens', token_filter_udf('filtered_tokens'))
    
    para_featurized_df = cv_model.transform(tokenized_df)

    para_idf = IDF(inputCol='rawfeatures', outputCol=output_clm)
    para_idf_model = para_idf.fit(para_featurized_df)

    tfidf_df = para_idf_model.transform(para_featurized_df)

    return tfidf_df

def run(sc, args):
    ds_path    = args[0]
    output_path  = args[1]
    
    model_path   = args[2]
    model_name   = args[3]

    original_dim = int(args[4])
    hidden_dim   = int(args[5])
    latent_dim   = int(args[6])

    cv_model_path  = 'text-reuse/pipeline/vsh-hashing/cv_model'
    cv_model  = CountVectorizerModel.load(cv_model_path)

    threshold = None
    if latent_dim == 8:
        threshold = np.array([ 0.1457837 ,  0.061413  , -0.03391605,  0.04686656, -0.14745404,
       -0.08641829, -0.04190724, -0.05972087])

    elif latent_dim == 16:
        threshold = np.array([ 0.00231892, -0.00791987,  0.00027306,  0.07018767, -0.07945273,
        0.01763633,  0.01450929,  0.04488222, -0.0289745 ,  0.02851318,
        0.01496754,  0.00133035, -0.00523619, -0.10513094,  0.07906742,
       -0.07930097])

    else:
       threshold = np.array([-0.01227623, -0.00382998, -0.00029179, -0.04484864, -0.02657753,
        0.01505825,  0.00319679, -0.01186464, -0.03057225,  0.02324941,
        0.01272652, -0.01289577, -0.02995954,  0.04656317, -0.01781761,
       -0.01934269,  0.1332021 ,  0.00064231,  0.01289176, -0.00131864,
        0.02279386, -0.06245026, -0.02096441,  0.01817522,  0.02722896,
        0.0211685 ,  0.01392594, -0.06448705,  0.00062385,  0.02365676,
       -0.01207885,  0.02566718])


    vdsh_loader = VSH.VDSHLoader(model_path, model_name,  threshold , original_dim, hidden_dim, latent_dim)

    df = sc.pickleFile(ds_path).toDF()

    tfidf_df  = tfidf(df, cv_model, 'paragraph', 'tfidf')
    tfidf_rdd = tfidf_df.rdd.repartition(8000)

    tfidf_rdd = tfidf_rdd.mapPartitions(lambda p: hash_partition(p, vdsh_loader))
    tfidf_rdd.saveAsPickleFile(output_path)
    