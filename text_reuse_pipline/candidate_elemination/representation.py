from pyspark.sql import Row
from pyspark.sql.functions import col, size
from pyspark.sql.functions import first, collect_list, collect_set, mean
from pyspark.sql.functions import UserDefinedFunction as udf
from pyspark.sql.types import *
from pyspark.ml.feature import NGram, CountVectorizer, VectorAssembler, Tokenizer, HashingTF, IDF, StopWordsRemover

import numpy as np

def para2vec(sc, df, input_clm, w2vec_lookup):
    #collect word representations and broadcast it
    words_rep = w2vec_lookup.rdd.map(lambda r: (r.word, r.vector)).collectAsMap()
    words_rep_bc = sc.broadcast(words_rep)

    tokenizer = Tokenizer(inputCol=input_clm, outputCol="tokens")
    tokenized_df = tokenizer.transform(df)

    #remove stopwords
    remover      = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
    tokenized_df = remover.transform(tokenized_df)

    def compute_norm(v):
        import numpy as np
        return float(np.linalg.norm(v))

    para_embeding_udf = udf(lambda tokens: paragraph_embeding(tokens, words_rep_bc, False), ArrayType(FloatType()))
    norm_udf = udf(lambda v: compute_norm(v), FloatType())


    paras = tokenized_df.withColumn('para_embeding', para_embeding_udf(tokenized_df.filtered_tokens))
    paras = paras.withColumn('norm', norm_udf(paras.para_embeding))

    return paras

def tfidf(df, input_clm, output_clm, tokens_filter_callback=None, no_features=260000):
    tokenizer = Tokenizer(inputCol=input_clm, outputCol="tokens")
    tokenized_df = tokenizer.transform(df)

    #remove stopwords
    remover      = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
    tokenized_df = remover.transform(tokenized_df)

    #extra filters on tokens
    if tokens_filter_callback != None:
        token_filter_udf = udf(lambda x: tokens_filter_callback(x), ArrayType(StringType()))
        tokenized_df     = tokenized_df.withColumn('filtered_tokens', token_filter_udf('filtered_tokens'))
    
    # use mllib to create the tf-idf vectors
    para_hashingTF = HashingTF(inputCol= "filtered_tokens", outputCol="rawfeatures", numFeatures=no_features)
    para_featurized_df = para_hashingTF.transform(tokenized_df)

    para_idf = IDF(inputCol="rawfeatures", outputCol=output_clm)
    para_idf_model = para_idf.fit(para_featurized_df)

    tfidf_df = para_idf_model.transform(para_featurized_df)

    norm_udf = udf(lambda x: float(x.norm(2)), FloatType())
    tfidf_df = tfidf_df.withColumn('norm', norm_udf(output_clm))

    return tfidf_df

#Create tfidf model over df1 and use it to transform df2
def joint_tfidf(df1, df2, input_clm, output_clm, tokens_filter_callback=None, no_features=260000):
    
    tokenizer = Tokenizer(inputCol=input_clm, outputCol="tokens")
    df1 = tokenizer.transform(df1)
    df2 = tokenizer.transform(df2)

    #remove stopwords
    remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
    df1 = remover.transform(df1)
    df2 = remover.transform(df2)

    #extra filters on tokens
    if tokens_filter_callback != None:
        token_filter_udf = udf(lambda x: tokens_filter_callback(x), ArrayType(StringType()))
        df1 = df1.withColumn('filtered_tokens', token_filter_udf('filtered_tokens'))
        df2 = df2.withColumn('filtered_tokens', token_filter_udf('filtered_tokens'))
    
    # use mllib to create the tf-idf vectors
    para_hashingTF = HashingTF(inputCol= "filtered_tokens", outputCol="rawfeatures", numFeatures=no_features)
    df = para_hashingTF.transform(df1)

    para_idf = IDF(inputCol="rawfeatures", outputCol=output_clm)
    model = para_idf.fit(df)

    df2 = para_hashingTF.transform(df2)
    df1 = para_hashingTF.transform(df1)
    df2 = model.transform(df2)
    df1 = model.transform(df1)

    norm_udf = udf(lambda x: float(x.norm(2)), FloatType())
    df2 = df2.withColumn('norm', norm_udf(output_clm))
    df1 = df1.withColumn('norm', norm_udf(output_clm))

    return df1, df2



def para2vec(sc, df, input_clm, w2vec_lookup):
    #collect word representations and broadcast it
    words_rep = w2vec_lookup.rdd.map(lambda r: (r.word, r.vector)).collectAsMap()
    words_rep_bc = sc.broadcast(words_rep)

    tokenizer = Tokenizer(inputCol=input_clm, outputCol="tokens")
    tokenized_df = tokenizer.transform(df)

    #remove stopwords
    remover      = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
    tokenized_df = remover.transform(tokenized_df)

    def compute_norm(v):
        import numpy as np
        return float(np.linalg.norm(v))

    para_embeding_udf = udf(lambda tokens: _paragraph_embeding(tokens, words_rep_bc, False), ArrayType(FloatType()))
    norm_udf = udf(lambda v: compute_norm(v), FloatType())


    paras = tokenized_df.withColumn('para_embeding', para_embeding_udf(tokenized_df.filtered_tokens))
    paras = paras.withColumn('norm', norm_udf(paras.para_embeding))

    return paras

def stopwords_ngrams(sc, df, input_clm, n):

    stop_words   = ['the', 'of', 'and', 'a', 'in', 'to',
               'is', 'was', 'it', 'for', 'with', 'he', 'be', 'on', 'i', 'that',
               'by', 'at', 'you', '\'s', 'are', 'not','his', 'this', 'from', 'but', 'had',
               'which', 'she', 'they', 'or', 'an', 'were', 'we', 'their', 'been',
               'has', 'have', 'will', 'would', 'her', 'n\'t', 'there', 'can', 'all',
               'as', 'if', 'who', 'what', 'said']

    #Functions to apply the critirion from the paper: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.307.4369&rep=rep1&type=pdf
    C = set(['the', 'of', 'and', 'a', 'in', 'to', '\'s'])

    def maximal_seq(s1, s2):
        l = 0
        for i in range(0, len(s1)):
            if s1[i] not in s2:
                continue
            j=i+1
            while j < len(s1):
                if s1[j] not in s2:
                    break
                j= j+1

            seq_len = j - i
            if seq_len > l:
                l = seq_len
        return l

    def filter_ngrams(ngrams, c, ngram_size): 
        def f(s): 
            return len(set(s).intersection(c)) < (ngram_size - 1) and maximal_seq(s, c) < ngram_size - 2
        
        return list(filter(lambda x: f(x), ngrams))

    def hash_ngram(ngrams):
        import hashlib
        result = list(map(lambda x: int(hashlib.sha1(x.encode()).hexdigest(), 16) % (10 ** 8), ngrams))
        return result

    #Represent each paragraph with its stopwords
    #Tokenize
    tokenizer = Tokenizer(inputCol=input_clm, outputCol="words")
    wiki      = tokenizer.transform(wiki)
    udf_stopwords = udf(lambda x: list(filter(lambda t: t in stop_words, x)), ArrayType(StringType()))
    #Keep only stopwords tokens
    para_df = wiki.withColumn('paragraph_stopwords', udf_stopwords('words'))


    ngram = NGram(n=n, inputCol='paragraph_stopwords', outputCol="{0}_grams".format(n))
    ngram_pipeline = Pipeline(stages= ngram)
    para_df_ngrams = ngram_pipeline.fit(para_df).transform(para_df)


    filter_udf = udf(lambda x: filter_ngrams(x, C, n), ArrayType(StringType()))
    para_df_ngrams = para_df_ngrams.withColumn('{0}_grams_filtered'.format(n), udf_i('{0}_grams'.format(n)))

    hahs_udf = udf(lambda x: hash_ngram(x), ArrayType(IntegerType()))
    para_df_ngrams = para_df_ngrams.withColumn('{0}_grams_hashed'.format(n), udf_i('{0}_grams_filtered'.format(n)))

    return para_df_ngrams


def merge_represenations(rdd1, rdd2, clm1, clm2):
    rdd1 = rdd1.map(lambda x: (x.p_id, (x.d_id, x[clm1])))
    rdd2 = rdd2.map(lambda x: (x.p_id, (x.d_id, x[clm2])))
    
    joined_rdd = rdd1.join(rdd2)
    df = joined_rdd.map(lambda r: (r[0], r[1][0][0], r[1][0][1], r[1][1][1])).toDF(['p_id', 'd_id', clm1, clm2])
    return df