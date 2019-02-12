from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections
from pyspark.sql.functions import udf, col, when, size
from pyspark.sql.types import StringType, ArrayType
import numpy as np

def binlist_to_int(blist):
    c = list(map(lambda x: str(x), blist))
    s = ''.join(c)
    return np.uint32(np.int32(int(s,2)))

def create_rbp(dim, L, D):
    rbps = [RandomBinaryProjections('rbp' + str(i), L) for i in range(0,D)]
    # Create engine with pipeline configuration
    engine = Engine(dim, lshashes= rbps)
    return engine

def hash_tfidf(vec, engine):
    vec  = vec.toArray()
    hashes = list(map(lambda lsh: binlist_to_int(lsh.hash_vector(vec)), engine.lshashes))
    return hashes

def hash_df(df, input_clm, output_clm, input_dim, L, D):
    engine = create_rbp(input_dim, L, D)
    hash_udf = udf(lambda v: hash_tfidf(v, engine), ArrayType(StringType()))
    df = df.withColumn(output_clm, hash_udf(input_clm))
    return df