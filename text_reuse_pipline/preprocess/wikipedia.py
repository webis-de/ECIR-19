import argparse
import logging
import operator
import re
from pyspark.sql.types import *
from pyspark.sql.functions import udf, col, when, size
from sklearn.feature_extraction.text import CountVectorizer
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StopWordsRemover

import numpy as np
from pyspark.mllib.stat import Statistics
import time
import sys
import os
import json
import gzip


def extract_categories_from_wiki_text(text):
    if text is None:
        return []

    textStartInd  = text.find("<text")
    catStartIndex = text.find("[[Category:")
    
    if textStartInd > catStartIndex:
        return []
    
    if catStartIndex != -1:
        catEndIndex = text.find("</text>")
        categories  = text[catStartIndex:catEndIndex]
        result = categories.replace("[[Category:", "").replace("]]", "").replace("]", "").replace(",", " ").replace("\r", ",").replace("\n", ",");
        result = result.split(',')
        return result
    else:
        return []

# Takes wiki_dump and extracted wiki and 
# generate a dataframe contains article_id, text as paragraphs, array of categories
# wiki_dump_df = sqlC.read.format('com.databricks.spark.xml').options(rowTag='page').load(wiki_dump_path)
# wiki_text_df = sqlC.read.format('com.databricks.spark.xml').options(
#            charset="UTF-8", nullValue="", rootTag='docs', rowTag='doc', mode='DROPMALFORMED'
#        ).load(wiki_text_path)
def extract_categories(wiki_dump_df, wiki_text_df, output_path):
    #Extract categories from wiki dump
    categories_extractor = udf(extract_categories_from_wiki_text, ArrayType(StringType()))
    isEmpty = udf(lambda x: len(x) == 0, BooleanType())
    wiki_dump_df = wiki_dump_df.where(wiki_dump_df.ns == 0)
    wiki_dump_df = wiki_dump_df.select("id", categories_extractor("revision.text._VALUE").alias("categories"))
    wiki_dump_df = wiki_dump_df.where(~isEmpty('categories'))
    wiki_dump_df = wiki_dump_df.select('id', 'categories')


    #Merge categories with text
    wiki_df = wiki_text_df.join(wiki_dump_df, on=['id', 'id'], how='left_outer')
    wiki_df.rdd.saveAsPickleFile(output_path)

    return wiki_df

#Extract lines from text into either lists/texts
def extract_paras(content, t='text'):
    pars = content.split('\n')
    
    #Structure each paragraph into text and type
    if t == 'list':
        pars = list(filter(lambda p: p.startswith('LIST-') or p.startswith('LIST#'), pars))
    else:
        pars = list(filter(lambda p: not(p.startswith('LIST-') or p.startswith('LIST#')), pars))
    
    #Skip the first line which is the title and empty string
    def filter_in(p):
        tokens = p.split()
        return len(tokens) > 0
    
    pars = list(filter(lambda x: filter_in(x), pars[1:]))
    return pars


#Performs tokenization and stop word removal using
#Nltk library
def nltk_process(text):
    from nltk import word_tokenize
    from nltk.corpus import stopwords

    tokens = word_tokenize(text)
    clean_tokens = [token for token in tokens if token not in stopwords.words('english')]

    return clean_tokens

#Generate only text paragraphs from text
def generate_text_paragraphs(df, input_clm, output_clm):
    #extract_list_paras_udf = udf(lambda x: extract_pars(x, 'list'), ArrayType(StringType()))
    extract_text_paras_udf = udf(lambda x: extract_paras(x, 'text'), ArrayType(StringType()))
    df = df.withColumn(output_clm, extract_text_paras_udf(df[input_clm]))
    return df


def normalize_paragraphs(df, input_clm, output_clm, min_tokens):
    
    def merge_short_paragraphs(paras, min_tokens):
        output_paras = []
        i = 1
        last_para_text_idx = 0
        output_paras.append(paras[0])
        while i < len(paras)-1:
            if len(paras[i].split()) < min_tokens or \
               len(output_paras[last_para_text_idx].split()) < min_tokens:
                output_paras[last_para_text_idx] = output_paras[last_para_text_idx] + ' ' + paras[i]
            else:
                output_paras.append(paras[i])
                last_para_text_idx = len(output_paras)-1 
            i+=1
        
        return output_paras

    merge_short_paragraphs_udf = udf(lambda x: merge_short_paragraphs(x, min_tokens), ArrayType(StringType()))
    df = df.withColumn(output_clm, merge_short_paragraphs_udf(input_clm))
    
    return df

def info_paragraphs(df, clm):
    df = df.where(col(clm).isNotNull())
    paragraphs = df.rdd.flatMap(lambda x: getattr(x, clm)).filter(lambda p: p != None) 
    paragraphs = paragraphs.map(lambda p: np.array(len(p.split())))
    summary = Statistics.colStats(paragraphs)

    return summary


#TODO move them to utility file
def extract(row, key):
    """Takes dictionary and key, returns tuple of (dict w/o key, dict[key])."""
    _dict = row.asDict()
    _list = _dict[key]
    del _dict[key]
    return (_dict, _list)


def add_to_dict(_dict, key, value):
    _dict[key] = value
    return _dict

def flatten_paragraphs(df, input_clm, output_clm):
    from pyspark.sql.functions import monotonically_increasing_id

    rdd = df.rdd
    # preserve rest of values in key, put list to flatten in value
    rdd = rdd.map(lambda x: extract(x, input_clm))
    # make a row for each item in value
    rdd = rdd.flatMapValues(lambda x: x)
    # add flattened value back into dictionary
    rdd = rdd.map(lambda x: add_to_dict(x[0], output_clm, x[1]))

    # convert back to dataframe
    df  = rdd.toDF()

    # adding ids for the paragraph
    df  = df.withColumn("para_id", monotonically_increasing_id())

    return df

def clean_up_text(df, input_clm, output_clm):
    
    def cleanup_text(text):
        import re
        cleaned_txt = re.sub('[.,;:{}=\()]+', ' ', text)
        return cleaned_txt

    cleaner_udf = udf(lambda x: cleanup_text(x), StringType())
    df = df.withColumn(output_clm, cleaner_udf(input_clm))

    return df

def filter_paras_by_length(df, input_clm, min_tokens):
    filter_udf = udf(lambda p: len(p.split()) > min_tokens , BooleanType())
    df = df.filter(filter_udf(df[input_clm]))
    return df
    