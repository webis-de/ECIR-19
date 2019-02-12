from text_reuse_pipline.preprocess.wikipedia import generate_text_paragraphs, normalize_paragraphs, flatten_paragraphs, clean_up_text, filter_paras_by_length
from text_reuse_pipline.preprocess.web import collect_docs, remove_headers, remove_footers
from text_reuse_pipline.helpers import utility

from pyspark.sql.types import *
from pyspark.sql.functions import udf, col, when, size
from pyspark import SparkConf, SparkContext
from pyspark.sql.context import SQLContext
import nltk
from nltk.corpus import stopwords

def run(sc, args):
    min_len = 3
    merging_threshold = 50
    min_tokens = 20

    main_content_min_tokens = 15
    main_content_min_stopwords = 4

    nltk.data.path.append('./')
    nltk.download('stopwords', download_dir='./')
    en_stopwords = set(stopwords.words('english'))

    puncs = [',', '.']

    input_file  = args[0]
    output_path = args[1]
    lang_detect_model_name = args[2] if len(args)>2 else None

    sqlC = SQLContext(sc)

    #Load web paragraphs and collect them into documents
    web = sc.textFile(input_file)
    web = web.repartition(8000)
    web = web.mapPartitions(lambda x: collect_docs(x, lang_detect_model_name, 'en'))
    
    #convert into df
    web_df = web.toDF(['d_url', 'paragraphs'])

    #Filter out these very short paragraphs
    filter_udf = udf(lambda paras: list(filter(lambda p: len(p.split()) >=min_len, paras)), ArrayType(StringType()))  
    web_df = web_df.withColumn('paragraphs', filter_udf('paragraphs'))

    #Filter out articles with empty paragraphs
    web_df = web_df.filter(size(col('paragraphs')) > 0)

    #Remove any header/footers
    remove_headers_udf = udf(lambda paras: remove_headers(paras, en_stopwords, puncs, main_content_min_tokens, main_content_min_stopwords), ArrayType(StringType()))
    remove_footers_udf = udf(lambda paras: remove_footers(paras, en_stopwords, puncs, main_content_min_tokens, main_content_min_stopwords), ArrayType(StringType()))
    web_df = web_df.withColumn('paragraphs', remove_headers_udf('paragraphs'))
    web_df = web_df.withColumn('paragraphs', remove_footers_udf('paragraphs'))
    web_df = web_df.filter(size(col('paragraphs')) > 0)

    #normalize paragraphs by merging paragraphs shorter than 50 tokens
    web_df = normalize_paragraphs(web_df, 'paragraphs', 'paragraphs', merging_threshold)

    #flatten docs into paragraphs
    web_df = flatten_paragraphs(web_df, 'paragraphs', 'paragraph')

    #clean-up the text 
    web_df = clean_up_text(web_df, 'paragraph', 'paragraph')

    #filter only paragraphs that are longer than a sentence
    web_df = filter_paras_by_length(web_df, 'paragraph', min_tokens)

    #generate integer ids for the pages by hashing their url
    gen_id_udf = udf(lambda x: utility.hash_string(x), LongType())
    web_df = web_df.withColumn('d_id', gen_id_udf('d_url'))

    #rename columns and save the df
    web_df = web_df.selectExpr('para_id as p_id', 'd_id', 'd_url', 'paragraph')
    web_df.rdd.saveAsPickleFile(output_path)