from text_reuse_pipline.preprocess.wikipedia import generate_text_paragraphs, normalize_paragraphs, flatten_paragraphs, clean_up_text, filter_paras_by_length
from pyspark.sql.functions import udf, col, when, size
from pyspark import SparkConf, SparkContext
from pyspark.sql.context import SQLContext

def run(sc, args):
    min_tokens  = 70
    min_length  = 20
    output_path = 'text-reuse/pipeline/wiki_preprocessed'
    input_file  = 'text-reuse/wiki_00'

    sqlC = SQLContext(sc)

    df = sqlC.read.format('com.databricks.spark.xml').options(charset="UTF-8", nullValue="", rowTag='doc', mode='DROPMALFORMED').load(input_file)
    df = df.selectExpr("_id as id", "_title as title", "content as content")
    df.show()

    #Preprocess Wikipedia

    #generate paragarphs
    df = generate_text_paragraphs(df, 'content', 'paragraphs')
    df.show()

    #Filter out empty paragraphs
    df = df.filter(size(col('paragraphs')) > 0)

    #normalize paragraphs
    df = normalize_paragraphs(df, 'paragraphs', 'paragraphs', min_tokens)
    df.show()

    #flatten docs into paragraphs
    df = flatten_paragraphs(df, 'paragraphs', 'paragraph')
    df.show()

    #clean-up the text 
    df = clean_up_text(df, 'paragraph', 'paragraph')
    df.show()

    #filter only paragraphs that are longer than min_tokens
    df = filter_paras_by_length(df, 'paragraph', min_length)

    #rename columns and save the df
    df = df.selectExpr('para_id as p_id', 'id as d_id', 'paragraph')
    df.rdd.saveAsPickleFile(output_path)