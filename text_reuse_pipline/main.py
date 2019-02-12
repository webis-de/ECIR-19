import pyspark
import os
import sys
import argparse
import importlib

if os.path.exists('text_reuse_pipline.zip'):
    sys.path.insert(0, 'text_reuse_pipline.zip')
else:
    sys.path.insert(0, './text_reuse_pipline.zip')

parser = argparse.ArgumentParser()
parser.add_argument('--job', type=str, required=True)
parser.add_argument('--job_args', nargs='*')
args = parser.parse_args()

print(args)

conf = (pyspark.SparkConf().setAppName(args.job).set("spark.network.timeout", 300)\
    .set("spark.executor.extraJavaOptions", "-XX:+UseCompressedOops -XX:+UseG1GC -XX:+UseStringDeduplication -Dio.netty.leakDetection.level=advanced")\
    .set('spark.dynamicAllocation.enabled', False)
)
sc   = pyspark.SparkContext(conf = conf)
sqlC = pyspark.SQLContext(sc)

job_module = importlib.import_module('text_reuse_pipline.jobs.%s' % args.job)
job_module.run(sc, args.job_args)