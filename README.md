# Text Reuse Pipeline

To build the Text reuse pipeline python module, run the following command:

    make build
 This will generate a `dist` folder. All the tasks could be run under the `dist` folder.

## Wikipedia dump extraction

Using the open source [tool](https://github.com/attardi/wikiextractor), we run the following command to extract the text from the Wiki text:

    python wiki_extractor.py -b 30G -s -ns 0 --filter_disambig_pages -o wiki_no_lists  enwiki-20160501-pages-articles.xml.bz2 &

The output should be copied to hdfs files using the following command: `hdfs dfs -put ./path_of_the_extracted_dump ./text-reuse/wiki_00`

## Wikipedia Text preprocessing

To perform text preprocessing on Wikipedia, Run the following two commands inside webis docker [image
](http://gitlab.webis.de). Assuming that the extracted Wikipedia dump is located under the following hdfs path: `text-reuse/wiki_00`
	

	## To perform paragraph re-balancing and cleaning up text
    PYSPARK_DRIVER_PYTHON=python3 spark-submit --master yarn --deploy-mode cluster --num-executors 100 --executor-cores 10 --executor-memory 25g --driver-memory 25g --conf spark.driver.maxResultSize=15g --conf spark.yarn.executor.memoryOverhead=25000 --conf spark.yarn.driver.memoryOverhead=25000 --packages com.databricks:spark-xml_2.11:0.4.1,com.databricks:spark-csv_2.10:1.5.0 --py-files ./text_reuse_pipeline.zip main.py --job wiki_preprocess
    ## The output is written to the following hdfs path: `text-reuse/pipeline/wiki_preprocessed

    ## To extract TFIDF vectors for each article
    PYSPARK_DRIVER_PYTHON=python3 spark-submit --master yarn --deploy-mode cluster --num-executors 100 --executor-cores 10 --executor-memory 25g --driver-memory 25g --conf spark.driver.maxResultSize=15g --conf spark.yarn.executor.memoryOverhead=25000 --conf spark.yarn.driver.memoryOverhead=25000 --packages com.databricks:spark-xml_2.11:0.4.1,com.databricks:spark-csv_2.10:1.5.0 --py-files ./text_reuse_pipeline.zip main.py --job wiki_represent --job_args tfidf
    ## The output is written to the following hdfs path: text-reuse/pipeline/wiki_rep_tfidf



## Wikipedia candidate elimination

To extract candidate articles from Wikipedia to be examined in the last subtask, we run the following command:

    PYSPARK_DRIVER_PYTHON=python3 spark-submit --master yarn --deploy-mode cluster --num-executors 200 --executor-cores 10 --executor-memory 25g --driver-memory 25g --conf spark.driver.maxResultSize=15g --conf spark.yarn.executor.memoryOverhead=25000 --conf spark.yarn.driver.memoryOverhead=25000 --packages com.databricks:spark-xml_2.11:0.4.1,com.databricks:spark-csv_2.10:1.5.0 --py-files ./text_reuse_pipeline.zip main.py --job wiki_candidate_elemination --job_args hdfs://betaweb020:8020/user/sile2804/cython_utils.so 0-10 0.025

This candidate elimination is divided into 100 batches. In the command 0-10 means perform candidate elimination on the batches from 0 to 10. The last argument is 0.025 is the threshold that we consider as the minimum similarity between two documents to be considered similar.

## Wikipedia text alignments
To run the detailed alignments on the candidate elimination output. Run the following command:

    PYSPARK_DRIVER_PYTHON=python3 spark-submit --master yarn --deploy-mode cluster --num-executors 200 --executor-cores 2 --executor-memory 45g --driver-memory 45g --conf spark.driver.maxResultSize=15g --packages com.databricks:spark-xml_2.11:0.4.1,com.databricks:spark-csv_2.10:1.5.0 --jars /home/sile2804/picapica.jar --py-files ./text_reuse_pipeline.zip main.py --job wiki_text_alignment --job_args text-reuse/pipeline/candidates/[0-2] text-reuse/pipeline/alignments/output_name threshold k &

Arguments for this command:
-  text-reuse/pipeline/candidates/[0-2] : hdfs input path (here it means run on parts 0-2 of the whole dataset) 
 - text-reuse/pipeline/alignments/output_name: hdfs output path
 - threshold is the minimum similarity threshold
 - k number of misses threshold

