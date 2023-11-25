import findspark
findspark.init()
from pyspark.sql.functions import from_json, udf, col, lower, regexp_replace, split, when, rand
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, BooleanType
import random
import json
import re
import string
import time
import csv

from io import StringIO
from pyspark.sql import SparkSession, Row
from delta import configure_spark_with_delta_pip

#calculating the start time of executing 
#start_time=time.time()


#Optimization 1. Cluster unable to allocate excecutors, full restart required as executors were held up by unkillable applications
#Added last three .config to allocate memory, force 3 executors and force 4 cores to be utilized
#10GB file
builder = SparkSession.builder.appName("Kryo_normalized_Test") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.executor.cores", "4") \
    .config("spark.executor.instances", "3") \
    .config("spark.executor.memory", "4g") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")  # Use Kryo Serializer
     
#    .config("spark.shuffle.service.enabled", "true") \

spark = configure_spark_with_delta_pip(builder).getOrCreate()

#Upload textfile to HDFS in terminal
##hdfs dfs -copyFromLocal /home/ubuntu/project/Amazon_aa_unstructured.txt hdfs://namenode:9000/user/ubuntu/

text_data = spark.read.text("hdfs:///project/Amazon_aa_unstructured.txt")
text_rdd = text_data.rdd.map(lambda row: row.value)  # Converting to RDD if needed

Startstring = '"overall"'
Endstring = ','

def process_lines(iterator):
    processing = False
    myList = []
    
    for line in iterator:
        if not processing:
            # Start processing when the line contains Startstring
            if Startstring in line:
                processing = True
                myList.append(line)
        else:
            myList.append(line)
            if line.endswith(Endstring):
                processing = False

    return myList

# Apply the function to the RDD
filtered_lines = text_rdd.mapPartitions(process_lines)

# Convert to DataFrame
df = filtered_lines.map(lambda x: Row(value=x)).toDF()

#df.show(3, truncate=False)

#Apply Split to generate required columns
Struct_df = df.select(split(col("value"), ",").getItem(0).alias("Overall"),
                     split(col("value"), ",").getItem(8).alias("ReviewText"))


#Struct_df.show(3, truncate=False)

# Filter the DataFrame to keep rows where 'ReviewText' contains '"reviewText":'
filtered_df = Struct_df.filter(col("ReviewText").contains("\"reviewText\""))
filtered_df = Struct_df.filter(col("Overall").contains("\"overall\""))

#filtered_df.show(3, truncate=False)

# Define patterns to removed
pattern_overall = '\"overall\": '
pattern_reviewtext = '\"reviewText\": '

# Remove the unwanted substrings
cleaned_df = filtered_df.withColumn("Overall", regexp_replace("Overall", pattern_overall, "")) \
               .withColumn("ReviewText", regexp_replace("ReviewText", pattern_reviewtext, ""))

#cleaned_df.show(3, truncate=False)

cleaned_df.write.format("delta").mode("overwrite").save("hdfs:///project/amazon_delta")

from pyspark.ml.feature import HashingTF, Tokenizer, StopWordsRemover
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.base import Transformer
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import IntegerType, StringType, ArrayType
from pyspark.sql.functions import when
from pyspark.sql import functions as F


#fetch the delta table from hdfs
delta_df = spark.read.format("delta").load("hdfs:///project/amazon_delta")
#delta_df.show(3,truncate=False)

#change overall rating that are 0 to 1 for future handling
delta_df = delta_df.withColumn("Overall", when(delta_df["Overall"] == 0, 1).otherwise(delta_df["Overall"]))
#If null value, set rating as 1
delta_df = delta_df.withColumn("Overall", when(delta_df["Overall"].isNull(), 1).otherwise(delta_df["Overall"]))

distribution = delta_df.groupBy("Overall").count().orderBy('Overall')
#distribution.show()

#create a even distribution of reviews over score
min_value = distribution.select(F.min(F.col("count"))).collect()[0][0]
distribution = distribution.withColumn("count_ratio", min_value / F.col("count"))
even_fractions = distribution.select('Overall', 'count_ratio').rdd.collectAsMap()
even_distribution = delta_df.sampleBy('Overall', even_fractions, seed = 1)
even_distribution.groupBy("Overall").count().orderBy('Overall').show()

# Convert 'overall' to binary (0 or 1)
#delta_df = delta_df.withColumn("overall", when(delta_df["overall"] <= 4, 0).otherwise(1).cast(IntegerType()))
delta_df = even_distribution.withColumn("Overall", when(delta_df["Overall"] < 3, 0).otherwise(1))
#delta_df.show(3,truncate=False)

# Tokenize the reviewText
tokenizer = Tokenizer(inputCol="ReviewText", outputCol="token")

# Define the StopWordsRemover
stop_words_remover = StopWordsRemover(inputCol="token", outputCol="words")

# Vectorize the filtered words using HashingTF
hashingTF = HashingTF(inputCol="words", outputCol="features")

# Define the Naive Bayes model
nb = NaiveBayes(featuresCol="features", labelCol="Overall")

# Create a pipeline with Tokenizer, StopWordsRemover, HashingTF, and Naive Bayes
pipeline = Pipeline(stages=[tokenizer, stop_words_remover, hashingTF, nb])

# Split the data into training and testing sets
train_data, test_data = delta_df.randomSplit([0.7, 0.3])

# Train the model
model = pipeline.fit(train_data)

# Make predictions on the test data
predictions = model.transform(test_data)

# Evaluate the model using MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(labelCol="Overall", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy:", accuracy)

# make predictions on new reviewText. 0 - Negative, 1 - Positive
new_review_text = "Thios book had no cohesion, would not recomend. I am dissapointed and upset"
#new_review_text = "I loved this book. It sure lived up to my expectation and WHAT A TWIST"
new_data = spark.createDataFrame([(new_review_text,)], ["ReviewText"])
new_predictions = model.transform(new_data)

# Show the prediction for the new reviewText
new_predictions.select("ReviewText", "prediction").show(truncate=False)


spark.stop()