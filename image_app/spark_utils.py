from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from django.conf import settings
from pyspark.ml.feature import Normalizer, BucketedRandomProjectionLSHModel
import numpy as np
import os


# Initialize Spark
def get_spark_session():
    if 'spark' not in globals():
        global spark
        spark = SparkSession.builder \
            .appName("ImageRetrievalApp") \
            .config("spark.driver.memory", "4g")  \
            .config("spark.executor.memory", "4g") \
            .config("spark.executor.cores", "3")   \
            .getOrCreate()
    return spark


get_spark_session()


def stop_spark_session():
    global spark
    if spark:
        spark.stop()
        del globals()['spark']


model = BucketedRandomProjectionLSHModel.load(os.path.join(settings.BASE_DIR,
                                                           "image_app/data/brp_lsh_model"))


def perform_query(query_embedding):
    global spark

    df = spark.read.parquet(os.path.join(
        settings.BASE_DIR, "image_app/data/dataframe"))
    transformed_df = model.transform(df)

    query_df = spark.createDataFrame(
        [(0, Vectors.dense(query_embedding))], ["id", "features"])

    # Normalize features
    normalizer = Normalizer(inputCol="features",
                            outputCol="normalizedFeatures", p=2.0)
    query_df = normalizer.transform(query_df)

    nearest_neighbors = model.approxNearestNeighbors(
        transformed_df, query_df.first()["normalizedFeatures"], 20)

    neighbors = nearest_neighbors.select("image_path", "distCol").collect()

    return neighbors
