from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip


class Session:
    def __init__(self):
        builder = SparkSession.builder.appName("DeltaLake") \
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
            .config("spark.jars.packages", "io.delta:delta-core_2.12:2.0.0")
        self.spark = configure_spark_with_delta_pip(builder).getOrCreate()

    def get_session(self):
        return self.spark
