from pyspark.sql.functions import col, hour, unix_timestamp
from pyspark.sql import functions as F


class ELTPipeline:
    def __init__(self, spark):
        self.spark = spark

    def transform(self):
        self._create_bronze()
        self._create_silver()
        self._create_gold()

    def _create_bronze(self):
        df = self.spark.read \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .csv("data/nyc_yellow_taxi.csv")

        df = df.drop(
            "pickup_longitude",
            "pickup_latitude",
            "store_and_fwd_flag",
            "dropoff_longitude",
            "dropoff_latitude",
            "payment_type",
            "fare_amount",
            "extra",
            "mta_tax",
            "tip_amount",
            "tolls_amount",
            "improvement_surcharge"
        )

        df = df.na.drop()

        df.write \
            .format("delta") \
            .mode("overwrite") \
            .save("data/bronze/df")

    def _create_silver(self):
        bronze_df = self.spark.read \
            .format("delta") \
            .load("data/bronze/df")

        bronze_df = bronze_df.withColumn(
            "trip_duration_sec",
            unix_timestamp(col("tpep_dropoff_datetime")) -
            unix_timestamp(col("tpep_pickup_datetime"))
        ).withColumn(
            "pickup_hour",
            hour(col("tpep_pickup_datetime"))
        ).drop(
            col("tpep_pickup_datetime")
        ).drop(
            col("tpep_dropoff_datetime")
        )

        bronze_df.write \
            .format("delta") \
            .mode("overwrite") \
            .save("data/silver/df")

    def _create_gold(self):
        silver_df = self.spark.read \
            .format("delta") \
            .load("data/silver/df")

        for hour in range(24):
            silver_df = silver_df.withColumn(f"pickup_hour_{hour}",
                                             F.when(F.col("pickup_hour") == hour, 1).otherwise(0))

        silver_df = silver_df.drop(col("pickup_hour"))

        train_df, test_df = silver_df.randomSplit([0.8, 0.2], seed=42)

        train_df.write \
            .format("delta") \
            .mode("overwrite") \
            .save("data/gold/train")

        test_df.write \
            .format("delta") \
            .mode("overwrite") \
            .save("data/gold/test")
