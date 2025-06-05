import mlflow
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.evaluation import RegressionEvaluator


class MLPipeline:
    def __init__(self, spark):
        self.spark = spark
        self.target_col = 'total_amount'

    def run(self):
        self._load_data()
        pipeline = self._prepare_pipeline()
        model = pipeline.fit(self.train_df)
        self._ml_flow(model)

    def _load_data(self):
        self.train_df = self.spark.read \
            .format("delta") \
            .load("data/gold/train")

        self.test_df = self.spark.read \
            .format("delta") \
            .load("data/gold/test")

    def _prepare_pipeline(self):
        feature_columns = [
            col for col in self.train_df.columns
            if col != self.target_col
        ]
        assembler = VectorAssembler(
            inputCols=feature_columns, outputCol="raw_features")

        scaler = StandardScaler(
            inputCol="raw_features",
            outputCol="scaled_features",
            withStd=True,
            withMean=True
        )

        final_assembler = VectorAssembler(
            inputCols=["scaled_features"],
            outputCol="features"
        )

        lr = LinearRegression(featuresCol="features", labelCol=self.target_col)
        pipeline = Pipeline(stages=[assembler, scaler, final_assembler, lr])

        return pipeline

    def _ml_flow(self, model):
        with mlflow.start_run(run_name="NYC Yellow Taxi Cost Prediction"):

            mlflow.spark.log_model(model, "Linear Regression")

            predictions = model.transform(self.test_df)

            evaluator = RegressionEvaluator(
                labelCol=self.target_col,
                predictionCol="prediction",
                metricName="rmse"
            )

            mlflow.log_param("intercept", model.stages[-1].intercept)

            rmse = evaluator.evaluate(predictions)
            r2 = evaluator.setMetricName("r2").evaluate(predictions)
            mae = evaluator.setMetricName("mae").evaluate(predictions)

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)
