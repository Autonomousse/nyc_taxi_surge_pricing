import os
import getpass
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark import StorageLevel
import pandas as pd

# -----------------------------------------------------------------------------------
# SETUP
# -----------------------------------------------------------------------------------
user = getpass.getuser()
base_path = f'/expanse/lustre/projects/uci157/{user}/nyc_taxi_surge_pricing/'
parquet_path = f'{base_path}df_model_parquet'
spark_temp = f'{base_path}spark_temp'
os.makedirs(spark_temp, exist_ok=True)

# failed with 8 cores and 128 GB RAM
'''
spark = SparkSession.builder \
    .config("spark.driver.maxResultSize", "8g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.memory.fraction", "0.8") \
    .config("spark.memory.storageFraction", "0.3") \
    .config("spark.local.dir", spark_temp) \
    .config("spark.driver.extraJavaOptions",
            f"-Djava.io.tmpdir={spark_temp}") \
    .getOrCreate()
'''

# running with 32 cores and 200GB RAM
spark = SparkSession.builder \
    .config("spark.driver.maxResultSize", "32g") \
    .config("spark.sql.shuffle.partitions", "265") \
    .config("spark.default.parallelism", "62") \
    .config("spark.memory.fraction", "0.8") \
    .config("spark.memory.storageFraction", "0.3") \
    .config("spark.local.dir", spark_temp) \
    .config("spark.driver.extraJavaOptions",
            f"-Djava.io.tmpdir={spark_temp}") \
    .getOrCreate()
# -----------------------------------------------------------------------------------
# READ FROM PARQUET
# -----------------------------------------------------------------------------------
print("Reading df_model from parquet...")
df_model = spark.read.parquet(parquet_path)
print(f"Loaded {df_model.count():,} rows from parquet")

# -----------------------------------------------------------------------------------
# CLASS BALANCE AND WEIGHTS
# -----------------------------------------------------------------------------------
# Already did this prior to writing to parquet
#df_model = df_model.filter(F.col("is_surge").isNotNull())

counts = df_model.groupBy("is_surge").count().collect()
total_rows = sum(row["count"] for row in counts)
count_dict = {row["is_surge"]: row["count"] for row in counts}

surge_count = count_dict[1]
non_surge_count = count_dict[0]

print(f"Total rows: {total_rows:,}")
for row in counts:
    print(f"is_surge={row['is_surge']}: {row['count']:,} ({round(row['count']/total_rows*100, 2)}%)")

weight_surge = total_rows / (2 * surge_count)
weight_non_surge = total_rows / (2 * non_surge_count)
print(f"Surge weight: {weight_surge:.4f}")
print(f"Non-surge weight: {weight_non_surge:.4f}")

df_model = df_model.withColumn(
    "class_weight",
    F.when(F.col("is_surge") == 1, weight_surge).otherwise(weight_non_surge)
)

# -----------------------------------------------------------------------------------
# FEATURE SETUP
# -----------------------------------------------------------------------------------
feature_cols = [
    "PULocationID", "DOLocationID", "PU_borough_index", "DO_borough_index",
    "license_index", "trip_time",
    "has_toll", "has_airport_fee", "has_congestion_surcharge",
    "hour_of_day", "day_of_week", "month", "is_weekend",
    "wait_time_secs", "demand_zscore", "high_demand"
]

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features",
    handleInvalid="skip"
)
df_assembled = assembler.transform(df_model)

# -----------------------------------------------------------------------------------
# TRAIN / VALIDATION / TEST SPLIT
# -----------------------------------------------------------------------------------
train_df, remaining_df = df_assembled.randomSplit([0.70, 0.30], seed=42)
val_df, test_df = remaining_df.randomSplit([0.50, 0.50], seed=42)

train_df.persist(StorageLevel.DISK_ONLY)
val_df.persist(StorageLevel.DISK_ONLY)
test_df.persist(StorageLevel.DISK_ONLY)
train_df.count()
val_df.count()
test_df.count()
print("Train/val/test splits persisted")

# -----------------------------------------------------------------------------------
# RANDOM FOREST MODEL
# -----------------------------------------------------------------------------------
rf = RandomForestClassifier(
    labelCol="is_surge",
    featuresCol="features",
    weightCol="class_weight",
    numTrees=10,       # number of trees in the forest
    maxDepth=12,        # deeper than DTC since ensemble handles overfitting
    maxBins=265,        # covers all 265 taxi zones
    featureSubsetStrategy="auto",  # sqrt(numFeatures) for classification
    seed=42
)

print("Training Random Forest...")
rf_model = rf.fit(train_df)
print("Training complete")

# -----------------------------------------------------------------------------------
# EVALUATORS
# -----------------------------------------------------------------------------------
binary_evaluator = BinaryClassificationEvaluator(
    labelCol="is_surge",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

multi_evaluator = MulticlassClassificationEvaluator(
    labelCol="is_surge",
    predictionCol="prediction"
)

# -----------------------------------------------------------------------------------
# TRAINING EVALUATION
# -----------------------------------------------------------------------------------
print("Evaluating on training set...")
train_predictions = rf_model.transform(train_df)
train_auc = binary_evaluator.evaluate(train_predictions)
print(f"Training AUC-ROC: {train_auc:.4f}")

# -----------------------------------------------------------------------------------
# VALIDATION EVALUATION
# -----------------------------------------------------------------------------------
print("Evaluating on validation set...")
val_predictions = rf_model.transform(val_df)
val_auc = binary_evaluator.evaluate(val_predictions)
print(f"Validation AUC-ROC: {val_auc:.4f}")
print(f"Gap: {train_auc - val_auc:.4f}")

for metric in ["accuracy", "f1", "weightedPrecision", "weightedRecall"]:
    score = multi_evaluator.evaluate(val_predictions, {multi_evaluator.metricName: metric})
    print(f"{metric}: {score:.4f}")

# -----------------------------------------------------------------------------------
# FEATURE IMPORTANCE
# -----------------------------------------------------------------------------------
feature_importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": rf_model.featureImportances.toArray()
}).sort_values("importance", ascending=False)
print(feature_importance.to_string())

# -----------------------------------------------------------------------------------
# FINAL TEST EVALUATION - only run once hyperparameters are finalized
# -----------------------------------------------------------------------------------
# Uncomment only when completely done tuning
# print("Evaluating on test set...")
# test_predictions = rf_model.transform(test_df)
# test_auc = binary_evaluator.evaluate(test_predictions)
# print(f"Test AUC-ROC: {test_auc:.4f}")
# for metric in ["accuracy", "f1", "weightedPrecision", "weightedRecall"]:
#     score = multi_evaluator.evaluate(test_predictions, {multi_evaluator.metricName: metric})
#     print(f"Test {metric}: {score:.4f}")