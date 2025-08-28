import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline

# Paths
raw_path = "data/raw/Titanic-Dataset.csv"
processed_path = "data/processed/"

print("Current working directory:", os.getcwd())
print("Looking for raw file at:", os.path.abspath(raw_path))

# Create Spark session
print("Creating Spark session...")
spark = SparkSession.builder \
    .appName("Titanic Preprocessing") \
    .getOrCreate()

print("Reading raw data...")
df = spark.read.csv(raw_path, header=True, inferSchema=True)
df.show(5)

# Fill missing values
median_age = df.approxQuantile("Age", [0.5], 0.0)[0]
df = df.fillna({'Age': median_age, 'Embarked': 'S'})


# df['Age'] = df['Age'].fillna(df['Age'].median())
# df['Embarked'] = df['Embarked'].fillna('S')
# df['Sex'] = df['Sex'].map({'male':0, 'female':1})
# df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# # Split train/test
# X = df.drop(columns=['Survived', 'Name', 'Ticket', 'Cabin', 'PassengerId'])
# y = df['Survived']



# Split train/test
train_ratio = 0.8
df_train, df_test = df.randomSplit([train_ratio, 1 - train_ratio], seed=42)

# Features and target
exclude_cols = ['Survived', 'Name', 'Ticket', 'Cabin', 'PassengerId']
feature_cols = [c for c in df.columns if c not in exclude_cols]
target_col = "Survived"
print("feature_cols==============",feature_cols)
X_train = df_train.select(*feature_cols)
y_train = df_train.select(target_col)
X_test = df_test.select(*feature_cols)
y_test = df_test.select(target_col)


print("Save processed files as CSV*****************************")
# Convert to Pandas and save as CSV (Windows-friendly)
X_train.toPandas().to_csv(os.path.join(processed_path, "X_train.csv"), index=False)
y_train.toPandas().to_csv(os.path.join(processed_path, "y_train.csv"), index=False)
X_test.toPandas().to_csv(os.path.join(processed_path, "X_test.csv"), index=False)
y_test.toPandas().to_csv(os.path.join(processed_path, "y_test.csv"), index=False)

# # Convert categorical column 'Sex' to numeric
# sex_indexer = StringIndexer(inputCol="Sex", outputCol="SexIndex", handleInvalid="keep")

# # One-hot encode 'Embarked'
# embarked_indexer = StringIndexer(inputCol="Embarked", outputCol="EmbarkedIndex", handleInvalid="keep")
# embarked_encoder = OneHotEncoder(inputCol="EmbarkedIndex", outputCol="EmbarkedVec")

# # Assemble features into a single vector
# feature_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'SexIndex', 'EmbarkedVec']
# assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# # Pipeline
# pipeline = Pipeline(stages=[sex_indexer, embarked_indexer, embarked_encoder, assembler])
# model = pipeline.fit(df)
# df_transformed = model.transform(df)

# # Select features and label
# final_df = df_transformed.select(col("features"), col("Survived").alias("label"))

# # Split train/test
# train_df, test_df = final_df.randomSplit([0.8, 0.2], seed=42)

# # Ensure processed directory exists
# os.makedirs(processed_path, exist_ok=True)

# # Save as Parquet (supports VectorUDT natively)
# train_df.write.mode("overwrite").parquet(os.path.join(processed_path, "train.parquet"))
# test_df.write.mode("overwrite").parquet(os.path.join(processed_path, "test.parquet"))

print("Spark preprocessing complete. Files saved at:", processed_path)

# Stop Spark session
spark.stop()
print("Spark session stopped.")
