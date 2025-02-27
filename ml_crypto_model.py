# Step 1: Prepare Spark Session and Load Data
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder.appName('CryptoML').getOrCreate()

# Step 2: Load each cryptocurrency dataset into Spark DataFrames
btc_df = spark.read.csv('BTC_daily_prices_last_year.csv', header=True, inferSchema=True)
eth_df = spark.read.csv('ETH_daily_prices_last_year.csv', header=True, inferSchema=True)
xrp_df = spark.read.csv('XRP_daily_prices_last_year.csv', header=True, inferSchema=True)
sol_df = spark.read.csv('SOL_daily_prices_last_year.csv', header=True, inferSchema=True)
doge_df = spark.read.csv('DOGE_daily_prices_last_year.csv', header=True, inferSchema=True)
ada_df = spark.read.csv('ADA_daily_prices_last_year.csv', header=True, inferSchema=True)




# Step 3: Rename price columns for clarity before joining
btc_df = btc_df.select(col('date'), col('price_usd').alias('BTC_price'))
eth_df = eth_df.select(col('date'), col('price_usd').alias('ETH_price'))
xrp_df = xrp_df.select(col('date'), col('price_usd').alias('XRP_price'))
sol_df = sol_df.select(col('date'), col('price_usd').alias('SOL_price'))
doge_df = doge_df.select(col('date'), col('price_usd').alias('DOGE_price'))
ada_df = ada_df.select(col('date'), col('price_usd').alias('ADA_price'))

# Step 4: Merge DataFrames into a single DataFrame
crypto_df = btc_df \
    .join(ada_df, 'date') \
    .join(eth_df, 'date') \
    .join(doge_df, 'date') \
    .join(sol_df, 'date') \
    .join(xrp_df, 'date') 
    
    


# Step 5: Verify combined data
crypto_df.show(10)

# Step 6: Prepare DataFrame for ML
# For simplicity, we assume BTC_price as target and others as predictors
from pyspark.ml.feature import VectorAssembler

feature_columns = ['ADA_price', 'DOGE_price', 'ETH_price', 'SOL_price', 'XRP_price']
assembler = VectorAssembler(inputCols=feature_columns, outputCol='features')

crypto_ml_df = assembler.transform(crypto_df).select('features', 'BTC_price')

# Display the final ML-ready DataFrame
crypto_ml_df.show(5)

# Select relevant columns
crypto_corr_df = crypto_df.select(
    'BTC_price', 'ADA_price', 'DOGE_price', 'ETH_price', 'SOL_price', 'XRP_price'
)

# Convert Spark DataFrame to Pandas for plotting
corr_pd = crypto_corr_df.toPandas()

# Compute correlation matrix
corr_matrix = corr_pd.corr()



import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 7))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Cryptocurrency Price Correlation Heatmap', fontsize=15)
plt.show()


# Feature Scaling (Standardization)

from pyspark.ml.feature import VectorAssembler, StandardScaler

feature_cols = ['ADA_price', 'DOGE_price', 'ETH_price', 'SOL_price', 'XRP_price']

# Assemble features
assembler = VectorAssembler(inputCols=feature_cols, outputCol='features_unscaled')
assembled_df = assembler.transform(crypto_df)

# Scale features
scaler = StandardScaler(inputCol='features_unscaled', outputCol='features_scaled', withMean=True, withStd=True)
scaler_model = scaler.fit(assembled_df)
scaled_df = scaler_model.transform(assembled_df)

# Inspect the scaled features clearly
scaled_df.select('date', 'features_scaled', 'BTC_price').show(10, truncate=False)


from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.sql.functions import col
from pyspark.ml.functions import vector_to_array  # Converts vectors to arrays
from pyspark.ml.regression import LinearRegression

# Define feature columns (ensure these match your dataset)
feature_cols = ['ADA_price', 'DOGE_price', 'ETH_price', 'SOL_price', 'XRP_price']

# Assemble feature vectors
assembler = VectorAssembler(inputCols=feature_cols, outputCol='features_unscaled')
assembled_df = assembler.transform(crypto_df)

# Apply standard scaling
scaler = StandardScaler(inputCol='features_unscaled', outputCol='features_scaled', withMean=True, withStd=True)
scaler_model = scaler.fit(assembled_df)

# Save the scaler model
scaler_model.save("/home/osbdet/notebooks/mba1/DATA ARCHITECTURES II/Test/scaler_model")
print("✅ StandardScaler model has been saved successfully!")

# Transform the dataset using the trained scaler
scaled_df = scaler_model.transform(assembled_df)

# Convert sparse vector to dense array for easy indexing
scaled_df = scaled_df.withColumn('features_array', vector_to_array('features_scaled'))

# Extract individual scaled features into separate columns
for i, f in enumerate(feature_cols):
    scaled_df = scaled_df.withColumn(f"{f}_scaled", col("features_array")[i])

# Prepare final dataset for model training
final_df = scaled_df.select(
    col("features_scaled").alias("features"), 
    col("BTC_price").alias("label")
)

# Split data into Training and Test sets
train_df, test_df = final_df.randomSplit([0.8, 0.2], seed=42)


import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYSPARK_SUBMIT_ARGS"] = "--conf spark.executor.extraLibraryPath=/usr/lib/x86_64-linux-gnu/ openblas"

# Train Linear Regression Model
lr = LinearRegression(featuresCol='features', labelCol='label', regParam=0.01)
lr_model = lr.fit(train_df)

# Save the model in the correct folder
lr_model.save("/home/osbdet/notebooks/mba1/DATA ARCHITECTURES II/Test/lr_model")
print("✅ Linear Regression model has been saved successfully!")


# 4. Evaluate model predictions

predictions = lr_model.transform(test_df)

from pyspark.ml.evaluation import RegressionEvaluator

# Initialize the evaluator for RMSE, R², and MAE
evaluator = RegressionEvaluator(labelCol='label', predictionCol='prediction')

# Calculate RMSE
rmse = evaluator.evaluate(predictions, {evaluator.metricName: 'rmse'})

# Calculate R²
r2 = evaluator.evaluate(predictions, {evaluator.metricName: 'r2'})

# Calculate MAE
mae = evaluator.evaluate(predictions, {evaluator.metricName: 'mae'})

# Print the evaluation metrics
print(f'RMSE: {rmse:.2f}')
print(f'R²: {r2:.4f}')
print(f'MAE: {mae:.2f}')



#5. Inspect predictions

predictions.select("features", "label", "prediction").show(365, truncate=False)


# 📌 Step-by-Step Guide to Model Interpretation and Visualization:


# Step 1: Collect Predictions for Visualization

predictions_pd = predictions.select('label', 'prediction').toPandas()



# Step 2: Plot Actual vs. Predicted Prices Clearly

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 7))
sns.scatterplot(x='label', y='prediction', data=predictions_pd, color='blue', alpha=0.6)
plt.plot([predictions_pd['label'].min(), predictions_pd['label'].max()],
         [predictions_pd['label'].min(), predictions_pd['label'].max()],
         color='red', linewidth=2, linestyle='--')

plt.title('Actual vs Predicted Bitcoin Prices', fontsize=16)
plt.xlabel('Actual BTC Price', fontsize=14)
plt.ylabel('Predicted BTC Price', fontsize=14)
plt.grid(True)
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Extract coefficients from the trained model
coefficients = lr_model.coefficients

# Create a DataFrame for feature importance
feature_importance = pd.DataFrame({
    'Feature': ['ADA_price', 'DOGE_price', 'ETH_price', 'SOL_price', 'XRP_price'],
    'Coefficient': coefficients
}).sort_values(by='Coefficient', ascending=False)

# Plot the feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=feature_importance, hue='Feature', legend=False, palette='viridis')
plt.title('Feature Importance (Regression Coefficients)', fontsize=16)
plt.xlabel('Coefficient Value', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.grid(True)
plt.show()


