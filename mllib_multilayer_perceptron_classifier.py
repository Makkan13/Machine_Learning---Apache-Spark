from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import os
from pyspark.sql import SparkSession
import time

# Setting enviromental variables
os.environ['PYSPARK_PYTHON'] = '/mnt/c/Users/Μακης/Documents/vscode_python/mlib_spark/venv/bin/python'
os.environ['SPARK_HOME'] = '/mnt/c/Spark/spark-3.5.0-bin-hadoop3'

# Starting Spark session
spark = SparkSession.builder \
    .appName("spark_mlib_multilayer_perceptron_classifier") \
    .config("spark.executor.cores", "1") \
    .config("spark.executor.memory", "8g") \
    .config("spark.executor.instances",'3')\
    .config("spark.driver.cores", "4") \
    .config("spark.driver.memory", "8g") \
    .getOrCreate()

# Load training data
train_data = spark.read.format("libsvm")\
    .load("{}/data/mllib/sample_multiclass_classification_data.txt".format(os.environ['SPARK_HOME']))


# Split the data into train and test
splits = train_data.randomSplit([0.6, 0.4], 1234)
train = splits[0]
test = splits[1]

# Specifying layers for the neural network
layers = [4, 5, 4, 3] # input layer  with 4 hidden input units (features), two intermediate of size 5 and 4 and output of size 3 (classes)

# Creating the trainer and set its parameters
trainer = MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=30, solver='l-bfgs',stepSize=0.03)
# Timing the model's training
start_time = time.time() 
# Training the model
model = trainer.fit(train)
# End_time  of  training
end_time = time.time() 
print(f"PyTorch Training Time: {end_time - start_time:.2f} seconds")

# Computing accuracy on the test set
result = model.transform(test)
predictionAndLabels = result.select("prediction", "label")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))

