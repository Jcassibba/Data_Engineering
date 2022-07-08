# this program is for data engineering and machine learning course

#import

from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vector
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifer
from pyspark.ml.evaluation import MultiClassClassificationEvaluator


#load

spark = SparkSession.builder.getOrCreate()
data = spark.read.csv("hmp.csv", inferSchema=True, header=True)

#convert csv to parquet

df.write.parquet(data)

#explore

df.show()
df.count()
len(df.cloumns)
df.printSchema()
df.describe().show()
c = str(df.groupBy("Brush_teeth").count()) 

#Vector tables

input_cols = ["brush_teeth"]
vec_assembler = VectorAssembler(inputcols = input_cols, outputCol = "features")
final_data = vec_assembler.transform(df)
final_data.show()

#perform intital k-mean / training

kmeans = KMeans(featuresCol = "features", k=int(c))

#model deploy to watson machine learning

model = kmeans.fit(final_data)

#inference

inference = model.trasform(final_data)

#hyperParameter tuning

model.transform(final_data).groupBy("prediction").count().show()

#resample data

df_resampled = df.groupBy("features", "brush_teeth").count()

#random Forest Classifiation

rf = RandomForestClassifier(labelCol = "label", featureCol = "features")
rf_Model=rf.fit(df_resampled)
rf_Model.transform(df_resampled).groupBy("prediction").count().show()
