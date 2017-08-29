package ws.vinta.albedo

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.SparkSession

object PersonalizedRankerCV {
  val appName = "LogisticRegressionCV"

  def main(args: Array[String]): Unit = {
    implicit val spark = SparkSession
      .builder()
      .appName(appName)
      .getOrCreate()

    implicit val sc = spark.sparkContext
    sc.setLogLevel("WARN")

    // Cross-validate Models

    val lr = new LogisticRegression()
      .setFeaturesCol("features")
      .setLabelCol("starring")

    val pipeline = new Pipeline()
      .setStages(Array(lr))

    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.maxIter, Array(20))
      .addGrid(lr.regParam, Array(0.0))
      .addGrid(lr.elasticNetParam, Array(0.0))
      .addGrid(lr.standardization, Array(true, false))
      .build()

    val evaluator = new BinaryClassificationEvaluator()
      .setMetricName("areaUnderROC")
      .setRawPredictionCol("rawPrediction")
      .setLabelCol("starring")

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEstimatorParamMaps(paramGrid)
      .setEvaluator(evaluator)
      .setNumFolds(2)

    //val cvModel = cv.fit(vectorTrainingDF)
    //
    //// Show best parameters
    //
    //val bestPipelineModel = cvModel.bestModel.asInstanceOf[PipelineModel]
    //val lrModel = bestPipelineModel.stages(0).asInstanceOf[LogisticRegressionModel]
    //lrModel.explainParams()

    spark.stop()
  }
}