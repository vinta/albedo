package ws.vinta.albedo.evaluators

import org.apache.spark.ml.evaluation.Evaluator
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.mllib.evaluation.RankingMetrics
import org.apache.spark.sql.{DataFrame, Dataset, Row}

class RankingEvaluator(override val uid: String, val userActualItemsDF: DataFrame)
  extends Evaluator with DefaultParamsWritable {

  def this(userActualItemsDF: DataFrame) = {
    this(Identifiable.randomUID("rankingEvaluator"), userActualItemsDF)
  }

  val metricName = new Param[String](this, "metricName", "評估方式")

  def getMetricName: String = $(metricName)

  def setMetricName(value: String): this.type = set(metricName, value)
  setDefault(metricName -> "ndcg@k")

  val k = new Param[Int](this, "k", "只評估前 k 個 items 的排序結果")

  def getK: Int = $(k)

  def setK(value: Int): this.type = set(k, value)
  setDefault(k -> 15)

  override def isLargerBetter: Boolean = $(metricName) match {
    case "map" => true
    case "ndcg@k" => true
    case "precision@k" => true
  }

  override def evaluate(dataset: Dataset[_]): Double = {
    import dataset.sparkSession.implicits._

    val userPredictedItemsDF = dataset.select($"user_id", $"recommendations.repo_id".alias("items"))

    val bothItemsRDD = userPredictedItemsDF.join(userActualItemsDF, Seq("user_id", "user_id"))
      .select(userPredictedItemsDF.col("items"), userActualItemsDF.col("items"))
      .rdd
      .map((row: Row) => {
        val userPredictedItems = row(0).asInstanceOf[Seq[Int]].slice(0, $(k))
        val userActualItems = row(1).asInstanceOf[Seq[Int]].slice(0, $(k))
        (userPredictedItems.toArray, userActualItems.toArray)
      })

    val rankingMetrics = new RankingMetrics(bothItemsRDD)
    val metric = $(metricName) match {
      case "map" => rankingMetrics.meanAveragePrecision
      case "ndcg@k" => rankingMetrics.ndcgAt($(k))
      case "precision@k" => rankingMetrics.precisionAt($(k))
    }
    metric
  }

  override def copy(extra: ParamMap): RankingEvaluator = {
    defaultCopy(extra)
  }
}