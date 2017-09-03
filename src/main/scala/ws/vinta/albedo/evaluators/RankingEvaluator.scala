package ws.vinta.albedo.evaluators

import org.apache.spark.ml.evaluation.Evaluator
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.mllib.evaluation.RankingMetrics
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.{collect_list, row_number}
import org.apache.spark.sql.types.{ArrayType, IntegerType, StructType}
import org.apache.spark.sql.{Column, DataFrame, Dataset, Row}
import ws.vinta.albedo.utils.SchemaUtils.checkColumnType

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

  val userCol = new Param[String](this, "userCol", "User 所在的欄位名稱")

  def getUserCol: String = $(userCol)

  def setUserCol(value: String): this.type = set(userCol, value)
  setDefault(userCol -> "user")

  val itemsCol = new Param[String](this, "itemsCol", "Items 所在的欄位名稱")

  def getItemsCol: String = $(itemsCol)

  def setItemsCol(value: String): this.type = set(itemsCol, value)
  setDefault(itemsCol -> "items")

  override def isLargerBetter: Boolean = $(metricName) match {
    case "map" => true
    case "ndcg@k" => true
    case "precision@k" => true
  }

  def evaluateSchema(schema: StructType): StructType = {
    checkColumnType(schema, $(userCol), IntegerType)
    checkColumnType(schema, $(itemsCol), ArrayType(IntegerType))

    schema
  }

  override def evaluate(userPredictedItemsDF: Dataset[_]): Double = {
    evaluateSchema(userActualItemsDF.schema)
    evaluateSchema(userPredictedItemsDF.schema)

    val bothItemsRDD = userPredictedItemsDF.join(userActualItemsDF, Seq($(userCol), $(userCol)))
      .select(userPredictedItemsDF.col($(itemsCol)), userActualItemsDF.col($(itemsCol)))
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

object RankingEvaluator {
  def intoUserActualItems(userCol: Column, itemCol: Column, orderByCol: Column, k: Int)(df: Dataset[_]): DataFrame = {
    import df.sparkSession.implicits._

    val windowSpec = Window.partitionBy(userCol).orderBy(orderByCol.desc)
    val userActualItemsDF = df
      .withColumn("row_number", row_number().over(windowSpec))
      .where($"row_number" <= k)
      .groupBy(userCol)
      .agg(collect_list(itemCol).alias("items"))

    userActualItemsDF
  }
}