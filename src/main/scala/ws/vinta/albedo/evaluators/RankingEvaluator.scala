package ws.vinta.albedo.evaluators

import org.apache.spark.ml.evaluation.Evaluator
import org.apache.spark.ml.param.{IntParam, Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.mllib.evaluation.RankingMetrics
import org.apache.spark.sql._
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import ws.vinta.albedo.settings
import ws.vinta.albedo.utils.DatasetUtils._

class RankingEvaluator(override val uid: String, val userActualItemsDF: Dataset[_])
  extends Evaluator with DefaultParamsWritable {

  def this(userActualItemsDF: Dataset[_]) = {
    this(Identifiable.randomUID("rankingEvaluator"), userActualItemsDF)
  }

  val metricName = new Param[String](this, "metricName", "Metric name (supports \"NDCG@k\" (default), \"Precision@k\", and \"MAP\")")

  def getFormattedMetricName: String = $(metricName).replaceAll("@k", s"@${$(k)}")

  def getMetricName: String = $(metricName)

  def setMetricName(value: String): this.type = set(metricName, value)
  setDefault(metricName -> "NDCG@k")

  val k = new IntParam(this, "k", "Evaluate top-k items for every user")

  def getK: Int = $(k)

  def setK(value: Int): this.type = set(k, value)
  setDefault(k -> 15)

  val userCol = new Param[String](this, "userCol", "User column name")

  def getUserCol: String = $(userCol)

  def setUserCol(value: String): this.type = set(userCol, value)
  setDefault(userCol -> "user")

  val itemsCol = new Param[String](this, "itemsCol", "Items column name")

  def getItemsCol: String = $(itemsCol)

  def setItemsCol(value: String): this.type = set(itemsCol, value)
  setDefault(itemsCol -> "items")

  override def isLargerBetter: Boolean = $(metricName) match {
    case "NDCG@k" => true
    case "Precision@k" => true
    case "MAP" => true
  }

  def evaluateSchema(schema: StructType): StructType = {
    def equalsIgnoreNullability(left: DataType, right: DataType): Boolean = {
      (left, right) match {
        case (ArrayType(leftElementType, _), ArrayType(rightElementType, _)) =>
          equalsIgnoreNullability(leftElementType, rightElementType)
        case (MapType(leftKeyType, leftValueType, _), MapType(rightKeyType, rightValueType, _)) =>
          equalsIgnoreNullability(leftKeyType, rightKeyType) && equalsIgnoreNullability(leftValueType, rightValueType)
        case (StructType(leftFields), StructType(rightFields)) =>
          leftFields.length == rightFields.length && leftFields.zip(rightFields).forall { case (l, r) =>
            l.name == r.name && equalsIgnoreNullability(l.dataType, r.dataType)
          }
        case (l, r) => l == r
      }
    }

    Map($(userCol) -> IntegerType, $(itemsCol) -> ArrayType(IntegerType))
      .foreach{
        case(columnName: String, expectedDataType: DataType) => {
          val actualDataType = schema(columnName).dataType
          require(equalsIgnoreNullability(actualDataType, expectedDataType), s"Column $columnName must be $expectedDataType but got $actualDataType.")
        }
      }

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
      case "NDCG@k" => rankingMetrics.ndcgAt($(k))
      case "Precision@k" => rankingMetrics.precisionAt($(k))
      case "MAP" => rankingMetrics.meanAveragePrecision
    }
    metric
  }

  override def copy(extra: ParamMap): RankingEvaluator = {
    defaultCopy(extra)
  }
}

object RankingEvaluator {
  def loadUserActualItemsDF(k: Int)(implicit spark: SparkSession): DataFrame = {
    import spark.implicits._

    val path = s"${settings.dataDir}/${settings.today}/userActualItemsDF-$k.parquet"
    loadOrCreateDataFrame(path, () => {
      val rawStarringDS = loadRawStarringDS()
      rawStarringDS.transform(intoUserActualItems($"user_id", $"repo_id", $"starred_at".desc, k))
    })
  }

  def intoUserActualItems(userCol: Column, itemCol: Column, orderByCol: Column, k: Int)(df: Dataset[_]): DataFrame = {
    import df.sparkSession.implicits._

    df
      .withColumn("rank", rank().over(Window.partitionBy(userCol).orderBy(orderByCol)))
      .where($"rank" <= k)
      .groupBy(userCol)
      .agg(collect_list(itemCol).alias("items"))
  }

  def intoUserPredictedItems(userCol: Column, itemCol: Column, orderByCol: Column, k: Int)(df: Dataset[_]): DataFrame = {
    import df.sparkSession.implicits._

    df
      .withColumn("rank", rank().over(Window.partitionBy(userCol).orderBy(orderByCol)))
      .where($"rank" <= k)
      .groupBy(userCol)
      .agg(collect_list(itemCol).alias("items"))
  }

  def intoUserPredictedItems(userCol: Column, nestedItemCol: Column)(df: Dataset[_]): DataFrame = {
    df.select(userCol, nestedItemCol.alias("items"))
  }
}