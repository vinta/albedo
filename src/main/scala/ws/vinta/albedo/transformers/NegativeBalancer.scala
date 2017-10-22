package ws.vinta.albedo.transformers

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{DoubleParam, Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.functions._

import scala.collection.mutable

class NegativeBalancer(override val uid: String, val bcPopularItems: Broadcast[mutable.LinkedHashSet[Int]])
  extends Transformer with DefaultParamsWritable {

  def this(bcPopularItems: Broadcast[mutable.LinkedHashSet[Int]]) = {
    this(Identifiable.randomUID("negativeBalancer"), bcPopularItems)
  }

  val userCol = new Param[String](this, "userCol", "User column name")

  def getUserCol: String = $(userCol)

  def setUserCol(value: String): this.type = set(userCol, value)
  setDefault(userCol -> "user")

  val itemCol = new Param[String](this, "itemCol", "Item column name")

  def getItemCol: String = $(itemCol)

  def setItemCol(value: String): this.type = set(itemCol, value)
  setDefault(itemCol -> "item")

  val timeCol = new Param[String](this, "timeCol", "Time column name")

  def getTimeCol: String = $(timeCol)

  def setTimeCol(value: String): this.type = set(timeCol, value)
  setDefault(timeCol -> "time")

  val labelCol = new Param[String](this, "labelCol", "Label column name")

  def getLabelCol: String = $(labelCol)

  def setLabelCol(value: String): this.type = set(labelCol, value)
  setDefault(labelCol -> "label")

  val negativeValue = new DoubleParam(this, "negativeValue", "The value of negative samples")

  def getNegativeValue: Double = $(negativeValue)

  def setNegativeValue(value: Double): this.type = set(negativeValue, value)
  setDefault(negativeValue -> 0.0)

  val negativePositiveRatio = new DoubleParam(this, "negativePositiveRatio", "The ratio of negative and positive samples")

  def getNegativePositiveRatio: Double = $(negativePositiveRatio)

  def setNegativePositiveRatio(value: Double): this.type = set(negativePositiveRatio, value)
  setDefault(negativePositiveRatio -> 1.0)

  override def transformSchema(schema: StructType): StructType = {
    Map($(userCol) -> IntegerType, $(itemCol) -> IntegerType, $(timeCol) -> TimestampType, $(labelCol) -> DoubleType)
      .foreach {
        case(columnName: String, expectedDataType: DataType) => {
          val actualDataType = schema(columnName).dataType
          require(actualDataType.equals(expectedDataType), s"Column $columnName must be of type $expectedDataType but was actually $actualDataType.")
        }
      }

    schema
  }

  override def transform(rawStarringDS: Dataset[_]): DataFrame = {
    transformSchema(rawStarringDS.schema)

    import rawStarringDS.sparkSession.implicits._

    val popularItems: mutable.LinkedHashSet[Int] = this.bcPopularItems.value

    val emptyItemSet = new mutable.HashSet[Int]
    val addToItemSet = (itemSet: mutable.HashSet[Int], item: Int) => itemSet += item
    val mergeItemSets = (set1: mutable.HashSet[Int], set2: mutable.HashSet[Int]) => set1 ++= set2

    val getUserNegativeItems = (userItemsPair: (Int, mutable.HashSet[Int])) => {
      val (user, positiveItems) = userItemsPair
      val negativeItems = popularItems.diff(positiveItems)
      val requiredNegativeItemsCount = (positiveItems.size * this.getNegativePositiveRatio).toInt
      (user, negativeItems.slice(0, requiredNegativeItemsCount))
    }
    val expandNegativeItems = (userItemsPair: (Int, mutable.LinkedHashSet[Int])) => {
      val (user, negativeItems) = userItemsPair
      negativeItems.map({(user, _)})
    }

    // TODO: 目前是假設傳進來的 dataset 都是 positive samples，之後可能得處理含有 negative samples 的情況
    val negativeDF = rawStarringDS
      .select($(userCol), $(itemCol))
      .rdd
      .map({
        case Row(user: Int, item: Int) => (user, item)
      })
      .aggregateByKey(emptyItemSet)(addToItemSet, mergeItemSets)
      .map(getUserNegativeItems)
      .flatMap(expandNegativeItems)
      .toDF($(userCol), $(itemCol))
      .select(col($(userCol)), col($(itemCol)), lit("1999-07-01T00:00:00.000+0000").cast("timestamp").alias($(timeCol)), lit($(negativeValue)).alias($(labelCol)))

    rawStarringDS
      .select(col("*"))
      .union(negativeDF)
  }

  override def copy(extra: ParamMap): this.type = {
    defaultCopy(extra)
  }
}

object NegativeBalancer extends DefaultParamsReadable[NegativeBalancer]