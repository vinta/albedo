package ws.vinta.albedo.transformers

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import ws.vinta.albedo.utils.SchemaUtils.checkColumnType

import scala.collection.mutable

class NegativeGenerator(override val uid: String, val bcPopularItems: Broadcast[mutable.LinkedHashSet[Int]])
  extends Transformer with DefaultParamsWritable {

  def this(bcPopularItems: Broadcast[mutable.LinkedHashSet[Int]]) = {
    this(Identifiable.randomUID("negativeGenerator"), bcPopularItems)
  }

  val userCol = new Param[String](this, "userCol", "User 所在的欄位名稱")

  def getUserCol: String = $(userCol)

  def setUserCol(value: String): this.type = set(userCol, value)
  setDefault(userCol -> "user")

  val itemCol = new Param[String](this, "itemCol", "Item 所在的欄位名稱")

  def getItemCol: String = $(itemCol)

  def setItemCol(value: String): this.type = set(itemCol, value)
  setDefault(itemCol -> "item")

  val labelCol = new Param[String](this, "labelCol", "Label 所在的欄位名稱")

  def getLabelCol: String = $(labelCol)

  def setLabelCol(value: String): this.type = set(labelCol, value)
  setDefault(labelCol -> "label")

  val negativeValue = new Param[Double](this, "negativeValue", "負樣本的值")

  def getNegativeValue: Double = $(negativeValue)

  def setNegativeValue(value: Double): this.type = set(negativeValue, value)
  setDefault(negativeValue -> 0.0)

  val negativePositiveRatio = new Param[Double](this, "negativePositiveRatio", "負樣本與正樣本的比例")

  def getNegativePositiveRatio: Double = $(negativePositiveRatio)

  def setNegativePositiveRatio(value: Double): this.type = set(negativePositiveRatio, value)
  setDefault(negativePositiveRatio -> 1.0)

  override def transformSchema(schema: StructType): StructType = {
    checkColumnType(schema, $(userCol), IntegerType)
    checkColumnType(schema, $(itemCol), IntegerType)
    checkColumnType(schema, $(labelCol), DoubleType)

    schema
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema)

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
      negativeItems.map({(user, _, $(negativeValue))})
    }

    import dataset.sparkSession.implicits._

    // TODO: 目前是假設傳進來的 dataset 都是 positive samples，之後可能得處理含有 negative samples 的情況
    val negativeDF = dataset
      .select($(userCol), $(itemCol))
      .rdd
      .map({
        case Row(user: Int, item: Int) => (user, item)
      })
      .aggregateByKey(emptyItemSet)(addToItemSet, mergeItemSets)
      .map(getUserNegativeItems)
      .flatMap(expandNegativeItems)
      .toDF($(userCol), $(itemCol), $(labelCol))

    dataset.select($(userCol), $(itemCol), $(labelCol)).union(negativeDF)
  }

  override def copy(extra: ParamMap): this.type = {
    defaultCopy(extra)
  }
}
