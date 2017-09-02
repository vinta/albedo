package ws.vinta.albedo.preprocessors

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.functions.{collect_list, struct}
import org.apache.spark.sql.types.{FloatType, IntegerType, StructType}
import org.apache.spark.sql.{DataFrame, Dataset}
import ws.vinta.albedo.utils.SchemaUtils.checkColumnType

class RecommendationFormatter(override val uid: String)
  extends Transformer with DefaultParamsWritable {

  def this() = {
    this(Identifiable.randomUID("negativeGenerator"))
  }

  val userCol = new Param[String](this, "userCol", "User id 所在的欄位名稱")

  def getUserCol: String = $(userCol)

  def setUserCol(value: String): this.type = set(userCol, value)
  setDefault(userCol -> "user")

  val itemCol = new Param[String](this, "itemCol", "Item id 所在的欄位名稱")

  def getItemCol: String = $(itemCol)

  def setItemCol(value: String): this.type = set(itemCol, value)
  setDefault(itemCol -> "item")

  val predictionCol = new Param[String](this, "predictionCol", "Prediction 所在的欄位名稱")

  def getPredictionCol: String = $(predictionCol)

  def setPredictionCol(value: String): this.type = set(predictionCol, value)
  setDefault(predictionCol -> "prediction")

  val outputCol = new Param[String](this, "outputCol", "ALS prediction 所在的欄位")

  def getOutputCol: String = $(outputCol)

  def setOutputCol(value: String): this.type = set(outputCol, value)
  setDefault(outputCol -> "recommendations")

  override def transformSchema(schema: StructType): StructType = {
    checkColumnType(schema, $(userCol), IntegerType)
    checkColumnType(schema, $(itemCol), IntegerType)
    checkColumnType(schema, $(predictionCol), FloatType)

    schema
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema)

    val userRecommendationsDF = dataset
      .groupBy($(userCol))
      .agg(collect_list(struct($(itemCol), $(predictionCol))).alias($(outputCol)))

    userRecommendationsDF
  }

  override def copy(extra: ParamMap): RecommendationFormatter = {
    defaultCopy(extra)
  }
}