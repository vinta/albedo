package ws.vinta.albedo.preprocessors

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.{FloatType, IntegerType, StructType}
import org.apache.spark.sql.{DataFrame, Dataset}
import ws.vinta.albedo.evaluators.RankingEvaluator._
import ws.vinta.albedo.utils.SchemaUtils.checkColumnType

class PredictionFormatter(override val uid: String)
  extends Transformer with DefaultParamsWritable {

  def this() = {
    this(Identifiable.randomUID("predictionFormatter"))
  }

  val userCol = new Param[String](this, "userCol", "User 所在的欄位名稱")

  def getUserCol: String = $(userCol)

  def setUserCol(value: String): this.type = set(userCol, value)
  setDefault(userCol -> "user")

  val itemCol = new Param[String](this, "itemCol", "Item 所在的欄位名稱")

  def getItemCol: String = $(itemCol)

  def setItemCol(value: String): this.type = set(itemCol, value)
  setDefault(itemCol -> "item")

  val predictionCol = new Param[String](this, "predictionCol", "Prediction 所在的欄位名稱")

  def getPredictionCol: String = $(predictionCol)

  def setPredictionCol(value: String): this.type = set(predictionCol, value)
  setDefault(predictionCol -> "prediction")

  override def transformSchema(schema: StructType): StructType = {
    checkColumnType(schema, $(userCol), IntegerType)
    checkColumnType(schema, $(itemCol), IntegerType)
    checkColumnType(schema, $(predictionCol), FloatType)

    schema
  }

  override def transform(alsPredictionDF: Dataset[_]): DataFrame = {
    transformSchema(alsPredictionDF.schema)

    alsPredictionDF.transform(intoUserPredictedItems(col($(userCol)), col($(itemCol)), col($(predictionCol)).desc))
  }

  override def copy(extra: ParamMap): PredictionFormatter = {
    defaultCopy(extra)
  }
}