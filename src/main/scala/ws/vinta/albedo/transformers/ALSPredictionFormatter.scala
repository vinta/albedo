package ws.vinta.albedo.transformers

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{IntParam, Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset}
import ws.vinta.albedo.evaluators.RankingEvaluator._

class ALSPredictionFormatter(override val uid: String)
  extends Transformer with DefaultParamsWritable {

  def this() = {
    this(Identifiable.randomUID("alsPredictionFormatter"))
  }

  val userCol = new Param[String](this, "userCol", "User column name")

  def getUserCol: String = $(userCol)

  def setUserCol(value: String): this.type = set(userCol, value)
  setDefault(userCol -> "user")

  val itemCol = new Param[String](this, "itemCol", "Item column name")

  def getItemCol: String = $(itemCol)

  def setItemCol(value: String): this.type = set(itemCol, value)
  setDefault(itemCol -> "item")

  val predictionCol = new Param[String](this, "predictionCol", "Prediction column name")

  def getPredictionCol: String = $(predictionCol)

  def setPredictionCol(value: String): this.type = set(predictionCol, value)
  setDefault(predictionCol -> "prediction")

  val topK = new IntParam(this, "topK", "Recommend top-k items for every user")

  def getTopK: Int = $(topK)

  def setTopK(value: Int): this.type = set(topK, value)
  setDefault(topK -> 15)

  override def transformSchema(schema: StructType): StructType = {
    Map($(userCol) -> IntegerType, $(itemCol) -> IntegerType, $(predictionCol) -> FloatType)
      .foreach{
        case(columnName: String, expectedDataType: DataType) => {
          val actualDataType = schema(columnName).dataType
          require(actualDataType.equals(expectedDataType), s"Column $columnName must be of type $expectedDataType but was actually $actualDataType.")
        }
      }

    schema
  }

  override def transform(alsPredictionDF: Dataset[_]): DataFrame = {
    transformSchema(alsPredictionDF.schema)

    alsPredictionDF.transform(intoUserPredictedItems(col($(userCol)), col($(itemCol)), col($(predictionCol)).desc, $(topK)))
  }

  override def copy(extra: ParamMap): this.type = {
    defaultCopy(extra)
  }
}