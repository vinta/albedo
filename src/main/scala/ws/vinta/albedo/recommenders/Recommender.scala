package ws.vinta.albedo.recommenders

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{IntParam, Param, ParamMap}
import org.apache.spark.ml.util.DefaultParamsWritable
import org.apache.spark.sql.types.{DataType, IntegerType, StructType}
import org.apache.spark.sql.{DataFrame, Dataset}

abstract class Recommender extends Transformer with DefaultParamsWritable {

  val userCol = new Param[String](this, "userCol", "User column name")

  def getUserCol: String = $(userCol)

  def setUserCol(value: String): this.type = set(userCol, value)
  setDefault(userCol -> "user")

  val itemCol = new Param[String](this, "itemCol", "Item column name")

  def getItemCol: String = $(itemCol)

  def setItemCol(value: String): this.type = set(itemCol, value)
  setDefault(itemCol -> "item")

  val scoreCol = new Param[String](this, "scoreCol", "Score column name")

  def getScoreCol: String = $(scoreCol)

  def setScoreCol(value: String): this.type = set(scoreCol, value)
  setDefault(scoreCol -> "score")

  val sourceCol = new Param[String](this, "sourceCol", "Source column name")

  def getSourceCol: String = $(sourceCol)

  def setSourceCol(value: String): this.type = set(sourceCol, value)
  setDefault(sourceCol -> "source")

  val topK = new IntParam(this, "topK", "Recommend top-k items for every user")

  def getTopK: Int = $(topK)

  def setTopK(value: Int): this.type = set(topK, value)
  setDefault(topK -> 15)

  override def transformSchema(schema: StructType): StructType = {
    Map($(userCol) -> IntegerType)
      .foreach{
        case(columnName: String, expectedDataType: DataType) => {
          val actualDataType = schema(columnName).dataType
          require(actualDataType.equals(IntegerType), s"Column $columnName must be of type $expectedDataType but was actually $actualDataType.")
        }
      }

    schema
  }

  override def transform(userDF: Dataset[_]): DataFrame = {
    recommendForUsers(userDF)
  }

  override def copy(extra: ParamMap): this.type = {
    defaultCopy(extra)
  }

  def source: String

  def recommendForUsers(userDF: Dataset[_]): DataFrame
}