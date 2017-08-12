package ws.vinta.albedo.utils

import org.apache.spark.sql.types.{DataType, StructType}

object SchemaUtils {
  def checkColumnType(schema: StructType, colName: String, dataType: DataType): Unit = {
    val actualDataType = schema(colName).dataType
    require(actualDataType.equals(dataType), s"Column $colName must be of type $dataType but was actually $actualDataType.")
  }
}