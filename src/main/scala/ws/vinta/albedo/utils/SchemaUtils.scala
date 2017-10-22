package ws.vinta.albedo.utils

import org.apache.spark.sql.types._

object SchemaUtils {
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

  def checkColumnType(schema: StructType, colName: String, expectedDataType: DataType): Unit = {
    val actualDataType = schema(colName).dataType
    require(actualDataType == expectedDataType, s"Column $colName must be of type $expectedDataType but was actually $actualDataType.")
  }
}