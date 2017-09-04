package ws.vinta.albedo.preprocessors

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}
import ws.vinta.albedo.utils.StringUtils._

class UserInfoTransformer(override val uid: String)
  extends Transformer with DefaultParamsWritable {

  def this() {
    this(Identifiable.randomUID("userInfoTransformer"))
  }

  override def transformSchema(schema: StructType): StructType = {
    schema
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    import dataset.sparkSession.implicits._

    val cleanCompanyUDF = udf((company: String) => {
      val temp = company
        .toLowerCase()
        .replaceAll("""\b(.com|.net|.org|.io)\b""", "")
        .replaceAll("""\W+""", " ")
        .replaceAll("""\s+""", " ")
        .replaceAll("""\b(http|https|www|inc|ltd|co ltd)\b""", "")
        .trim()
      extractWordsIncludeCJK(temp).mkString(" ")
    })

    val cleanEmailUDF = udf((email: String) => {
      extractEmailDomain(email.toLowerCase().trim())
    })

    val cleanLocationUDF = udf((location: String) => {
      val temp = try {
        val pattern = """([\w ]+),([\w, ]+)""".r
        val pattern(city, country) = location
        city
      } catch {
        case _: MatchError => {
          location
        }
      }
      val temp2 = temp
        .toLowerCase()
        .replaceAll("""\W+""", " ")
        .replaceAll("""\s+""", " ")
        .replaceAll("""\b(city)\b""", "")
        .trim()
      extractWordsIncludeCJK(temp2).mkString(" ")
    })

    dataset
      .withColumn("company", cleanCompanyUDF($"company"))
      .withColumn("email", cleanEmailUDF($"email"))
      .withColumn("location", cleanLocationUDF($"location"))
  }

  override def copy(extra: ParamMap): this.type = {
    defaultCopy(extra)
  }
}