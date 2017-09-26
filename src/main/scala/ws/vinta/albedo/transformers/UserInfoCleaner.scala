package ws.vinta.albedo.transformers

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{AnalysisException, DataFrame, Dataset}
import ws.vinta.albedo.settings
import ws.vinta.albedo.closures.StringFunctions._
import ws.vinta.albedo.closures.UDFs._

class UserInfoCleaner(override val uid: String)
  extends Transformer with DefaultParamsWritable {

  def this() {
    this(Identifiable.randomUID("userInfoCleaner"))
  }

  override def transformSchema(schema: StructType): StructType = {
    schema
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    val spark = dataset.sparkSession
    import spark.implicits._

    //val cleanCompanyUDF = udf((company: String) => {
    //  val temp1 = company
    //    .toLowerCase()
    //    .replaceAll("""\b(.com|.net|.org|.io)\b""", "")
    //    .replaceAll("""\W+""", " ")
    //    .replaceAll("""\s+""", " ")
    //    .replaceAll("""\b(http|https|www|inc|ltd|co ltd)\b""", "")
    //    .trim()
    //  val temp2 = extractWordsIncludeCJK(temp1).mkString(" ")
    //  if (temp2.isEmpty)
    //    "__empty"
    //  else
    //    temp2
    //})

    val cleanEmailUDF = udf((email: String) => {
      val temp1 = email.toLowerCase().trim()
      val temp2 = extractEmailDomain(temp1)
      if (temp2.isEmpty)
        "__empty"
      else
        temp2
    })

    val cleanLocationUDF = udf((location: String) => {
      val temp1 = try {
        val pattern = s"([$wordPatternIncludeCJK]+),\\s*([$wordPatternIncludeCJK]+)".r
        val pattern(city, _) = location
        city
      } catch {
        case _: MatchError => {
          location
        }
      }
      val temp2 = temp1
        .toLowerCase()
        .replaceAll("""[~!@#$^%&*\\(\\)_+={}\\[\\]|;:\"'<,>.?`/\\\\-]+""", " ")
        .replaceAll("""\s+""", " ")
        .replaceAll("""\b(city)\b""", "")
        .trim()
      val temp3 = extractWordsIncludeCJK(temp2).mkString(" ")
      if (temp3.isEmpty)
        "__empty"
      else
        temp3
    })

    val cleanDF = dataset
      .na.fill("", Array("bio", "blog", "company", "email", "location", "name"))
      .withColumn("clean_company", cleanCompanyUDF($"company"))
      .withColumn("clean_email", cleanEmailUDF($"email"))
      .withColumn("clean_location", cleanLocationUDF($"location"))

    val savePath = s"${settings.dataDir}/${settings.today}/cleanUserInfoDF.parquet"
    val outDF: DataFrame = try {
      spark.read.parquet(savePath)
    } catch {
      case e: AnalysisException => {
        if (e.getMessage().contains("Path does not exist")) {
          cleanDF.write.mode("overwrite").parquet(savePath)
          cleanDF
        } else {
          throw e
        }
      }
    }
    outDF
  }

  override def copy(extra: ParamMap): this.type = {
    defaultCopy(extra)
  }
}