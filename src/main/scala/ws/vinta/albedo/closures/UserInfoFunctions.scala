package ws.vinta.albedo.closures

import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.udf
import ws.vinta.albedo.closures.StringFunctions._

object UserInfoFunctions extends Serializable {
  val cleanCompanyUDF: UserDefinedFunction = udf((company: String) => {
    val temp1 = company
      .toLowerCase()
      .replaceAll("""\b(.com|.net|.org|.io)\b""", "")
      .replaceAll("""\W+""", " ")
      .replaceAll("""\s+""", " ")
      .replaceAll("""\b(http|https|www|inc|ltd|co ltd)\b""", "")
      .trim()
    val temp2 = extractWordsIncludeCJK(temp1).mkString(" ")
    if (temp2.isEmpty)
      "__empty"
    else
      temp2
  })

  val cleanEmailUDF: UserDefinedFunction = udf((email: String) => {
    val temp1 = email.toLowerCase().trim()
    val temp2 = extractEmailDomain(temp1)
    if (temp2.isEmpty)
      "__empty"
    else
      temp2
  })

  val cleanLocationUDF: UserDefinedFunction = udf((location: String) => {
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
}