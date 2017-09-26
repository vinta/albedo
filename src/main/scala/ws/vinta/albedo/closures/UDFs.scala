package ws.vinta.albedo.closures

import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.udf
import ws.vinta.albedo.closures.StringFunctions._

import scala.util.control.Breaks.{break, breakable}

object UDFs extends Serializable {
  def cleanCompanyUDF: UserDefinedFunction = udf[String, String]((company: String) => {
    val temp1 = company
      .toLowerCase()
      .replaceAll("""\b(.com|.net|.org|.io|.co.uk|.co|.eu|.fr|.de|.ru)\b""", "")
      .replaceAll("""\b(formerly|previously|ex\-)\b""", "")
      .replaceAll("""\W+""", " ")
      .replaceAll("""\s+""", " ")
      .replaceAll("""\b(http|https|www|co ltd|pvt ltd|ltd|inc|llc)\b""", "")
      .trim()
    val temp2 = extractWordsIncludeCJK(temp1).mkString(" ")
    if (temp2.isEmpty)
      "__empty"
    else
      temp2
  })

  def cleanEmailUDF: UserDefinedFunction = udf[String, String]((email: String) => {
    val temp1 = email.toLowerCase().trim()
    val temp2 = extractEmailDomain(temp1)
    if (temp2.isEmpty)
      "__empty"
    else
      temp2
  })

  def cleanLocationUDF: UserDefinedFunction = udf[String, String]((location: String) => {
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

  def containsAnyOfUDF(substrings: Array[String], shouldLower: Boolean = false): UserDefinedFunction = udf[Double, String]((text: String) => {
    var result = 0.0
    breakable {
      for (substring <- substrings) {
        if (text.contains(substring)) {
          result = 1.0
          break
        }
      }
    }
    result
  })
}