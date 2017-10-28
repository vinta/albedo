package ws.vinta.albedo.closures

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._
import ws.vinta.albedo.closures.StringFunctions._

import scala.util.control.Breaks.{break, breakable}

object UDFs extends Serializable {
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

  def toArrayUDF: UserDefinedFunction = udf[Array[Double], Vector]((vector: Vector) => {
    vector.toArray
  })

  def numNonzerosOfVectorUDF: UserDefinedFunction = udf[Int, Vector]((vector: Vector) => {
    vector.numNonzeros
  })

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

  def repoLanguageIndexInUserRecentRepoLanguagesUDF = udf((repo_language: String, user_recent_repo_languages: Seq[String]) => {
    val index = user_recent_repo_languages.indexOf(repo_language.toLowerCase())
    if (index < 0) user_recent_repo_languages.size + 50 else index
  })

  def repoLanguageCountInUserRecentRepoLanguagesUDF = udf((repo_language: String, user_recent_repo_languages: Seq[String]) => {
    user_recent_repo_languages.count(_ == repo_language.toLowerCase())
  })
}