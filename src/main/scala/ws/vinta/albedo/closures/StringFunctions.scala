package ws.vinta.albedo.closures

import scala.util.matching.Regex

object StringFunctions extends Serializable {
  val wordPatternEngOnly = """\w\.\-_"""
  val wordPatternIncludeCJK = """\w\.\-_\p{InHiragana}\p{InKatakana}\p{InBopomofo}\p{InCJKCompatibilityIdeographs}\p{InCJKUnifiedIdeographs}"""

  val reExtractWords: Regex = s"([$wordPatternEngOnly]+)".r
  val reExtractWordsIncludeCJK: Regex = s"([$wordPatternIncludeCJK]+)".r
  val reExtractEmailDomain: Regex = s"@([$wordPatternEngOnly]+)".r

  val extractWords: (String) => List[String] = (text: String) => {
    reExtractWords.findAllIn(text).toList
  }

  val extractWordsIncludeCJK: (String) => List[String] = (text: String) => {
    reExtractWordsIncludeCJK.findAllIn(text).toList
  }

  val extractEmailDomain: (String) => String = (email: String) => {
    try {
      reExtractEmailDomain.findAllIn(email).matchData.toList(0).group(1)
    } catch {
      case _: IndexOutOfBoundsException => {
        email
      }
    }
  }
}