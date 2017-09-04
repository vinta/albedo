package ws.vinta.albedo.utils

object StringUtils {
  def extractWords(text: String): List[String] = {
    """([\w._\-]+)""".r.findAllIn(text).toList
  }

  def extractWordsIncludeCJK(text: String): List[String] = {
    """([\w._\-\p{InHiragana}\p{InKatakana}\p{InBopomofo}\p{InCJKCompatibilityIdeographs}\p{InCJKUnifiedIdeographs}]+)""".r.findAllIn(text).toList
  }

  def extractEmailDomain(email: String): String = {
    try {
      """@([\w._\-]+)""".r.findAllIn(email).matchData.toList(0).group(1)
    } catch {
      case _: IndexOutOfBoundsException => {
        email
      }
    }
  }
}