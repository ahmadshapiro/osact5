from locale import normalize
import unicodedata, re, demoji
import pyarabic.araby as araby
import pyarabic.trans
from camel_tools.utils.charsets import AR_CHARSET
from camel_tools.utils.normalize import normalize_unicode, normalize_alef_ar
from sacremoses import MosesTokenizer, MosesDetokenizer
from sacremoses import MosesPunctNormalizer
import string

def _remove_control_characters(s: str):
  """
  
  Remove control characters EG : "\r , \n etc.."
  
  """
  if len(s) == 0:
    return s
  
  return "".join(ch for ch in s if unicodedata.category(ch)[0]!="C")

def _remove_ar_diacritics(line: str):
  if len(line) == 0:
    return line
  
  #remove tashkeel except shadda
  TATWEEL = u'\u0640'
  line = araby.reduce_tashkeel(line)
  line = araby.strip_shadda(line)
  letters, marks = araby.separate(line)
  new_marks = [c if araby.is_shadda(c) else TATWEEL for c in marks]
  new_marks = ''.join(new_marks)
  line = araby.joint(letters, new_marks)
  line = araby.strip_tatweel(line)
  return line

def _remove_url(line: str):
  """
  
  Remove urls
  
  """

  if len(line) == 0:
    return line
  # remove hyperlinks
  line = re.sub(r'https?:\/\/.*[\r\n]*', '', line)
  pattern = re.compile(r'http[a-zA-Z0-9.?/&=:]*')
  return pattern.sub('', line.strip())
  
def _keep_wanted_words_only(line: str):
  """
  
  remove lines with majority non arabic words
  
  """
  if len(line) == 0:
    return line
  
  # Concatinate all Arabic characters into a string
  ar_str = u''.join(AR_CHARSET)
  
  # Compile a regular expression using above string
  arabic_re = re.compile(r'^[' + re.escape(ar_str) + r']+$')
  words = line.split()
  word_count = len(words)
  
  #find arabic/english letters only
  line = ' '.join(re.findall(r'[\u0021-\u007A\u0600-\u06FF]+',line))
  
  ar_count = 0
  for word in words:
    if arabic_re.match(word) is not None:
      ar_count += 1
  if word_count > 0 and (ar_count / word_count + 0.01) > 0.6:
    return line
  else:
    return ''

def _unbias_emojis(line: str):
  emoji_pattern=re.compile(u"[\U0001F3FB-\U0001F3FF]", re.VERBOSE)
  return emoji_pattern.sub('', line).strip()

def _clean_twitter(line: str):
  
  """
  
  Tweets related cleaning
  
  """
  # remove hashtags (only removing the hash # sign from the word)
  line = re.sub(r'#', '', line)
  # remove mentions
  line = re.sub(r':', '', line)
  line = re.sub(r'RT @USER', '', line)
  line = re.sub(r'@USER', '', line)
  line = re.sub(r'URL', '', line)
  line = re.sub(r'<LF>', '', line)
  return line

def _remove_unwanted_symbols(sent: str):

    """

    remove unwanted symbols

    """
    #sent = re.sub("[\(\[\<].*?[\)\]\>]", "", sent)
    symbols = '[\^|/#&@\[\]\{\}~<=>`—*_]+'
    sent = re.sub(symbols, " ", sent)
    return sent.strip()

def _remove_repeated_chars(line: str):
  # replace letters repeated more than 2 times to only two occurunces
  line = re.sub(r'(.)\1{2,}', r'\1', line)
  # replace punctiuation repeated more than 2 times to only two occurunces
  line = re.sub(r'''([\u2000-\u206F\u2E00-\u2E7F\\'!"#$%&()*+,\-.\/:;<=>?@\[\]^_`{|}~])\1{1,}''', r'\1', line)
  return line

def _normalize_ar(line: str):
  line = normalize_alef_ar(line)
  line = normalize_unicode(line)
  line = pyarabic.trans.normalize_digits(line, source='all', out='west')
  line = line.replace( '؟', '?').replace('،', ',').replace('؛', ';')
  return line

def _sacremoses(line: str, mpn, mt, md):
  
  line = mt.tokenize(line, return_str=True, escape=False)
  line = mpn.normalize(line)
  line = md.detokenize(line.split())
  
  return line


def apply_cleaning(lines):
  mpn = MosesPunctNormalizer()
  mt, md = MosesTokenizer(lang='ar'), MosesDetokenizer(lang='ar')
  lines = list(lines)
  for i in range(len(lines)):
    lines[i] = _remove_control_characters(lines[i])
    lines[i] = _clean_twitter(lines[i])
    lines[i] = _remove_url(lines[i])
    lines[i] = _unbias_emojis(lines[i])
    lines[i] = _remove_ar_diacritics(lines[i])
    lines[i] = _remove_repeated_chars(lines[i])
    lines[i] = _remove_unwanted_symbols(lines[i])
    lines[i] = _normalize_ar(lines[i])
    lines[i] = _sacremoses(lines[i], mpn, mt, md)


  return lines
