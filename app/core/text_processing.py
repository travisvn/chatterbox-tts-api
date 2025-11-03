"""
Text processing utilities for TTS
- Robust sentence splitting (abbrev/decimals/quotes/ellipses), bullet handling, non-verbal cues
- TTS-friendly normalization baked in (°, ℃/℉/K, primes, %, currencies, fractions, ellipses, µ/Ω, per-slash, etc.)
- Enhanced with ordinal/roman numeral/time normalization, performance optimizations, and robust error handling.
- Uses `num2words` library if available for superior number-to-word conversion.
- Correctly verbalizes large, comma-formatted numbers and handles parenthetical content gracefully.
- Intelligently handles monetary values with magnitude words (e.g., "$4.65 billion").
- Expands common timezone abbreviations (e.g., "ET" -> "Eastern Time").
- Normalizes dates ("November 4" -> "November fourth") and number ranges ("2018-2019" -> "2018 to 2019").
- Handles full dates ("November 3, 2025") as a single unit for natural prosody.
- Splits sentences at headline-style colons for natural pauses.
- Includes phonetic hints, scientific notation, chemical formulas, and other advanced edge cases.
- Verbalizes appended symbols like in "OPEC+".
- Converts parentheticals to comma-separated clauses for improved prosody.
"""

from __future__ import annotations
import gc
import logging
import re
from typing import List, Optional, Tuple, Set, Dict
from functools import lru_cache

import torch
from app.config import Config
from app.models.long_text import LongTextChunk

logger = logging.getLogger(__name__)

# =============================================================================
#              OPTIONAL DEPENDENCY: num2words
# =============================================================================
try:
    from num2words import num2words
    _NUM2WORDS_AVAILABLE = True
    logger.info("num2words library found. Using advanced number-to-word conversion.")
except ImportError:
    _NUM2WORDS_AVAILABLE = False
    logger.info("num2words library not found. Falling back to basic number-to-words conversion.")

# =============================================================================
#              CUSTOMIZATION: PHONETIC HINTS
# =============================================================================
# Add custom pronunciations for acronyms, jargon, or brand names.
# Keys are case-insensitive and treated as whole words.
PHONETIC_HINTS = {
    "SQL": "sequel",
    "GIF": "jiff",
    "NGINX": "engine-x",
    "LLM": "L L M",
    "API": "A P I",
}
# Pre-process hints for regex
_PHONETIC_HINTS_UPPER = {k.upper(): v for k, v in PHONETIC_HINTS.items()}
_PHONETIC_RE = re.compile(r"\b(" + "|".join(_PHONETIC_HINTS_UPPER.keys()) + r")\b", re.IGNORECASE)


# =============================================================================
#              PERFORMANCE: PRE-COMPILED REGEX PATTERNS
# =============================================================================

_URL_RE = re.compile(r"""(?P<url>(?:(?:https?|ftp)://)[^\s<>'"()]+)""", re.IGNORECASE | re.VERBOSE)
_EMAIL_RE = re.compile(r"""(?P<email>\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b)""", re.VERBOSE)
_ELLIPSIS_RE = re.compile(r"\u2026")  # …
_MANUAL_ELLIPSIS_RE = re.compile(r"(?<!\.)\.\.\.(?!\.)") # ...
_TEMP_C_RE = re.compile(r"(?P<val>-?[\d,]+(?:\.\d+)?)\s*(?:°\s*C|℃)\b", re.IGNORECASE)
_TEMP_F_RE = re.compile(r"(?P<val>-?[\d,]+(?:\.\d+)?)\s*(?:°\s*F|℉)\b", re.IGNORECASE)
_TEMP_K_RE = re.compile(r"(?P<val>-?[\d,]+(?:\.\d+)?)\s*K\b")
_DEGREE_RE = re.compile(r"(?P<deg>[\d,]+(?:\.\d+)?)\s*°(?!\s*[CFcf])")
_DMS_LONG_RE = re.compile(r"(?P<d>\d{1,3})\s*°\s*(?P<m>\d{1,2})\s*[′']\s*(?P<s>\d{1,2})\s*[″\"]\s*(?P<h>[NSEW])?", re.IGNORECASE)
_DMS_SHORT_RE = re.compile(r"(?P<d>\d{1,3})\s*°\s*(?P<m>\d{1,2})\s*[′']\s*(?P<h>[NSEW])?", re.IGNORECASE)
_FEET_INCHES_RE = re.compile(r"(?P<ft>\d{1,2})\s*[′']\s*(?P<in>\d{1,2})\s*[″\"]\b")
_FEET_RE = re.compile(r"(?P<ft>\d{1,2})\s*[′']\b")
_INCHES_RE = re.compile(r"(?P<inch>\d{1,2})\s*[″\"]\b")
_PERCENT_RE = re.compile(r"(?P<val>-?[\d,]+(?:\.\d+)?)\s*%")
_PERMILLE_RE = re.compile(r"(?P<val>-?[\d,]+(?:\.\d+)?)\s*‰")
_BASIS_PTS_RE = re.compile(r"(?P<val>-?[\d,]+(?:\.\d+)?)\s*ⱀ")
_CURRENCY_MAGNITUDE_RE = re.compile(r"(?P<sym>[$€£¥₹₩₦₽₪])\s?(?P<amt>[\d,]+(?:\.\d+)?)\s*(?P<mag>million|billion|trillion)\b", re.IGNORECASE)
_CURRENCY_PRE_RE = re.compile(r"(?<!\w)(?P<sym>[$€£¥₹₩₦₽₪])\s?(?P<amt>[\d,]+(?:\.\d+)?)(?!\s*(?:million|billion|trillion)\b)", re.IGNORECASE)
_CURRENCY_POST_RE = re.compile(r"(?P<amt>[\d,]+(?:\.\d+)?)\s?(?P<sym>[€£¥₹₩₦₽₪])\b")
_AMPERSAND_RE = re.compile(r"(?<=\w)\s*&\s*(?=\w)")
_WORD_PLUS_RE = re.compile(r"\b([A-Z][a-zA-Z0-9]*)\+(?!\w)")
_MICRO_UNITS_RE = re.compile(r"(?P<num>[\d,]+(?:\.\d+)?)\s*[µμ]\s?(?P<u>[A-Za-z]+)\b")
_KILOOHM_RE = re.compile(r"(?P<num>[\d,]+(?:\.\d+)?)\s*kΩ\b", re.IGNORECASE)
_MEGAOHM_RE = re.compile(r"(?P<num>[\d,]+(?:\.\d+)?)\s*MΩ\b", re.IGNORECASE)
_OHM_RE = re.compile(r"(?P<num>[\d,]+(?:\.\d+)?)\s*Ω\b")
_PER_SLASH_RE = re.compile(r"\b(?P<a>[A-Za-z]{1,6})\s*/\s*(?P<b>[A-Za-z]{1,6})\b")
_HASHTAG_NUM_RE = re.compile(r"#(?P<num>[\d,]+)\b")
_HASHTAG_TAG_RE = re.compile(r"#(?P<tag>[A-Za-z_][A-Za-z0-9_]*)")
_MENTION_RE = re.compile(r"@(?P<user>[A-Za-z0-9_]{2,})\b")
_ORDINAL_RE = re.compile(r"\b(\d+)(st|nd|rd|th)\b")
_TIME_12H_RE = re.compile(r"\b(\d{1,2}):(\d{2})\s*([AP]\.?M\.?)\b", re.IGNORECASE)
_TIME_24H_RE = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")
_YEAR_RANGE_RE = re.compile(r"\b(\d{4})\s?[–-]\s?(\d{4})\b")
_NUMBER_RANGE_RE = re.compile(r"\b(\d+)\s?[–-]\s?(\d+)\b")
_ROMAN_NUMERAL_RE = re.compile(r"\b(X|IX|IV|V?I{0,3})\b") # Common cases up to 10
_FORMATTED_NUMBER_RE = re.compile(r'\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b')
_SIMPLE_NUMBER_RE = re.compile(r'\b\d+\b')
_PARENS_ACRONYM_RE = re.compile(r"\s+\(([A-Z]{2,6})\)")
_SCIENTIFIC_NOTATION_RE = re.compile(r"\b([\d\.]+)\s?[xXeE]\s?10\^([\d\.\-]+)\b", re.IGNORECASE)
_US_PHONE_RE = re.compile(r"\b\(?(\d{3})\)?[\s.-]?(\d{3})[\s.-]?(\d{4})\b")
_CHEM_FORMULA_RE = re.compile(r"\b([A-Z][a-z]?)(\d+)\b")
_WHITESPACE_RE = re.compile(r"\s{2,}")
_COMMA_CLEANUP_RE = re.compile(r"(\s*,\s*){2,}")

_MONTH_NAMES = r"January|February|March|April|May|June|July|August|September|October|November|December"
_DATE_FULL_RE = re.compile(fr"\b({_MONTH_NAMES})\s+(\d{{1,2}})(?:st|nd|rd|th)?,\s+(\d{{4}})\b", re.IGNORECASE)
_DATE_MONTH_DAY_RE = re.compile(fr"\b({_MONTH_NAMES})\s+(\d{{1,2}})(?!st|nd|rd|th)\b", re.IGNORECASE)

_TIMEZONES = {
    "ET": "Eastern Time", "EST": "Eastern Standard Time", "EDT": "Eastern Daylight Time",
    "CT": "Central Time", "CST": "Central Standard Time", "CDT": "Central Daylight Time",
    "MT": "Mountain Time", "MST": "Mountain Standard Time", "MDT": "Mountain Daylight Time",
    "PT": "Pacific Time", "PST": "Pacific Standard Time", "PDT": "Pacific Daylight Time",
    "UTC": "Coordinated Universal Time", "GMT": "Greenwich Mean Time",
}
_TIMEZONE_RE = re.compile(r"\b(" + "|".join(_TIMEZONES.keys()) + r")\b", re.IGNORECASE)

# =============================================================================
#                               NORMALIZATION
# =============================================================================

def _mask(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Mask URLs and emails to prevent them from being normalized.

    Returns:
        Tuple of (masked_text, mapping_dict) where mapping_dict
        maps placeholder tokens to original URLs/emails.
    """
    mapping: Dict[str, str] = {}
    idx = 0

    def repl_url(m):
        nonlocal idx
        key = f"__URL_{idx}__"
        mapping[key] = m.group("url")
        idx += 1
        return key

    def repl_email(m):
        nonlocal idx
        key = f"__EMAIL_{idx}__"
        mapping[key] = m.group("email")
        idx += 1
        return key

    text = _URL_RE.sub(repl_url, text)
    text = _EMAIL_RE.sub(repl_email, text)
    return text, mapping

def _unmask(text: str, mapping: Dict[str, str], *, read_urls: bool) -> str:
    """Replaces placeholder tokens with original or spoken URLs/emails."""
    for key, val in mapping.items():
        if read_urls:
            spoken = val.replace("://", " colon slash slash ").replace("/", " slash ")
            spoken = spoken.replace(".", " dot ").replace("-", " dash ")
            text = text.replace(key, spoken)
        else:
            text = text.replace(key, val)
    return text

def _pluralize(unit: str, value: str) -> str:
    """Pluralizes a unit based on the numeric value."""
    try:
        clean_value = value.replace(",", "")
        v = float(clean_value)
    except (ValueError, AttributeError) as e:
        logger.warning(f"Could not parse value for pluralization: {value}. Details: {e}")
        return unit
    return unit if abs(v) == 1 else unit + "s"

@lru_cache(maxsize=1024)
def _verbalize_number(num_str: str, to_year: bool = False) -> str:
    """Converts a number string to words, with special handling for years. Cached for performance."""
    if not num_str:
        return ""
    
    clean_num_str = num_str.replace(',', '')
    
    try:
        # Handle decimals
        if '.' in clean_num_str:
            integer_part, decimal_part = clean_num_str.split('.', 1)
            if _NUM2WORDS_AVAILABLE:
                return f"{num2words(int(integer_part))} point {' '.join(num2words(int(c)) for c in decimal_part)}"
            else:
                return f"{_verbalize_number(integer_part)} point {' '.join(_verbalize_number(c) for c in decimal_part)}"

        num = int(clean_num_str)
        
        if _NUM2WORDS_AVAILABLE:
            return num2words(num, to='year' if to_year else 'cardinal')
        
        # Fallback implementation
        if to_year and 1000 <= num <= 2999:
            if num % 100 == 0 and num % 1000 != 0: return f"{_verbalize_number(str(num//100))} hundred"
            return f"{_verbalize_number(str(num//100))} {_verbalize_number(str(num%100))}"

        if num < 0: return f"minus {_verbalize_number(str(abs(num)))}"
        if num < 1000:
            if num == 0: return "zero"
            ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
            teens = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
            tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
            if 1 <= num < 10: return ones[num]
            if 10 <= num < 20: return teens[num - 10]
            if 20 <= num < 100: return tens[num // 10] + (" " + ones[num % 10] if num % 10 else "")
            if 100 <= num < 1000: return ones[num // 100] + " hundred" + (" " + _verbalize_number(str(num % 100)) if num % 100 else "")
        
        if num < 1_000_000:
            thousands = num // 1000
            remainder = num % 1000
            result = f"{_verbalize_number(str(thousands))} thousand"
            if remainder: result += f" {_verbalize_number(str(remainder))}"
            return result
        
        if num < 1_000_000_000:
            millions = num // 1_000_000
            remainder = num % 1_000_000
            result = f"{_verbalize_number(str(millions))} million"
            if remainder: result += f" {_verbalize_number(str(remainder))}"
            return result

        logger.warning(f"Basic number to words fallback cannot handle: {num_str}")
        return clean_num_str

    except (ValueError, AttributeError):
        return num_str

def _to_ordinal_word(num: int) -> str:
    """Converts an integer to its ordinal word form."""
    if _NUM2WORDS_AVAILABLE:
        return num2words(num, to='ordinal')
    # Fallback logic
    if 11 <= num % 100 <= 13: return f"{_verbalize_number(str(num))}th"
    last_digit = num % 10
    if last_digit == 1: return f"{_verbalize_number(str(num))}st"
    if last_digit == 2: return f"{_verbalize_number(str(num))}nd"
    if last_digit == 3: return f"{_verbalize_number(str(num))}rd"
    return f"{_verbalize_number(str(num))}th"

def normalize_for_tts(
    text: str,
    *,
    speak_marks: bool = False,
    convert_ascii_fractions: bool = False,
    read_urls: bool = False,
) -> str:
    """Normalize symbols to spoken words for TTS. Safe defaults; URLs/emails protected."""
    if not text or text.isspace():
        return text

    s, maskmap = _mask(text)
    
    # Apply phonetic hints first to override any other rules
    s = _PHONETIC_RE.sub(lambda m: _PHONETIC_HINTS_UPPER[m.group(1).upper()], s)

    # Chemical formulas and subscripts
    _SUBSCRIPT_MAP = {'₀':'0', '₁':'1', '₂':'2', '₃':'3', '₄':'4', '₅':'5', '₆':'6', '₇':'7', '₈':'8', '₉':'9'}
    s = "".join(_SUBSCRIPT_MAP.get(c, c) for c in s)
    s = _CHEM_FORMULA_RE.sub(lambda m: f"{m.group(1)} {_verbalize_number(m.group(2))}", s)

    # Ellipses
    s = _ELLIPSIS_RE.sub(", ", s)
    s = _MANUAL_ELLIPSIS_RE.sub(", ", s)

    # Temperature (°C/℉), Kelvin
    s = _TEMP_C_RE.sub(lambda m: f"{_verbalize_number(m.group('val'))} degrees Celsius", s)
    s = _TEMP_F_RE.sub(lambda m: f"{_verbalize_number(m.group('val'))} degrees Fahrenheit", s)
    s = _TEMP_K_RE.sub(lambda m: f"{_verbalize_number(m.group('val'))} kelvins", s)

    # Bare degree (angles)
    s = _DEGREE_RE.sub(lambda m: f"{_verbalize_number(m.group('deg'))} degrees", s)

    # DMS angles & primes
    def repl_dms(m):
        deg, minutes, seconds, hemi = m.group("d"), m.group("m"), m.group("s"), m.group("h")
        parts = [f"{_verbalize_number(deg)} degrees"]
        if minutes: parts.append(f"{_verbalize_number(minutes)} minutes")
        if seconds: parts.append(f"{_verbalize_number(seconds)} seconds")
        if hemi:    parts.append(hemi.strip())
        return " ".join(parts)
    s = _DMS_LONG_RE.sub(repl_dms, s)
    s = _DMS_SHORT_RE.sub(
        lambda m: f"{_verbalize_number(m.group('d'))} degrees {_verbalize_number(m.group('m'))} minutes" + (f" {m.group('h')}" if m.group('h') else ""),
        s
    )

    # Heights 5′10″ / 5'10"
    s = _FEET_INCHES_RE.sub(
        lambda m: f"{_verbalize_number(m.group('ft'))} {_pluralize('foot', m.group('ft'))} "
                  f"{_verbalize_number(m.group('in'))} {_pluralize('inch', m.group('in'))}",
        s
    )
    s = _FEET_RE.sub(lambda m: f"{_verbalize_number(m.group('ft'))} {_pluralize('foot', m.group('ft'))}", s)
    s = _INCHES_RE.sub(lambda m: f"{_verbalize_number(m.group('inch'))} {_pluralize('inch', m.group('inch'))}", s)

    # Percent / permille / basis points
    s = _PERCENT_RE.sub(lambda m: f"{_verbalize_number(m.group('val'))} percent", s)
    s = _PERMILLE_RE.sub(lambda m: f"{_verbalize_number(m.group('val'))} per mille", s)
    s = _BASIS_PTS_RE.sub(lambda m: f"{_verbalize_number(m.group('val'))} basis points", s)

    # Currencies
    _CURRENCY_NAMES = {"$": "dollar", "€": "euro", "£": "pound", "¥": "yen", "₹": "rupee", "₩": "won", "₦": "naira", "₽": "ruble", "₪": "shekel"}
    def repl_currency_magnitude(m):
        amount = _verbalize_number(m.group('amt'))
        magnitude = m.group('mag').lower()
        currency_name = _CURRENCY_NAMES.get(m.group('sym'), 'currency')
        pluralized_currency = _pluralize(currency_name, "2") # The amount is > 1
        return f"{amount} {magnitude} {pluralized_currency}"
    s = _CURRENCY_MAGNITUDE_RE.sub(repl_currency_magnitude, s)
    s = _CURRENCY_PRE_RE.sub(lambda m: f"{_verbalize_number(m.group('amt'))} {_pluralize(_CURRENCY_NAMES.get(m.group('sym'), 'currency'), m.group('amt'))}", s)
    s = _CURRENCY_POST_RE.sub(lambda m: f"{_verbalize_number(m.group('amt'))} {_pluralize(_CURRENCY_NAMES.get(m.group('sym'), 'currency'), m.group('amt'))}", s)

    # Unicode fractions
    _FRACTIONS = {"½": "one half", "⅓": "one third", "⅔": "two thirds", "¼": "one quarter", "¾": "three quarters", "⅛": "one eighth", "⅜": "three eighths", "⅝": "five eighths", "⅞": "seven eighths", "⅕": "one fifth", "⅖": "two fifths", "⅗": "three fifths", "⅘": "four fifths", "⅙": "one sixth", "⅚": "five sixths", "⅐": "one seventh", "⅑": "one ninth", "⅒": "one tenth"}
    s = "".join(_FRACTIONS.get(ch, ch) for ch in s)
    if convert_ascii_fractions:
        s = re.sub(r"\b1/2\b", "one half", s); s = re.sub(r"\b1/4\b", "one quarter", s); s = re.sub(r"\b3/4\b", "three quarters", s)

    # Dates and ordinals (most specific rule first)
    s = _DATE_FULL_RE.sub(lambda m: f"{m.group(1)} {_to_ordinal_word(int(m.group(2)))} {_verbalize_number(m.group(3), to_year=True)}", s)
    s = _ORDINAL_RE.sub(lambda m: _to_ordinal_word(int(m.group(1))), s)
    s = _DATE_MONTH_DAY_RE.sub(lambda m: f"{m.group(1)} {_to_ordinal_word(int(m.group(2)))}", s)

    # Time, timezones, and ranges
    def repl_time_12h(m):
        hour, minute, ampm = int(m.group(1)), int(m.group(2)), m.group(3)
        ampm_spoken = " ".join(list(ampm.replace('.','').upper()))
        if hour == 12 and minute == 0 and 'PM' in ampm_spoken: return "noon"
        if hour == 12 and minute == 0 and 'AM' in ampm_spoken: return "midnight"
        h_word = _verbalize_number(str(hour)); m_word = "o'clock" if minute == 0 else f"oh {_verbalize_number(str(minute))}" if minute < 10 else _verbalize_number(str(minute))
        return f"{h_word} {m_word} {ampm_spoken}"
    def repl_time_24h(m):
        hour, minute = int(m.group(1)), int(m.group(2))
        if hour == 0 and minute == 0: return "midnight"
        if hour == 12 and minute == 0: return "noon"
        h_word = _verbalize_number(str(hour)); m_word = f"oh {_verbalize_number(str(minute))}" if minute < 10 else _verbalize_number(str(minute))
        if minute == 0: return f"{h_word} hundred hours"
        return f"{h_word} {m_word}"
    s = _TIME_12H_RE.sub(repl_time_12h, s)
    s = _TIME_24H_RE.sub(repl_time_24h, s)
    s = _TIMEZONE_RE.sub(lambda m: _TIMEZONES[m.group(1).upper()], s)
    s = _YEAR_RANGE_RE.sub(lambda m: f"{_verbalize_number(m.group(1), to_year=True)} to {_verbalize_number(m.group(2), to_year=True)}", s)
    s = _NUMBER_RANGE_RE.sub(lambda m: f"{_verbalize_number(m.group(1))} to {_verbalize_number(m.group(2))}", s)
    
    # Other complex patterns
    s = _SCIENTIFIC_NOTATION_RE.sub(lambda m: f"{_verbalize_number(m.group(1))} times ten to the power of {_verbalize_number(m.group(2))}", s)
    s = _US_PHONE_RE.sub(lambda m: " ".join([_verbalize_number(c) for c in f"{m.group(1)}{m.group(2)}{m.group(3)}"]), s)

    # Standalone numbers
    s = _FORMATTED_NUMBER_RE.sub(lambda m: _verbalize_number(m.group(0)), s)
    s = _SIMPLE_NUMBER_RE.sub(lambda m: _verbalize_number(m.group(0)), s) # Catch numbers missed by other rules

    # Roman numerals (common cases)
    _ROMAN_MAP = {"I": "one", "II": "two", "III": "three", "IV": "four", "V": "five", "VI": "six", "VII": "seven", "VIII": "eight", "IX": "nine", "X": "ten"}
    s = _ROMAN_NUMERAL_RE.sub(lambda m: _ROMAN_MAP.get(m.group(1), m.group(1)), s)

    # Ampersand and other symbols
    s = _AMPERSAND_RE.sub(" and ", s)
    s = _WORD_PLUS_RE.sub(lambda m: f"{m.group(1)} plus", s)
    s = re.sub(r"^\s*&\s*(?=\w)", "and ", s); s = re.sub(r"(?<=\w)\s*&\s*$", " and", s)

    # Section/Paragraph signs
    s = re.sub(r"§\s*", "section ", s); s = re.sub(r"¶\s*", "paragraph ", s)

    # µ/μ + units, Ω/kΩ/MΩ
    s = _MICRO_UNITS_RE.sub(lambda m: f"{_verbalize_number(m.group('num'))} micro{m.group('u')}", s)
    s = _KILOOHM_RE.sub(lambda m: f"{_verbalize_number(m.group('num'))} kiloohms", s)
    s = _MEGAOHM_RE.sub(lambda m: f"{_verbalize_number(m.group('num'))} megaohms", s)
    s = _OHM_RE.sub(lambda m: f"{_verbalize_number(m.group('num'))} ohms", s)

    # unit per slash (URLs are masked)
    s = _PER_SLASH_RE.sub(lambda m: f"{m.group('a')} per {m.group('b')}", s)

    # hashtags & mentions
    def repl_hashtag(m):
        tag = m.group('tag')
        spoken_tag = re.sub(r'([A-Z])', r' \1', tag).strip()
        return f"hashtag {spoken_tag}"
    s = _HASHTAG_NUM_RE.sub(lambda m: f"number {_verbalize_number(m.group('num'))}", s)
    s = _HASHTAG_TAG_RE.sub(repl_hashtag, s)
    s = _MENTION_RE.sub(lambda m: f"at {m.group('user')}", s)

    # TM / R / ©
    if speak_marks:
        s = s.replace("™", " trademark ").replace("®", " registered ").replace("©", " copyright ")
    else:
        s = s.replace("™", "").replace("®", "").replace("©", "")

    # Handle parentheses for prosody
    s = _PARENS_ACRONYM_RE.sub(lambda m: ", " + " ".join(list(m.group(1))) + ",", s)
    s = s.replace("(", ", ").replace(")", ", ")

    # Final cleanup
    s = _COMMA_CLEANUP_RE.sub(", ", s) # Clean up duplicate commas from parenthesis replacement
    s = re.sub(r"^,\s*", "", s)       # Remove leading comma
    s = re.sub(r"\s*,$", "", s)       # Remove trailing comma
    s = _WHITESPACE_RE.sub(" ", s).strip()
    s = _unmask(s, maskmap, read_urls=read_urls)
    return s


# =============================================================================
#                           ADVANCED SPLITTING
# =============================================================================

ABBREVIATIONS: Set[str] = {"mr.", "mrs.", "ms.", "dr.", "prof.", "rev.", "hon.", "st.", "etc.", "e.g.", "i.e.", "vs.", "approx.", "apt.", "dept.", "fig.", "gen.", "gov.", "inc.", "jr.", "sr.", "ltd.", "no.", "p.", "pp.", "vol.", "op.", "cit.", "ca.", "cf.", "ed.", "esp.", "et.", "al.", "ibid.", "id.", "inf.", "sup.", "viz.", "sc.", "fl.", "d.", "b.", "r.", "c.", "v.", "u.s.", "u.k.", "a.m.", "p.m.", "a.d.", "b.c."}
TITLES_NO_PERIOD: Set[str] = {"mr", "mrs", "ms", "dr", "prof", "rev", "hon", "st", "sgt", "capt", "lt", "col", "gen"}
NUMBER_DOT_NUMBER_PATTERN = re.compile(r"(?<!\d\.)\d*\.\d+")
VERSION_PATTERN = re.compile(r"[vV]?\d+(?:\.\d+)+")
POTENTIAL_END_PATTERN = re.compile(r'([.!?:]|(?<=\w):)(["\'”’)]?)(\s+|$)')
BULLET_POINT_PATTERN = re.compile(r"(?:^|\n)\s*(?:[-•*]|\d+\.)\s+")
UNICODE_ELLIPSIS = "…"

def _is_valid_sentence_end(text: str, period_index: int) -> bool:
    """Determines if a period marks a true sentence end."""
    if (period_index > 0 and text[period_index - 1] == ".") or \
       (period_index + 1 < len(text) and text[period_index + 1] == "."):
        return False
    word_start = period_index - 1
    scan_limit = max(0, period_index - 20)
    while word_start >= scan_limit and not text[word_start].isspace():
        word_start -= 1
    word_with_dot = text[word_start + 1: period_index + 1].lower()
    if word_with_dot in ABBREVIATIONS:
        return False
    context_start = max(0, period_index - 20); context_end = min(len(text), period_index + 20)
    context = text[context_start:context_end]
    rel_idx = period_index - context_start
    for pattern in (NUMBER_DOT_NUMBER_PATTERN, VERSION_PATTERN):
        for m in pattern.finditer(context):
            if m.start() <= rel_idx < m.end():
                is_last_char = (rel_idx == m.end() - 1)
                is_followed_by_space_or_eos = (period_index + 1 == len(text) or text[period_index + 1].isspace())
                if not (is_last_char and is_followed_by_space_or_eos):
                    return False
    return True

def _split_text_by_punctuation(text: str) -> List[str]:
    """Splits text based on punctuation, respecting abbreviations and numbers."""
    sentences: List[str] = []; last_split = 0
    i = 0
    while i < len(text):
        if text[i] == UNICODE_ELLIPSIS:
            j = i + 1
            if j >= len(text) or text[j].isspace():
                seg = text[last_split:j].strip()
                if seg: sentences.append(seg)
                last_split = j
            i += 1
            continue
        i += 1
    for m in POTENTIAL_END_PATTERN.finditer(text):
        punc_idx = m.start(1); punc = text[punc_idx]
        cut_after = m.start(1) + 1 + (len(m.group(2)) if m.group(2) else 0)
        if punc in ("!", "?"):
            seg = text[last_split:cut_after].strip()
            if seg: sentences.append(seg)
            last_split = m.end()
            continue
        if punc == ".":
            if _is_valid_sentence_end(text, punc_idx):
                seg = text[last_split:cut_after].strip()
                if seg: sentences.append(seg)
                last_split = m.end()
        if punc == ":":
            next_char_index = m.end()
            if next_char_index < len(text) and text[next_char_index].isupper():
                seg = text[last_split:cut_after].strip()
                if seg: sentences.append(seg)
                last_split = m.end()
    remainder = text[last_split:].strip()
    if remainder: sentences.append(remainder)
    return [s for s in sentences if s]

def _advanced_split_into_sentences(text: str) -> List[str]:
    """Splits text into sentences, handling bullet points and normalizing line breaks."""
    if not text or text.isspace(): return []
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    bullet_matches = list(BULLET_POINT_PATTERN.finditer(t))
    collected: List[str] = []
    def _append_sentences_from(segment: str):
        for s in _split_text_by_punctuation(segment.strip()):
            if s: collected.append(s)
    if bullet_matches:
        cur = 0
        for i, bm in enumerate(bullet_matches):
            start = bm.start()
            if i == 0 and start > cur:
                pre = t[cur:start].strip()
                if pre: _append_sentences_from(pre)
            next_start = bullet_matches[i + 1].start() if i + 1 < len(bullet_matches) else len(t)
            bullet_seg = t[start:next_start].strip()
            if bullet_seg: collected.append(bullet_seg)
            cur = next_start
        if cur < len(t):
            post = t[cur:].strip()
            if post: _append_sentences_from(post)
        return collected
    return _split_text_by_punctuation(t)

def _preprocess_and_segment_text_simple(full_text: str) -> List[str]:
    """Segments text into sentences."""
    if not full_text or full_text.isspace(): return []
    return _advanced_split_into_sentences(full_text)

# =============================================================================
#                          PUBLIC API
# =============================================================================

def split_text_into_chunks(text: str, max_length: int = None) -> list:
    if max_length is None: max_length = Config.MAX_CHUNK_LENGTH
    text = normalize_for_tts(text)
    if len(text) <= max_length: return [text]
    chunks: List[str] = []; current = ""
    sentences = _preprocess_and_segment_text_simple(text)
    for sentence in sentences:
        s = sentence.strip()
        if not s: continue
        if len(current) + (1 if current else 0) + len(s) <= max_length:
            current = (current + " " + s) if current else s
        else:
            if current: chunks.append(current.strip())
            if len(s) > max_length:
                chunks.extend(_split_long_sentence(s, max_length))
                current = ""
            else:
                current = s
    if current: chunks.append(current.strip())
    return [c for c in chunks if c.strip()]

def split_text_for_streaming(text: str, chunk_size: Optional[int] = None, strategy: Optional[str] = None, quality: Optional[str] = None) -> List[str]:
    text = normalize_for_tts(text)
    settings = get_streaming_settings(chunk_size, strategy, quality)
    chunk_size = settings["chunk_size"]; strategy = settings["strategy"]
    if strategy == "paragraph": return _split_by_paragraphs(text, chunk_size)
    elif strategy == "sentence": return _split_by_sentences(text, chunk_size)
    elif strategy == "word": return _split_by_words(text, chunk_size)
    elif strategy == "fixed": return _split_by_fixed_size(text, chunk_size)
    else: return _split_by_sentences(text, chunk_size)

def _split_by_paragraphs(text: str, max_length: int) -> List[str]:
    paragraphs = re.split(r'\n\s*\n', text.strip())
    chunks: List[str] = []; current = ""
    for p in [p.strip() for p in paragraphs if p.strip()]:
        if len(current) + (2 if current else 0) + len(p) <= max_length:
            current = (current + "\n\n" + p) if current else p
        else:
            if current: chunks.append(current.strip())
            if len(p) > max_length:
                chunks.extend(_split_by_sentences(p, max_length))
                current = ""
            else:
                current = p
    if current: chunks.append(current.strip())
    return [c for c in chunks if c.strip()]

def _pack_sentences_to_chunks(sentences: List[str], max_length: int) -> List[str]:
    chunks: List[str] = []; cur_parts: List[str] = []; cur_len = 0
    for s in [s.strip() for s in sentences if s.strip()]:
        if not cur_parts:
            cur_parts = [s]; cur_len = len(s)
        elif cur_len + 1 + len(s) <= max_length:
            cur_parts.append(s); cur_len += 1 + len(s)
        else:
            chunks.append(" ".join(cur_parts))
            cur_parts = [s]; cur_len = len(s)
        if cur_len > max_length and len(cur_parts) == 1:
            chunks.extend(_split_long_sentence(cur_parts[0], max_length))
            cur_parts = []; cur_len = 0
    if cur_parts: chunks.append(" ".join(cur_parts))
    return [c for c in chunks if c.strip()]

def _split_by_sentences(text: str, max_length: int) -> List[str]:
    return _pack_sentences_to_chunks(_preprocess_and_segment_text_simple(text), max_length)

def _split_by_words(text: str, max_length: int) -> List[str]:
    words = text.split(); chunks: List[str] = []; current = ""
    for word in words:
        if len(current) + (1 if current else 0) + len(word) <= max_length:
            current = (current + " " + word) if current else word
        else:
            if current: chunks.append(current.strip())
            if len(word) > max_length:
                for i in range(0, len(word), max_length):
                    if piece := word[i:i + max_length]: chunks.append(piece)
                current = ""
            else:
                current = word
    if current: chunks.append(current.strip())
    return [c for c in chunks if c.strip()]

def _split_by_fixed_size(text: str, chunk_size: int) -> List[str]:
    return [chunk for i in range(0, len(text), chunk_size) if (chunk := text[i:i + chunk_size].strip())]

def _split_long_sentence(sentence: str, max_length: int) -> List[str]:
    delimiters = [', ', '; ', ' - ', ' — ', ': ', ' and ', ' or ', ' but ']; chunks = [sentence]
    for delim in delimiters:
        new_chunks: List[str] = []
        for ch in chunks:
            if len(ch) <= max_length: new_chunks.append(ch)
            else:
                parts = ch.split(delim); cur = ""
                for i, part in enumerate(parts):
                    prospective = (delim if i > 0 else "") + part
                    if len(cur) + len(prospective) <= max_length: cur += prospective
                    else:
                        if cur: new_chunks.append(cur)
                        cur = part
                if cur: new_chunks.append(cur)
        chunks = new_chunks
    final_chunks: List[str] = []
    for ch in chunks:
        if len(ch) <= max_length: final_chunks.append(ch)
        else: final_chunks.extend(_split_by_words(ch, max_length))
    return [c.strip() for c in final_chunks if c.strip()]

def get_streaming_settings(streaming_chunk_size: Optional[int], streaming_strategy: Optional[str], streaming_quality: Optional[str]) -> dict:
    settings = {"chunk_size": streaming_chunk_size or 200, "strategy": streaming_strategy or "sentence", "quality": streaming_quality or "balanced"}
    if streaming_quality and not streaming_chunk_size:
        if streaming_quality == "fast": settings["chunk_size"] = 100
        elif streaming_quality == "high": settings["chunk_size"] = 300
    if streaming_quality and not streaming_strategy:
        if streaming_quality == "fast": settings["strategy"] = "word"
        elif streaming_quality == "high": settings["strategy"] = "paragraph"
    return settings

def concatenate_audio_chunks(audio_chunks: list, sample_rate: int) -> torch.Tensor:
    if not audio_chunks: return torch.tensor([])
    if len(audio_chunks) == 1: return audio_chunks[0]
    silence_samples = int(0.1 * sample_rate)
    device = audio_chunks[0].device if hasattr(audio_chunks[0], 'device') else 'cpu'
    silence = torch.zeros(1, silence_samples, device=device)
    with torch.no_grad():
        concatenated = audio_chunks[0]
        for i, chunk in enumerate(audio_chunks[1:], 1):
            concatenated = torch.cat([concatenated, silence, chunk], dim=1)
            if i % 10 == 0:
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
    return concatenated

def split_text_for_long_generation(text: str, max_chunk_size: Optional[int] = None, overlap_chars: int = 0) -> List[LongTextChunk]:
    if max_chunk_size is None: max_chunk_size = Config.LONG_TEXT_CHUNK_SIZE
    text = normalize_for_tts(text)
    effective_max = min(max_chunk_size, Config.MAX_TOTAL_LENGTH - 100)
    sane_overlap = min(overlap_chars, effective_max // 2)
    chunks: List[LongTextChunk] = []; idx = 0; remaining = text.strip()
    while remaining:
        if len(remaining) <= effective_max:
            chunk_text = remaining; remaining = ""
        else:
            chunk_text, remaining = _find_best_split_point(remaining, effective_max, sane_overlap)
        chunk = LongTextChunk(index=idx, text=chunk_text, text_preview=chunk_text[:50] + ("..." if len(chunk_text) > 50 else ""), character_count=len(chunk_text))
        chunks.append(chunk); idx += 1
    return chunks

def _find_best_split_point(text: str, max_length: int, overlap_chars: int = 0) -> Tuple[str, str]:
    if len(text) <= max_length: return text, ""
    if r := _try_split_at_paragraphs(text, max_length, overlap_chars): return r
    if r := _try_split_at_sentences(text, max_length, overlap_chars): return r
    if r := _try_split_at_clauses(text, max_length, overlap_chars): return r
    return _split_at_words(text, max_length, overlap_chars)

def _try_split_at_paragraphs(text: str, max_length: int, overlap_chars: int) -> Optional[Tuple[str, str]]:
    matches = list(re.finditer(r'\n\s*\n', text)); best = None
    if not matches: return None
    for m in matches:
        if (split_pos := m.end()) <= max_length: best = split_pos
        else: break
    if best and best > max_length * 0.5:
        chunk_text = text[:best].strip()
        remaining = text[max(0, best - overlap_chars):].strip()
        return chunk_text, remaining
    return None

def _try_split_at_sentences(text: str, max_length: int, overlap_chars: int) -> Optional[Tuple[str, str]]:
    sentences = _preprocess_and_segment_text_simple(text)
    if not sentences: return None
    cum = 0; last_ok_idx = -1
    for i, s in enumerate(sentences):
        add = (1 if cum > 0 else 0) + len(s)
        if cum + add <= max_length: cum += add; last_ok_idx = i
        else: break
    if last_ok_idx >= 0 and cum > max_length * 0.4:
        chunk_text = " ".join(sentences[:last_ok_idx + 1]).strip()
        original_start_pos = text.find(sentences[last_ok_idx + 1]) if last_ok_idx + 1 < len(sentences) else len(chunk_text)
        remaining = text[max(0, original_start_pos - overlap_chars):].strip()
        return chunk_text, remaining
    return None

def _try_split_at_clauses(text: str, max_length: int, overlap_chars: int) -> Optional[Tuple[str, str]]:
    clause_delims = [', ', '; ', ': ', ' - ', ' — ', ' and ', ' or ', ' but ', ' while ', ' when ']
    best_split = 0
    for d in clause_delims:
        if (pos := text.rfind(d, 0, max_length)) != -1:
            best_split = max(best_split, pos + len(d))
    if best_split and best_split > max_length * 0.3:
        chunk_text = text[:best_split].strip()
        remaining = text[max(0, best_split - overlap_chars):].strip()
        return chunk_text, remaining
    return None

def _split_at_words(text: str, max_length: int, overlap_chars: int) -> Tuple[str, str]:
    if len(text) <= max_length: return text, ""
    split_pos = text.rfind(' ', 0, max_length)
    if split_pos == -1: split_pos = max_length
    chunk_text = text[:split_pos].strip()
    remaining = text[max(0, split_pos - overlap_chars):].strip()
    return chunk_text, remaining

def estimate_processing_time(text_length: int, avg_chars_per_second: float = 25.0) -> int:
    base_time = text_length / avg_chars_per_second
    num_chunks = max(1, (text_length + Config.LONG_TEXT_CHUNK_SIZE - 1) // Config.LONG_TEXT_CHUNK_SIZE)
    overhead = 5 + (num_chunks * 2) + 10
    return int(base_time + overhead)

def validate_long_text_input(text: str) -> Tuple[bool, str]:
    if not text or not text.strip(): return False, "Input text cannot be empty"
    text_length = len(text.strip())
    if text_length <= Config.MAX_TOTAL_LENGTH: return False, f"Text is {text_length} characters. Use regular TTS for texts under {Config.MAX_TOTAL_LENGTH} characters"
    if text_length > Config.LONG_TEXT_MAX_LENGTH: return False, f"Text is too long ({text_length} characters). Maximum allowed: {Config.LONG_TEXT_MAX_LENGTH}"
    words = text.split()
    if len(words) > 50 and len(set(words)) < len(words) * 0.1: return False, "Text appears to be excessively repetitive"
    return True, ""
