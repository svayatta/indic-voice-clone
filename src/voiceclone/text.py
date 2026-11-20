"""Indic text utilities: transliteration, syllable counts, CER, aligner tokens.

The aligner's default romanizer (unidecode) drops the inherent vowel, so
"ನಿಮ್ಮ ಮನೆ" becomes "nimm mne". `translit` keeps the inherent 'a' unless a
virama or vowel sign follows, and outputs only a-z for the MMS CTC vocabulary.

Other Brahmic scripts are handled by mapping them into the Kannada block
first: Unicode lays out Devanagari, Bengali, Gurmukhi, Gujarati, Odia, Tamil,
Telugu, Kannada and Malayalam in parallel 128-codepoint blocks, so the same
offset is the same letter or sign. Scripts outside these blocks (for example
Urdu) transliterate to an empty string.
"""
import re
import unicodedata

V = {"ಅ": "a", "ಆ": "aa", "ಇ": "i", "ಈ": "ii", "ಉ": "u", "ಊ": "uu", "ಋ": "ru", "ಎ": "e", "ಏ": "ee",
     "ಐ": "ai", "ಒ": "o", "ಓ": "oo", "ಔ": "au"}
M = {"ಾ": "aa", "ಿ": "i", "ೀ": "ii", "ು": "u", "ೂ": "uu", "ೃ": "ru", "ೆ": "e", "ೇ": "ee", "ೈ": "ai",
     "ೊ": "o", "ೋ": "oo", "ೌ": "au"}
C = {"ಕ": "k", "ಖ": "kh", "ಗ": "g", "ಘ": "gh", "ಙ": "n", "ಚ": "ch", "ಛ": "chh", "ಜ": "j", "ಝ": "jh",
     "ಞ": "n", "ಟ": "t", "ಠ": "th", "ಡ": "d", "ಢ": "dh", "ಣ": "n", "ತ": "t", "ಥ": "th", "ದ": "d",
     "ಧ": "dh", "ನ": "n", "ಪ": "p", "ಫ": "ph", "ಬ": "b", "ಭ": "bh", "ಮ": "m", "ಯ": "y", "ರ": "r",
     "ಱ": "r", "ಲ": "l", "ವ": "v", "ಶ": "sh", "ಷ": "sh", "ಸ": "s", "ಹ": "h", "ಳ": "l", "ೞ": "l"}
VIRAMA, ANUSVARA, VISARGA, NUKTA = "್", "ಂ", "ಃ", "಼"
STAR = "<star>"

KANNADA_BLOCK = 0x0C80
BRAHMIC_BLOCKS = (0x0900, 0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0D00)
NUKTA_OFFSET = 0x3C
# Letters with no Kannada letter at the same offset: Devanagari candra vowels, Tamil and
# Malayalam ன and ழ, Malayalam chillu letters (consonant + virama), Bengali khanda ta
EXCEPTIONS = {"\u090D": "ಎ", "\u0911": "ಒ", "\u0945": "ೆ", "\u0949": "ೊ",
              "\u0BA9": "ನ", "\u0D29": "ನ", "\u0BB4": "ೞ", "\u0D34": "ೞ",
              "\u0D7A": "ಣ್", "\u0D7B": "ನ್", "\u0D7C": "ರ್", "\u0D7D": "ಲ್", "\u0D7E": "ಳ್", "\u0D7F": "ಕ್",
              "\u09CE": "ತ್"}


def _decompose_nukta(ch: str) -> str:
    """Split precomposed nukta letters (क़ -> क + ़) and nothing else.

    Full NFD would also split Tamil, Malayalam and Bengali two-part vowel
    signs (ோ -> ெ + ா), which would then transliterate as the wrong vowel.
    """
    d = unicodedata.decomposition(ch)
    if d and not d.startswith("<"):
        parts = [int(x, 16) for x in d.split()]
        if len(parts) == 2 and (parts[1] & 0x7F) == NUKTA_OFFSET:
            return chr(parts[0]) + chr(parts[1])
    return ch


def to_kannada_script(text: str) -> str:
    """Re-encode any parallel-block Brahmic script as Kannada; other characters unchanged."""
    out = []
    for ch in "".join(_decompose_nukta(c) for c in text):
        if ch in EXCEPTIONS:
            out.append(EXCEPTIONS[ch]); continue
        c = ord(ch)
        for base in BRAHMIC_BLOCKS:
            if base <= c < base + 0x80:
                ch = chr(c - base + KANNADA_BLOCK)
                break
        out.append(ch)
    return "".join(out)


def translit(word: str) -> str:
    word = to_kannada_script(word)
    out, i, n = [], 0, len(word)
    while i < n:
        ch = word[i]
        if ch in C:
            out.append(C[ch])
            j = i + 1
            if j < n and word[j] == NUKTA:
                j += 1
            if j < n and word[j] == VIRAMA:
                i = j + 1
            elif j < n and word[j] in M:
                out.append(M[word[j]]); i = j + 1
            else:
                out.append("a"); i = j
        elif ch in V:
            out.append(V[ch]); i += 1
        elif ch == ANUSVARA:
            out.append("m"); i += 1
        elif ch == VISARGA:
            out.append("h"); i += 1
        elif "a" <= ch.lower() <= "z":
            out.append(ch.lower()); i += 1
        else:
            i += 1  # punctuation, ZWJ/ZWNJ, digits
    return "".join(out)


def syllables(text: str) -> int:
    """Vowel groups in the transliteration: a cheap syllable count for rate checks."""
    return len(re.findall(r"[aeiou]+", " ".join(translit(w) for w in text.split())))


def romanize(text: str) -> str:
    """Letters-only transliteration, the reference string for CER."""
    return re.sub(r"[^a-z]", "", "".join(translit(w) for w in text.split()))


def cer(hyp: str, ref: str) -> float:
    """Character error rate: Levenshtein distance / reference length."""
    import numpy as np

    d = np.arange(len(hyp) + 1)
    for j, rc in enumerate(ref, 1):
        prev, d[0] = d[0], j
        for i, hc in enumerate(hyp, 1):
            prev, d[i] = d[i], min(d[i] + 1, d[i - 1] + 1, prev + (hc != rc))
    return d[len(hyp)] / max(1, len(ref))


def strip_stage_directions(text: str) -> str:
    return re.sub(r"\([^)]*\)", " ", text)


def strip_quotes(text: str) -> str:
    text = re.sub(r"[\"'‘’“”]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def collapse_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def star_tokens(text: str) -> tuple[list[str], list[str]]:
    """Aligner inputs: each word preceded by a <star> wildcard, plus a trailing <star>.

    The trailing <star> soaks up any audio after the last word so it is not
    glued onto that word. Returns ([], []) if no word transliterates.
    """
    toks, txt = [], []
    for w in text.split():
        lat = translit(w)
        if lat:
            toks += [STAR, " ".join(lat)]; txt += [STAR, w]
    if not toks:
        return [], []
    return toks + [STAR], txt + [STAR]


def split_sentences(text: str) -> list[str]:
    """Split after . ? ! । ॥ where white space or a new word follows.

    "ಒಂದು।ಎರಡು" (no space) gives two sentences. A number such as "3.5", a row
    of marks ("...") and a closing quote or bracket after the mark stay whole.
    """
    return [s for s in re.split(r"(?<=[.?!।॥])(?:\s+|(?=[^\s\d.?!।॥\"'”’)\]]))", text.strip()) if s]
