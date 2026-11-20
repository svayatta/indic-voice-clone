"""Transliteration feeds the aligner, rate filters and CER, so mistakes here
silently corrupt the training set rather than raising errors."""
from voiceclone.text import STAR, cer, romanize, star_tokens, syllables, translit


def test_inherent_vowel_is_kept():
    # unidecode turns this into "nimm mne"; the MMS aligner then cannot find the vowels
    assert translit("ನಿಮ್ಮ") == "nimma"
    assert translit("ಮನೆ") == "mane"


def test_virama_suppresses_inherent_vowel():
    assert translit("ಆಗ್ಬೇಕು") == "aagbeeku"


def test_anusvara_visarga_and_conjuncts():
    assert translit("ಅಹಂ") == "aham"
    assert translit("ದುಃಖಿ") == "duhkhi"
    assert translit("ಕ್ಷಮಾ") == "kshamaa"


def test_output_is_only_mms_vocabulary_letters():
    out = translit("ಹ್ಯಾಗೆ, ರೀ! 123 ‌")
    assert set(out) <= set("abcdefghijklmnopqrstuvwxyz")


def test_syllables_count_vowel_groups():
    # rate filters divide by this: "ನಮಸ್ಕಾರ" = na-mas-kaa-ra
    assert syllables("ನಮಸ್ಕಾರ") == 4
    assert syllables("") == 0


def test_romanize_drops_spaces_and_punctuation():
    assert romanize("ನಿಮ್ಮ ಮನೆ.") == "nimmamane"


def test_cer_basics():
    assert cer("abc", "abc") == 0
    assert cer("abd", "abc") == 1 / 3
    assert cer("", "abcd") == 1.0
    assert cer("x", "") == 1.0    # no division by zero on empty reference


def test_star_tokens_wrap_words_and_absorb_trailing_audio():
    toks, txt = star_tokens("ನಿಮ್ಮ , ಮನೆ")
    # punctuation-only "words" are dropped, each word is preceded by <star>,
    # and a trailing <star> keeps extra audio off the last word
    assert toks == [STAR, "n i m m a", STAR, "m a n e", STAR]
    assert txt == [STAR, "ನಿಮ್ಮ", STAR, "ಮನೆ", STAR]
    assert star_tokens("... !") == ([], [])


# --- other Indic scripts -------------------------------------------------
# Svara speaks 19 languages. If text in another script transliterates to "",
# syllables() is 0, synthesis is capped at about 1.6 s, the aligner gets no
# tokens and best-of divides by zero. These pin the multi-script path.
import pytest

MULTI = {
    "Marathi": ("नमस्कार, सगळे कसे आहात?", "namaskaara"),
    "Hindi": ("नमस्ते", "namastee"),
    "Telugu": ("నమస్కారం", "namaskaaram"),
    "Tamil": ("வணக்கம்", "vanakkam"),
    "Malayalam": ("നമസ്കാരം", "namaskaaram"),
    "Bengali": ("নমস্কার", "namaskaara"),
    "Gujarati": ("નમસ્તે", "namastee"),
    "Gurmukhi": ("ਸਤ", "sata"),
}


@pytest.mark.parametrize("lang", MULTI)
def test_other_scripts_transliterate(lang):
    text, first_word = MULTI[lang]
    assert translit(text.split()[0].strip(",")) == first_word
    assert syllables(text) > 0
    assert star_tokens(text)[0], "aligner must get tokens"


def test_two_part_vowel_signs_are_not_split():
    # Tamil ோ canonically decomposes to ெ + ா; splitting it would read as "e"
    assert translit("எல்லோரும்") == "elloorum"


def test_nukta_letters_and_candra_vowels():
    assert translit("ज़रूर") == "jaruura"      # precomposed nukta letter
    assert translit("डॉक्टर") == "doktara"     # candra o, as in English loanwords


@pytest.mark.parametrize("word, roman", [
    ("என்ன", "enna"), ("மழை", "malai"),                       # Tamil ன and ழ have no Kannada twin
    ("അവൻ", "avan"), ("അവൾ", "aval"), ("അവർ", "avar"),       # Malayalam chillu letters
    ("കൽ", "kal"), ("മഴ", "mala"), ("অকস্মাৎ", "akasmaat"),     # Malayalam ഴ, Bengali khanda ta
])
def test_letters_without_a_kannada_twin_are_kept(word, roman):
    assert translit(word) == roman


def test_syllables_count_letters_without_a_kannada_twin():
    assert syllables("என்ன") == 2


def test_kannada_unchanged_by_script_mapping():
    assert translit("ನಿಮ್ಮ") == "nimma" and translit("ಕ್ಷಮಾ") == "kshamaa"


def test_unsupported_script_gives_empty_not_garbage():
    assert translit("سلام") == "" and syllables("سلام") == 0
