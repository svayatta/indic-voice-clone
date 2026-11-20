"""Svara-TTS prompt layout and SNAC audio-token mapping (pure Python, no torch).

Prompt (from Kenpath/svara-tts-inference):
  BOS
  [human]  SOH <|audio|> ref_transcript_ids EOH EOT        (optional)
  [ai]     SOA SOS ref_audio_tokens EOS EOA EOT
  [human]  SOH <|audio|> target_text_ids EOH EOT
  [ai]     SOA SOS  -> model generates audio tokens, then EOS EOA

Audio: 7 tokens per SNAC coarse frame (about 12 frames/s), in the order
  c0[i], c1[2i], c2[4i], c2[4i+1], c1[2i+1], c2[4i+2], c2[4i+3]
each offset into the vocabulary as code + 128266 + position*4096.
"""
BOS, EOT = 128000, 128009
SOS, EOS, SOH, EOH, SOA, EOA, PAD = 128257, 128258, 128259, 128260, 128261, 128262, 128263
AUDIO_TOKEN = 156939
AUDIO_START, BAND = 128266, 4096
FRAMES_PER_SEC = 12
TOKENS_PER_FRAME = 7


def frames_to_tokens(c0: list[int], c1: list[int], c2: list[int]) -> list[int]:
    """Interleave SNAC's three code levels into vocabulary-offset audio tokens."""
    out = []
    for i in range(len(c0)):
        frame = [c0[i], c1[2 * i], c2[4 * i], c2[4 * i + 1], c1[2 * i + 1], c2[4 * i + 2], c2[4 * i + 3]]
        out.extend(v + AUDIO_START + b * BAND for b, v in enumerate(frame))
    return out


def tokens_to_frames(tokens: list[int]) -> tuple[list[int], list[int], list[int]]:
    """Inverse of frames_to_tokens. Stops at the first token outside its band
    (the model emitted something that is not audio) and drops a partial frame."""
    codes = []
    for i, t in enumerate(tokens):
        v = t - AUDIO_START - (i % TOKENS_PER_FRAME) * BAND
        if v < 0 or v >= BAND:
            break
        codes.append(v)
    n = len(codes) // TOKENS_PER_FRAME
    f = [codes[k * 7:(k + 1) * 7] for k in range(n)]
    c0 = [r[0] for r in f]
    c1 = [x for r in f for x in (r[1], r[4])]
    c2 = [x for r in f for x in (r[2], r[3], r[5], r[6])]
    return c0, c1, c2


def human_block(tok, text: str) -> list[int]:
    ids = tok(text, add_special_tokens=False).input_ids
    return [SOH, AUDIO_TOKEN, *ids, EOH, EOT]


def audio_block(tokens: list[int]) -> list[int]:
    return [SOA, SOS, *tokens, EOS, EOA, EOT]


def build_prompt(tok, target_text: str, ref_tokens: list[int] | None, ref_text: str | None) -> list[int]:
    ids = [BOS]
    if ref_tokens is not None:
        if ref_text:
            ids += human_block(tok, ref_text)
        ids += audio_block(ref_tokens)
    ids += human_block(tok, target_text)
    ids += [SOA, SOS]
    return ids


def training_example(tok, ref_text: str, ref_tokens: list[int], text: str, tokens: list[int]):
    """Reference-swap pair: same layout as inference, loss only on the target audio block."""
    prompt = [BOS] + human_block(tok, ref_text) + audio_block(ref_tokens) + human_block(tok, text) + [SOA, SOS]
    answer = tokens + [EOS, EOA, EOT]
    return prompt + answer, [-100] * len(prompt) + answer


def token_budget(n_syllables: int, max_sec: float, min_sps: float, max_sps: float) -> tuple[int, int]:
    """(min_new_tokens, max_new_tokens) for a sentence of `n_syllables`.

    The cap is syllables/min_sps + 1.5 s (at most max_sec) so a slow model
    cannot ramble; the floor is syllables/max_sps s so it cannot stop after
    a syllable. Both are whole frames of 7 tokens (+8 slack on the cap).
    With no syllables (text in a script `translit` cannot read) the length
    cannot be estimated, so only the max_sec cap applies. The floor never
    exceeds the cap, even when the text needs more speech than max_sec.
    """
    if n_syllables <= 0:
        return 0, int(max_sec * FRAMES_PER_SEC * TOKENS_PER_FRAME) + 8
    cap = min(max_sec, n_syllables / min_sps + 1.5)
    max_new = int(cap * FRAMES_PER_SEC * TOKENS_PER_FRAME) + 8
    min_new = int(n_syllables / max_sps * FRAMES_PER_SEC) * TOKENS_PER_FRAME
    return min(min_new, max_new), max_new
