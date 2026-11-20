"""The Svara model only works if audio tokens and the prompt match its training
layout exactly; an off-by-one band or block produces noise, not an error."""
import pytest

from voiceclone.svara.tokens import (AUDIO_START, AUDIO_TOKEN, BAND, BOS, EOA, EOH, EOS, EOT, SOA, SOH, SOS,
                                     build_prompt, frames_to_tokens, token_budget, tokens_to_frames,
                                     training_example)


def codes(n):
    c0 = [i % BAND for i in range(n)]
    c1 = [(100 + i) % BAND for i in range(2 * n)]
    c2 = [(200 + i) % BAND for i in range(4 * n)]
    return c0, c1, c2


def test_round_trip():
    c = codes(5)
    assert tokens_to_frames(frames_to_tokens(*c)) == c


def test_frame_interleave_order_and_band_offsets():
    c0, c1, c2 = [1], [2, 3], [4, 5, 6, 7]
    t = frames_to_tokens(c0, c1, c2)
    # order c0, c1[0], c2[0], c2[1], c1[1], c2[2], c2[3]; position k lives in band k
    assert t == [v + AUDIO_START + k * BAND for k, v in enumerate([1, 2, 4, 5, 3, 6, 7])]


def test_decode_stops_at_first_non_audio_token_and_drops_partial_frame():
    t = frames_to_tokens(*codes(3))
    junk = t[:10] + [EOS] + t[11:]       # a stray control token in frame 2
    c0, c1, c2 = tokens_to_frames(junk)
    assert len(c0) == 1 and len(c1) == 2 and len(c2) == 4


class FakeTok:
    def __call__(self, text, add_special_tokens=False):
        class R:
            input_ids = [ord(ch) for ch in text]
        return R()


def test_prompt_layout_with_reference_and_transcript():
    ref = [AUDIO_START + 1] * 7
    ids = build_prompt(FakeTok(), "B", ref, "A")
    assert ids == [BOS, SOH, AUDIO_TOKEN, ord("A"), EOH, EOT,
                   SOA, SOS, *ref, EOS, EOA, EOT,
                   SOH, AUDIO_TOKEN, ord("B"), EOH, EOT,
                   SOA, SOS]


def test_prompt_without_reference_transcript_skips_that_turn():
    ids = build_prompt(FakeTok(), "B", [AUDIO_START] * 7, None)
    assert ids[:3] == [BOS, SOA, SOS]


def test_training_example_matches_inference_prompt_and_masks_it():
    ref, tgt = [AUDIO_START + 1] * 7, [AUDIO_START + 2] * 7
    ids, labels = training_example(FakeTok(), "A", ref, "B", tgt)
    prompt = build_prompt(FakeTok(), "B", ref, "A")
    assert ids[: len(prompt)] == prompt                 # train exactly as we infer
    assert labels[: len(prompt)] == [-100] * len(prompt)  # no loss on the prompt
    assert labels[len(prompt):] == tgt + [EOS, EOA, EOT]  # learn to stop, too


def test_token_budget_caps_rambling_and_prevents_early_stop():
    mn, mx = token_budget(n_syllables=30, max_sec=20.0, min_sps=2.0, max_sps=6.0)
    assert mx == int((30 / 2.0 + 1.5) * 12 * 7) + 8     # 16.5 s cap
    assert mn == int(30 / 6.0 * 12) * 7                 # 5 s floor, whole frames
    assert mn % 7 == 0
    _, mx_long = token_budget(n_syllables=500, max_sec=20.0, min_sps=2.0, max_sps=6.0)
    assert mx_long == int(20.0 * 12 * 7) + 8            # never beyond max_sec


def test_token_budget_floor_never_exceeds_cap():
    # regression: 150 syllables need 25 s, the cap is 20 s -> floor was above the cap
    mn, mx = token_budget(n_syllables=150, max_sec=20.0, min_sps=2.0, max_sps=6.0)
    assert mn <= mx
    assert mx == int(20.0 * 12 * 7) + 8


@pytest.mark.parametrize("n", [0, 1])
def test_token_budget_tiny_inputs(n):
    mn, mx = token_budget(n, 20.0, 2.0, 6.0)
    assert 0 <= mn < mx


def test_marathi_sentence_is_not_cut_to_a_syllable():
    # regression: Devanagari used to count as 0 syllables -> 1.6 s cap
    from voiceclone.text import syllables

    _, mx = token_budget(syllables("नमस्कार, सगळे कसे आहात?"), 20.0, 2.5, 6.0)
    assert mx / (12 * 7) > 4.0


def test_unknown_length_falls_back_to_max_sec_cap():
    mn, mx = token_budget(0, 20.0, 2.0, 6.0)
    assert mn == 0 and mx == int(20.0 * 12 * 7) + 8
