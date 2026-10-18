import math

import pytest
import torch
import torch.nn.functional as F

from punarvaani.models.encoder.avhubert import EncoderConfig, VisualSpeechEncoder
from punarvaani.models.path_a.decoding import (
    CharNGramLM,
    attention_rescore,
    ctc_greedy,
    ctc_prefix_beam_search,
)
from punarvaani.models.path_a.model import VSRModel
from punarvaani.models.path_a.phrase_mode import PhraseBank
from punarvaani.text.vocab import CharVocab


def _ctc_ll(log_probs, tokens):
    return -F.ctc_loss(
        log_probs.unsqueeze(1), torch.tensor([tokens]), torch.tensor([log_probs.shape[0]]),
        torch.tensor([len(tokens)]), reduction="sum",
    ).item()  # fmt: skip


def test_greedy_merges_repeats_and_drops_blanks():
    v = 5
    path = [1, 1, 0, 1, 2, 2, 0, 0, 3]
    lp = torch.full((1, len(path), v), -10.0)
    lp[0, torch.arange(len(path)), torch.tensor(path)] = 0.0
    assert ctc_greedy(lp) == [[1, 1, 2, 3]]


def test_prefix_beam_search_scores_equal_exact_ctc_likelihood():
    torch.manual_seed(0)
    lp = torch.randn(8, 4).log_softmax(-1)
    hyps = ctc_prefix_beam_search(lp, beam=200, prune_top_k=4, prune_threshold=-1e9)
    for h in hyps[:10]:
        if h.tokens:
            assert math.isclose(h.ctc_score, _ctc_ll(lp, h.tokens), rel_tol=1e-4, abs_tol=1e-4)
    # Best prefix-beam hypothesis is at least as likely as the greedy one.
    greedy = ctc_greedy(lp.unsqueeze(0))[0]
    assert hyps[0].ctc_score >= _ctc_ll(lp, greedy) - 1e-6


def test_char_lm_prefers_seen_sequences_and_normalises():
    lm = CharNGramLM(order=3, vocab_size=6).fit([[1, 2, 3], [1, 2, 3], [1, 2, 4]])
    assert lm.prob([1, 2], 3) > lm.prob([1, 2], 4) > lm.prob([1, 2], 5)
    assert math.isclose(sum(lm.prob([1, 2], t) for t in range(6)), 1.0, rel_tol=1e-6)


def test_lm_fusion_breaks_a_visual_tie():
    # Two tokens equally likely from video; the LM knows the sequence 1 -> 2.
    lp = torch.log(torch.tensor([[0.1, 0.9, 0.0, 0.0], [0.1, 0.0, 0.45, 0.45]]) + 1e-9)
    lm = CharNGramLM(order=2, vocab_size=4).fit([[1, 2]] * 20)
    assert ctc_prefix_beam_search(lp, beam=4, lm=lm, lm_weight=1.0)[0].tokens == [1, 2]


@pytest.fixture
def tiny_vsr():
    torch.manual_seed(0)
    enc = VisualSpeechEncoder(EncoderConfig.preset("tiny"))
    return VSRModel(enc, vocab_size=12, decoder_layers=1, decoder_ffn=64)


def test_vsr_forward_losses_and_backward(tiny_vsr):
    video = torch.randn(2, 16, 44, 44)
    lengths = torch.tensor([16, 12])
    targets = torch.tensor([[5, 6, 7], [8, 9, 1]])  # second one padded with pad_id=1
    out = tiny_vsr(video, lengths, targets, torch.tensor([3, 2]))
    assert out.ctc_log_probs.shape == (2, 16, 12)
    assert torch.isfinite(out.loss) and out.att_loss > 0
    out.loss.backward()
    assert tiny_vsr.ctc_head[1].weight.grad is not None


def test_decoder_io_and_rescoring(tiny_vsr):
    dec_in, dec_out = tiny_vsr.make_decoder_io(torch.tensor([[5, 6, 7], [8, 9, 1]]), torch.tensor([3, 2]))
    assert dec_in.tolist() == [[2, 5, 6, 7], [2, 8, 9, 1]]
    assert dec_out.tolist() == [[5, 6, 7, 2], [8, 9, 2, -100]]
    tiny_vsr.eval()
    out = tiny_vsr(torch.randn(1, 10, 44, 44), torch.tensor([10]))
    hyps = ctc_prefix_beam_search(out.ctc_log_probs[0], beam=3)
    rescored = attention_rescore(tiny_vsr, out.encoder_out, out.padding_mask, hyps)
    assert rescored and all(h.att_score <= 0 for h in rescored)
    assert rescored == sorted(rescored, key=lambda h: h.score, reverse=True)


def _posteriors_for(vocab, text, frames_per_char=3, noise=0.0):
    """Fake CTC output that clearly spells ``text``."""
    ids = vocab.encode(text)
    t = len(ids) * frames_per_char
    lp = torch.full((t, len(vocab)), -8.0)
    for k, tok in enumerate(ids):
        lp[k * frames_per_char, tok] = 0.0
        lp[k * frames_per_char + 1 : (k + 1) * frames_per_char, vocab.blank_id] = 0.0
    lp = lp + noise * torch.randn_like(lp)
    return lp.log_softmax(-1)


PHRASES = ["मुझे पानी चाहिए", "दर्द हो रहा है", "डॉक्टर को बुलाओ", "मुझे भूख लगी है", "धन्यवाद"]


def test_phrase_mode_picks_the_spoken_phrase():
    vocab = CharVocab()
    bank = PhraseBank(PHRASES, vocab)
    m = bank.match(_posteriors_for(vocab, "दर्द हो रहा है"))
    assert m.accepted and m.phrase == "दर्द हो रहा है" and m.posterior > 0.99


def test_phrase_mode_rejects_out_of_set_speech():
    vocab = CharVocab()
    bank = PhraseBank(PHRASES, vocab)
    m = bank.match(_posteriors_for(vocab, "कल बारिश होगी क्या"))
    assert not m.accepted


def test_prior_breaks_ties_and_learns_from_confirmations():
    vocab = CharVocab()
    bank = PhraseBank(["पल", "बल"], vocab)  # identical on the lips
    lp = (_posteriors_for(vocab, "पल").exp() * 0.5 + _posteriors_for(vocab, "बल").exp() * 0.5).log()
    for _ in range(5):
        bank.confirm(1)
    assert bank.match(lp, min_posterior=0.0).phrase == "बल"


def test_text_fallback_matches_on_visemes():
    bank = PhraseBank(PHRASES, CharVocab())
    # "बानी" / "भुझे" are lip-reading confusions of पानी / मुझे
    assert bank.match_text("भुझे बानी चाहिए")[0][0] == "मुझे पानी चाहिए"


def test_cached_tts_and_pipeline_smoke(tmp_path):
    from punarvaani.models.path_a.pipeline import PathAPipeline
    from punarvaani.models.path_a.tts import CachedTTS, ToneTTS

    vocab = CharVocab()
    torch.manual_seed(0)
    model = VSRModel(
        VisualSpeechEncoder(EncoderConfig.preset("tiny")), len(vocab), decoder_layers=1, decoder_ffn=64
    )
    tts = CachedTTS(ToneTTS(), tmp_path)
    assert tts.prerender(PHRASES) == len(PHRASES)
    assert torch.equal(CachedTTS(ToneTTS(), tmp_path).synthesize(PHRASES[0]), tts.synthesize(PHRASES[0]))

    pipe = PathAPipeline(model, vocab, PhraseBank(PHRASES, vocab), tts=tts, beam=3)
    res = pipe.recognize(torch.randn(30, 44, 44))
    assert res.mode in ("phrase", "open") and "total" in res.timings_ms
    strict = PathAPipeline(model, vocab, PhraseBank(PHRASES, vocab), mode="phrase")
    assert strict.recognize(torch.randn(30, 44, 44)).mode in ("phrase", "rejected")
