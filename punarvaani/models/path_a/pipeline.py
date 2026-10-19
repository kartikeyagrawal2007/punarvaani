"""Path A end to end: lips -> characters -> (phrase | open text) -> correction -> voice.

Decision flow for one utterance::

    CTC posteriors ──► phrase bank match ──accepted──► phrase text (pre-rendered audio)
                             │ rejected
                             ▼
                 prefix beam search (+char LM) ─► attention rescoring ─► n-best
                             ▼
                 corrector (LLM, else lexicon) ─► TTS in the user's voice

In ``mode="phrase"`` a rejection returns nothing and the app asks the user to
repeat - saying nothing beats saying the wrong thing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from punarvaani.models.path_a.decoding import CharNGramLM, attention_rescore, ctc_prefix_beam_search
from punarvaani.models.path_a.llm_correction import CorrectionContext, Corrector
from punarvaani.models.path_a.model import VSRModel
from punarvaani.models.path_a.phrase_mode import PhraseBank, PhraseMatch
from punarvaani.models.path_a.tts import TTSBackend
from punarvaani.text.vocab import CharVocab


@dataclass
class PathAResult:
    text: str
    mode: str  # "phrase", "open", "rejected"
    confidence: float
    nbest: list[tuple[str, float]] = field(default_factory=list)
    phrase_match: PhraseMatch | None = None
    correction_source: str = ""
    audio: torch.Tensor | None = None
    sample_rate: int = 16000
    timings_ms: dict[str, float] = field(default_factory=dict)


class PathAPipeline:
    def __init__(
        self,
        model: VSRModel,
        vocab: CharVocab,
        phrase_bank: PhraseBank | None = None,
        corrector: Corrector | None = None,
        tts: TTSBackend | None = None,
        lm: CharNGramLM | None = None,
        mode: str = "auto",
        beam: int = 10,
        lm_weight: float = 0.3,
        length_bonus: float = 0.5,
        rescore_ctc_weight: float = 0.3,
    ):
        if mode not in ("auto", "phrase", "open"):
            raise ValueError(mode)
        self.model, self.vocab, self.phrase_bank = model.eval(), vocab, phrase_bank
        self.corrector, self.tts, self.lm = corrector, tts, lm
        self.mode, self.beam, self.lm_weight = mode, beam, lm_weight
        self.length_bonus, self.rescore_ctc_weight = length_bonus, rescore_ctc_weight
        self.context = CorrectionContext(phrases=list(phrase_bank.phrases) if phrase_bank else [])

    @torch.no_grad()
    def recognize(self, video: torch.Tensor, speak: bool = True) -> PathAResult:
        """``video``: (T, H, W) normalised mouth crops of one utterance."""
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        device = next(self.model.parameters()).device
        out = self.model(video.unsqueeze(0).to(device), torch.tensor([video.shape[0]], device=device))
        log_probs = out.ctc_log_probs[0].cpu()
        timings["encoder"] = (time.perf_counter() - t0) * 1e3

        result: PathAResult | None = None
        if self.phrase_bank is not None and self.mode in ("auto", "phrase"):
            t = time.perf_counter()
            match = self.phrase_bank.match(log_probs)
            timings["phrase_match"] = (time.perf_counter() - t) * 1e3
            if match.accepted:
                result = PathAResult(match.phrase, "phrase", match.posterior, phrase_match=match)
            elif self.mode == "phrase":
                result = PathAResult("", "rejected", match.posterior, phrase_match=match)

        if result is None:
            t = time.perf_counter()
            hyps = ctc_prefix_beam_search(
                log_probs, self.beam, self.vocab.blank_id, self.lm, self.lm_weight, self.length_bonus
            )
            hyps = attention_rescore(
                self.model, out.encoder_out, out.padding_mask, hyps, self.rescore_ctc_weight
            )
            nbest = [(self.vocab.decode(h.tokens), h.score) for h in hyps]
            timings["beam_search"] = (time.perf_counter() - t) * 1e3
            text, conf, source = (nbest[0][0] if nbest else ""), 0.5, "beam"
            if self.corrector is not None and nbest:
                t = time.perf_counter()
                corr = self.corrector.correct(nbest, self.context)
                timings["correction"] = (time.perf_counter() - t) * 1e3
                text, conf, source = corr.text, corr.confidence, corr.source
            result = PathAResult(text, "open", conf, nbest=nbest, correction_source=source)

        if speak and self.tts is not None and result.text:
            t = time.perf_counter()
            result.audio = self.tts.synthesize(result.text)
            result.sample_rate = self.tts.sample_rate
            timings["tts"] = (time.perf_counter() - t) * 1e3
        if result.text:
            self.context.recent_turns.append(result.text)
        timings["total"] = (time.perf_counter() - t0) * 1e3
        result.timings_ms = timings
        return result
