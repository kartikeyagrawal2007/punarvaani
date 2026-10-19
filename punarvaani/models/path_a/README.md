# models/path_a

Path A ("stenographer + newsreader"): lips → Hindi characters → phrase match or n-best → context correction → TTS in the user's voice.

| File | What |
|---|---|
| `model.py` | `VSRModel`: encoder + CTC head + transformer decoder, hybrid CTC/attention loss |
| `decoding.py` | CTC greedy, prefix beam search (log space), Witten-Bell character n-gram LM fusion, attention rescoring |
| `phrase_mode.py` | closed-set recognition by exact CTC likelihood, usage prior, rejection, viseme-level text fallback |
| `llm_correction.py` | `ClaudeCorrector` (structured JSON, offline fallback) and `LexiconCorrector` (viseme Viterbi + bigram LM) |
| `tts.py` | IndicF5 zero-shot voice cloning wrapper, test TTS, phrase audio cache |
| `pipeline.py` | `PathAPipeline` with per-stage latency |

**Owner:** System lead
