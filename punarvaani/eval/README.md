# eval

| File | What |
|---|---|
| `metrics.py` | corpus WER/CER with S/D/I, viseme error rate, paired bootstrap, MOS with CI, EER, latency meter |
| `evaluate.py` | evaluate a run: greedy vs beam(+LM), phrase-mode accuracy/rejection, latency; `compare()` two runs |

Run: `punarvaani evaluate --run outputs/hindi_adapt --manifest data/hindi_av/test.jsonl --phrases configs/phrases_hi.txt`

**Owner:** System lead
