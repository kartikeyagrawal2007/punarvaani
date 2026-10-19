# models/hybrid

Stretch goal: Path B generates the sound while Path A's text guides it toward the correct words.

`guidance.py`: a unit-ASR model (soft units → character CTC, trained on audio only) and `guide_units`, which takes gradient steps on Path B's unit logits to lower the CTC loss of Path A's text, with a trust-region term that keeps Path B's timing.

**Owner:** Speech + system leads
