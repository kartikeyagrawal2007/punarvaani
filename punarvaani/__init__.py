"""Punarvaani (पुनर्वाणी): Hindi lip-to-speech in the user's own voice.

Package map (mirrors the architecture in README.md):

    text/           Hindi normalisation, character vocabulary, phonemes and visemes
    preprocessing/  [1] lip crop from face landmarks, audio features
    data_engine/    Hindi audio-visual dataset mining (segmenting, sync check, manifests)
    datasets/       PyTorch datasets, collation, augmentation, synthetic toy data
    models/encoder  [2] AV-HuBERT-style visual speech encoder + LoRA
    models/path_a   Path A: lips -> Hindi text -> correction -> TTS
    models/path_b   Path B: lips -> speech units -> unit HiFi-GAN vocoder
    models/hybrid   Stretch: Path A text guides Path B units
    voice_rebuild/  old voice notes -> clean reference + speaker embedding
    training/       training loops (VSR, units, vocoder, personalisation)
    eval/           WER/CER, speaker similarity, MOS, latency, significance tests
    app/            real-time streaming pipeline
"""

__version__ = "0.1.0"
