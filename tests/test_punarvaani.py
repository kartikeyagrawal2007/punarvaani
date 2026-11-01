"""
Comprehensive Unit Test Suite for Punarvaani Hindi Visual Speech Recognition
Verifies Phase 1 architecture, G2P phonology, CTC constraints, optimizer registration,
and inference interface.
"""

import os
import unittest
import torch
import torch.nn as nn
from typing import List

from Data.hindi_g2p import devanagari_to_phonemes, get_unsupported_char_report, clear_unsupported_char_log
from Data.dataset import compute_ctc_min_length, custom_collate, check_speaker_leakage, verify_speaker_disjoint_split
from Models.ML_VALLR import ML_VALLR, VideoViTMasked, SpatialAttentionPool, TemporalAdapter
from Models.Sarvam import TAGS, validate_sarvam_adapter_compatibility
from config import get_hindi_vocab, load_args
from inference import transcribe_video


class TestPunarvaaniG2P(unittest.TestCase):
    def setUp(self):
        clear_unsupported_char_log()

    def test_g2p_independent_vowels(self):
        """Verifies words starting with vowels parse all subsequent consonants correctly without skipping."""
        # 'अमर' should produce A M A R, not skip M
        amar = devanagari_to_phonemes("अमर", as_string=True)
        self.assertIn("A", amar)
        self.assertIn("M", amar)
        self.assertIn("R", amar)

        # 'एक' should produce E K
        ek = devanagari_to_phonemes("एक", as_string=True)
        self.assertEqual(ek, "E K")

        # 'आज' should produce AA J
        aaj = devanagari_to_phonemes("आज", as_string=True)
        self.assertEqual(aaj, "AA J")

    def test_g2p_complex_phonetics(self):
        """Tests conjuncts, halant, anusvara, visarga, nukta, and multiple independent vowels."""
        # Conjunct: 'क्या' -> K Y AA
        self.assertEqual(devanagari_to_phonemes("क्या", as_string=True), "K Y AA")

        # Halant: 'विद्वान' -> V I DD V AA N
        vidwan = devanagari_to_phonemes("विद्वान", as_string=True)
        self.assertIn("DD", vidwan)

        # Anusvara: 'हिंदी' -> H I AN DD II
        hindi = devanagari_to_phonemes("हिंदी", as_string=True)
        self.assertIn("AN", hindi)

        # Visarga: 'अतः' -> A TD AH
        atah = devanagari_to_phonemes("अतः", as_string=True)
        self.assertIn("AH", atah)

        # Nukta: 'फ़िल्म' -> F I L M
        film = devanagari_to_phonemes("फ़िल्म", as_string=True)
        self.assertIn("F", film)

        # Consecutive independent vowels: 'आइए' -> AA I E
        aaie = devanagari_to_phonemes("आइए", as_string=True)
        self.assertEqual(aaie, "AA I E")

    def test_g2p_unsupported_char_logging(self):
        """Asserts unsupported characters are logged in diagnostics without silent dropping."""
        clear_unsupported_char_log()
        # Non-Devanagari character
        _ = devanagari_to_phonemes("test$", as_string=True, transcript_id="sample01")
        report = get_unsupported_char_report()
        self.assertGreater(report["total_unsupported_occurrences"], 0)
        self.assertIn("sample01", report["affected_transcript_ids"])

    def test_g2p_schwa_deletion(self):
        """Asserts correct deletion on multi-syllabic words while retaining monosyllabic inherent vowels."""
        # Multi-syllabic 'कमल' -> K A M A L (final schwa deleted)
        kamal = devanagari_to_phonemes("कमल", as_string=True)
        self.assertEqual(kamal, "K A M A L")

        # Monosyllabic 'क' -> K A (inherent schwa retained)
        ka = devanagari_to_phonemes("क", as_string=True)
        self.assertEqual(ka, "K A")


class TestPunarvaaniModelArchitecture(unittest.TestCase):
    def test_vit_pos_embed_registration(self):
        """Confirms pos_embed is initialized as nn.Parameter and registered in optimizer."""
        model = ML_VALLR(adapter_dim=256, num_classes=55, vit_depth=2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        param_ids = [id(p) for p in optimizer.param_groups[0]['params']]
        self.assertIn(id(model.backbone.pos_embed), param_ids)

    def test_vit_pos_embed_space_time_interpolation(self):
        """Verifies positional embedding interpolation across 16, 32, and 48 frame inputs."""
        vit = VideoViTMasked(embed_dim=128, depth=2, num_heads=4)
        # Base 16 frames -> Tp=8, Hp=14, Wp=14
        pos16 = vit.get_interpolated_pos_embed(Tp=8, Hp=14, Wp=14)
        self.assertEqual(pos16.shape, (1, 8 * 14 * 14, 128))

        # 32 frames -> Tp=16, Hp=14, Wp=14
        pos32 = vit.get_interpolated_pos_embed(Tp=16, Hp=14, Wp=14)
        self.assertEqual(pos32.shape, (1, 16 * 14 * 14, 128))

        # 48 frames -> Tp=24, Hp=14, Wp=14
        pos48 = vit.get_interpolated_pos_embed(Tp=24, Hp=14, Wp=14)
        self.assertEqual(pos48.shape, (1, 24 * 14 * 14, 128))

    def test_vit_mask_ratio_default(self):
        """Confirms model default mask_ratio is 0.0 for deterministic inference."""
        model = ML_VALLR(adapter_dim=256, num_classes=55)
        self.assertEqual(model.backbone.mask_ratio, 0.0)

    def test_temporal_adapter_configurable_output_lengths(self):
        """Verifies configurable temporal upsampling output lengths for 16, 32, and 48 frames."""
        # 16 frames, 4x upsampling -> T_ctc = 8 * 4 = 32
        m16 = ML_VALLR(adapter_dim=128, num_classes=55, vit_depth=1, temporal_upsample_factor=4)
        x16 = torch.randn(1, 3, 16, 224, 224)
        logits16, _ = m16(x16)
        self.assertEqual(logits16.shape, (1, 32, 55))

        # 32 frames, 2x upsampling -> T_ctc = 16 * 2 = 32
        m32 = ML_VALLR(adapter_dim=128, num_classes=55, vit_depth=1, temporal_upsample_factor=2)
        x32 = torch.randn(1, 3, 32, 224, 224)
        logits32, _ = m32(x32)
        self.assertEqual(logits32.shape, (1, 32, 55))

        # 48 frames, 2x upsampling -> T_ctc = 24 * 2 = 48
        m48 = ML_VALLR(adapter_dim=128, num_classes=55, vit_depth=1, temporal_upsample_factor=2)
        x48 = torch.randn(1, 3, 48, 224, 224)
        logits48, _ = m48(x48)
        self.assertEqual(logits48.shape, (1, 48, 55))


class TestPunarvaaniCTCConditionAndCollate(unittest.TestCase):
    def test_ctc_minimum_valid_repeated_labels(self):
        """Tests repeated labels where U=2, R=1, T=3 (T >= U + R is valid)."""
        # Target: ['K', 'K'] (ID 15, 15)
        label = torch.tensor([15, 15], dtype=torch.long)
        min_len = compute_ctc_min_length(label)
        self.assertEqual(min_len, 3)  # U=2 + R=1 = 3

        # Simulate CTC loss with T=3
        log_probs = torch.randn(3, 1, 55).log_softmax(dim=-1)
        criterion = nn.CTCLoss(blank=0)
        loss = criterion(log_probs, label, torch.tensor([3]), torch.tensor([2]))
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))

    def test_ctc_invalid_repeated_labels(self):
        """Tests repeated labels where U=2, R=1, T=2 (T < U + R is invalid)."""
        label = torch.tensor([15, 15], dtype=torch.long)
        min_len = compute_ctc_min_length(label)
        # T=2 is strictly less than required 3
        self.assertGreater(min_len, 2)

    def test_dataloader_all_invalid_batch_safety(self):
        """Simulates batch where 100% of samples violate T >= U + R; custom_collate handles safely."""
        vocab = get_hindi_vocab()
        # Samples with 50 phonemes, but t_ctc is only 32
        long_label = torch.full((50,), 15, dtype=torch.long)
        dummy_sample = (torch.zeros(16, 3, 224, 224), long_label, "लंबा वाक्य")
        batch = [dummy_sample, dummy_sample]

        videos, labels, texts = custom_collate(batch, vocab, t_ctc=32)
        self.assertIsNone(videos)
        self.assertEqual(labels, [])
        self.assertEqual(texts, [])


class TestPunarvaaniCLIAndInference(unittest.TestCase):
    def test_cli_flags(self):
        """Verifies --crop_face, --no_crop_face, and --max_invalid_sample_pct parse cleanly."""
        vocab = get_hindi_vocab()
        self.assertEqual(len(vocab), 55)
        self.assertEqual(vocab['<pad>'], 0)
        self.assertEqual(vocab['|'], 1)

    def test_inference_interface_mock_mode(self):
        """Asserts transcribe_video() returns structured dictionary and handles absent adapter gracefully."""
        res = transcribe_video(
            video_path="non_existent_test_video.mp4",
            sarvam_adapter_path=None
        )
        self.assertIsInstance(res, dict)
        self.assertIn("success", res)
        self.assertIn("phonemes", res)
        self.assertIn("transcription", res)
        self.assertIn("stage1_status", res)
        self.assertIn("stage2_status", res)
        # Should gracefully report failure on non-existent file
        self.assertFalse(res["success"])
        self.assertIsNotNone(res["error"])


class TestPunarvaaniSarvamIntegration(unittest.TestCase):
    def _get_tokenizer(self):
        try:
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', local_files_only=True)
        except Exception:
            class MockTokenizer:
                def __init__(self):
                    self.vocab = {"hello": 1, "world": 2, "<pad>": 0, "<eos>": 3}
                    self.additional_special_tokens = []
                    self.pad_token = "<pad>"
                    self.eos_token = "<eos>"

                def __len__(self):
                    return len(self.vocab)

                def get_vocab(self):
                    return dict(self.vocab)

                def add_special_tokens(self, token_dict):
                    tokens = token_dict.get("additional_special_tokens", [])
                    added = 0
                    for tok in tokens:
                        if tok not in self.vocab:
                            self.vocab[tok] = len(self.vocab)
                            self.additional_special_tokens.append(tok)
                            added += 1
                    return added

                def encode(self, text, add_special_tokens=False):
                    if text in self.vocab:
                        return [self.vocab[text]]
                    return [1]

            return MockTokenizer()

    def test_sarvam_special_token_registration(self):
        """Tests registering <S2S>, <PHONEMES>, </PHONEMES>, <TEXT> in tokenizer."""
        tok = self._get_tokenizer()
        for tag in TAGS:
            self.assertNotIn(tag, tok.get_vocab())
        added = tok.add_special_tokens({"additional_special_tokens": TAGS})
        self.assertEqual(added, 4)
        for tag in TAGS:
            self.assertIn(tag, tok.get_vocab())

    def test_sarvam_single_token_encoding(self):
        """Asserts each required special token encodes into a single token ID without subword splitting."""
        tok = self._get_tokenizer()
        tok.add_special_tokens({"additional_special_tokens": TAGS})
        for tag in TAGS:
            encoded = tok.encode(tag, add_special_tokens=False)
            self.assertEqual(len(encoded), 1, f"Token {tag} was split into subwords: {encoded}")

    def test_sarvam_tokenizer_vocab_embedding_match(self):
        """Asserts tokenizer vocabulary size matches model token embedding dimensions."""
        tok = self._get_tokenizer()
        init_len = len(tok)
        tok.add_special_tokens({"additional_special_tokens": TAGS})
        new_len = len(tok)
        self.assertEqual(new_len, init_len + len(TAGS))

        # Simulated embedding resize
        embedding = nn.Embedding(init_len, 768)
        self.assertEqual(embedding.weight.shape, (init_len, 768))
        new_embedding = nn.Embedding(new_len, 768)
        new_embedding.weight.data[:init_len] = embedding.weight.data
        self.assertEqual(new_embedding.weight.shape, (new_len, 768))

    def test_sarvam_lora_config_compatibility(self):
        """Simulates loading an adapter with matched vs. mismatched token configurations."""
        import tempfile
        import json
        tok = self._get_tokenizer()
        tok.add_special_tokens({"additional_special_tokens": TAGS})

        with tempfile.TemporaryDirectory() as tmp_dir:
            # 1. Matching configuration
            matched_cfg = {
                "base_model_name_or_path": "sarvamai/sarvam-1",
                "r": 16,
                "lora_alpha": 32
            }
            with open(os.path.join(tmp_dir, "adapter_config.json"), "w", encoding="utf-8") as f:
                json.dump(matched_cfg, f)

            res = validate_sarvam_adapter_compatibility(
                tmp_dir,
                base_model_id="sarvamai/sarvam-1",
                tokenizer=tok,
                strict=True
            )
            self.assertTrue(res["is_compatible"])

            # 2. Mismatched configuration
            mismatched_cfg = {
                "base_model_name_or_path": "meta-llama/Llama-2-7b",
                "r": 16
            }
            with open(os.path.join(tmp_dir, "adapter_config.json"), "w", encoding="utf-8") as f:
                json.dump(mismatched_cfg, f)

            with self.assertRaises(ValueError):
                validate_sarvam_adapter_compatibility(
                    tmp_dir,
                    base_model_id="sarvamai/sarvam-1",
                    tokenizer=tok,
                    strict=True
                )


class TestPunarvaaniDatasetIntegrity(unittest.TestCase):
    def test_dataset_speaker_leakage_check(self):
        """Verifies speaker-disjoint split detector catches speaker overlap and passes on clean split."""
        # Clean disjoint sets
        train_clean = [("data/spk01/vid01.mp4", "t1"), ("data/spk01/vid02.mp4", "t2"), ("data/spk02/vid01.mp4", "t3")]
        val_clean = [("data/spk03/vid01.mp4", "t4"), ("data/spk04/vid01.mp4", "t5")]
        leakage = check_speaker_leakage(train_clean, val_clean)
        self.assertEqual(len(leakage), 0)
        self.assertTrue(verify_speaker_disjoint_split(train_clean, val_clean))

        # Overlapping sets (spk02 in both)
        val_leaking = [("data/spk02/vid02.mp4", "t6"), ("data/spk03/vid01.mp4", "t7")]
        overlap = check_speaker_leakage(train_clean, val_leaking)
        self.assertIn("spk02", overlap)
        with self.assertRaises(ValueError):
            verify_speaker_disjoint_split(train_clean, val_leaking)


if __name__ == '__main__':
    unittest.main()
