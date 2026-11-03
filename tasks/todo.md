# Pre-dataset hardening — execution checklist

## Phase 0 — Harness
- [x] T0.1 Baseline 18 tests recorded
- [x] T0.2 Synthetic dataset module + script
- [x] T0.3 Metrics module

## Phase 1 — Low-risk fixes
- [x] T1.1 B5 G2P schwa deletion + golden tests
- [x] T1.2 B2 evaluation `re` + metrics
- [x] T1.3 B8 fp32 CTC under autocast
- [x] T1.4 B6 max-invalid enforcement + warmup scheduler
- [x] T1.5 B3 README/API kwarg

## Phase 2 — Data ingestion
- [x] T2.1 B4 true mouth ROI + geometry tests
- [x] T2.2 B1 preprocessing script
- [x] T2.3 B7 speaker-disjoint dataset
- [x] T2.4 Variable-length/all-frames `VSRItem`
- [x] T2.5 Preflight validator

## Phase 3 — Variable-length model
- [x] T3.1 Collate with lengths/masks
- [x] T3.2 ViT padding masks + spatial token pooling
- [x] T3.3 `VALLR.forward(video_lengths)`
- [x] T3.4 Train/eval/infer wiring

## Phase 4 — Stage 2 readiness
- [x] T4.1 Enforce adapter validation
- [x] T4.2 Adapter preflight
- [x] T4.3 Mock-mode eval path

## Phase 5 — Verification
- [x] T5.1 Synthetic end-to-end smoke
- [x] T5.2 Full test suite
- [x] T5.3 Docs reconciliation

## Definition of done
- [x] 18 pre-existing tests pass
- [x] Every bug has a regression test
- [x] Preflight clean on synthetic, actionable on broken
- [x] 1 train step + eval + infer on CPU
- [x] 2–3 s / 35–45 phoneme clip accepted
