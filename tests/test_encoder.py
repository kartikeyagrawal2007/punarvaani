import torch

from punarvaani.models.encoder.avhubert import (
    EncoderConfig,
    VisualSpeechEncoder,
    convert_fairseq_state_dict,
    load_fairseq_checkpoint,
)
from punarvaani.models.encoder.lora import (
    inject_lora,
    load_adapter,
    mark_trainable,
    merge_adapter,
    save_adapter,
    set_active_adapters,
)
from punarvaani.utils import lengths_to_padding_mask


def tiny_encoder(**kw):
    torch.manual_seed(0)
    return VisualSpeechEncoder(EncoderConfig.preset("tiny", **kw)).eval()


def test_encoder_shapes_and_padding_invariance():
    enc = tiny_encoder()
    video = torch.randn(2, 12, 44, 44)
    mask = lengths_to_padding_mask(torch.tensor([12, 8]))
    out, hidden = enc(video, mask, return_all_layers=True)
    assert out.shape == (2, 12, 64) and len(hidden) == 2
    # The padded sample must give the same features as running it alone, whatever
    # garbage sits in its padded frames.
    alone = enc(video[1:, :8])
    assert torch.allclose(out[1, :8], alone[0], atol=1e-4)


def test_large_preset_parameter_count_matches_avhubert_large():
    cfg = EncoderConfig.preset("large")
    with torch.device("meta"):
        enc = VisualSpeechEncoder(cfg)
    n = sum(p.numel() for p in enc.parameters())
    assert 320e6 < n < 340e6  # AV-HuBERT Large is ~325M including the ResNet front-end


def test_lora_starts_as_identity_and_merges_exactly():
    enc = tiny_encoder()
    x = torch.randn(1, 10, 44, 44)
    before = enc(x)
    n = inject_lora(enc, "hindi", r=4, alpha=8)
    assert n == 2 * 5  # 2 layers x (q, v, out, fc1, fc2)
    assert torch.allclose(enc(x), before, atol=1e-6)  # B is zero-initialised

    for name, p in enc.named_parameters():
        if "lora_B" in name:
            p.data.normal_(0, 0.1)
    adapted = enc(x)
    assert not torch.allclose(adapted, before, atol=1e-4)
    merge_adapter(enc, "hindi")
    assert torch.allclose(enc(x), adapted, atol=1e-5)


def test_stacked_adapters_trainability_and_roundtrip(tmp_path):
    enc = tiny_encoder()
    inject_lora(enc, "hindi", r=4, alpha=8)
    inject_lora(enc, "user_a", r=2, alpha=4, targets=("q_proj", "v_proj"), layers=[1])
    trainable = mark_trainable(enc, ["user_a"])
    assert trainable == 2 * (2 * 64 + 64 * 2)  # q and v of layer 1 only
    for name, p in enc.named_parameters():
        if "user_a" in name:
            p.data.normal_(0, 0.1)
    x = torch.randn(1, 10, 44, 44)
    ref = enc(x)
    save_adapter(enc, "user_a", tmp_path / "user_a.pt", meta={"user": "a"})

    set_active_adapters(enc, ["hindi"])
    assert not torch.allclose(enc(x), ref, atol=1e-4)

    fresh = tiny_encoder()
    inject_lora(fresh, "hindi", r=4, alpha=8)
    fresh.load_state_dict({k: v for k, v in enc.state_dict().items() if "user_a" not in k})
    assert load_adapter(fresh, tmp_path / "user_a.pt") == "user_a"
    assert torch.allclose(fresh(x), ref, atol=1e-5)


def test_fairseq_checkpoint_conversion(tmp_path):
    enc = tiny_encoder()
    ours = enc.state_dict()
    fairseq = {}
    for k, v in ours.items():
        k2 = k.replace("pos_conv.conv.parametrizations.weight.original0", "pos_conv.0.weight_g")
        k2 = k2.replace("pos_conv.conv.parametrizations.weight.original1", "pos_conv.0.weight_v")
        k2 = k2.replace("pos_conv.conv.bias", "pos_conv.0.bias")
        fairseq["w2v_encoder.w2v_model." + k2] = v + 1.0 if v.is_floating_point() else v
    fairseq["w2v_encoder.w2v_model.final_proj.weight"] = torch.zeros(3, 3)
    assert set(convert_fairseq_state_dict(fairseq)) == set(ours)

    torch.save({"model": fairseq}, tmp_path / "fake_avhubert.pt")
    report = load_fairseq_checkpoint(enc, tmp_path / "fake_avhubert.pt")
    assert report["missing"] == [] and report["unexpected"] == []
    assert torch.allclose(enc.post_extract_proj.weight, ours["post_extract_proj.weight"])  # +1 applied


def test_merge_and_unload_restores_plain_names():
    from punarvaani.models.encoder.lora import merge_and_unload

    enc = tiny_encoder()
    plain_keys = set(enc.state_dict())
    inject_lora(enc, "hindi", r=4, alpha=8)
    for name, p in enc.named_parameters():
        if "lora_B" in name:
            p.data.normal_(0, 0.1)
    x = torch.randn(1, 10, 44, 44)
    ref = enc(x)
    merge_and_unload(enc)
    assert set(enc.state_dict()) == plain_keys
    assert torch.allclose(enc(x), ref, atol=1e-5)
