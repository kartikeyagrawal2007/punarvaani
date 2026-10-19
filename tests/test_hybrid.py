import torch

from punarvaani.models.hybrid.guidance import UnitASR, guide_units


def _train_unit_asr(num_units=8, vocab=8, steps=300):
    """Toy world: unit k (held for 4 frames at 50 Hz) is character k; unit 0 is silence."""
    torch.manual_seed(0)
    asr = UnitASR(num_units, vocab, dim=32, layers=2)
    opt = torch.optim.Adam(asr.parameters(), 3e-3)
    for _ in range(steps):
        seqs = [torch.randint(1, num_units, (3,)) for _ in range(16)]
        units = torch.stack([torch.cat([torch.zeros(4, dtype=torch.long), s.repeat_interleave(4), torch.zeros(4, dtype=torch.long)])
                             for s in seqs])  # fmt: skip
        tokens = torch.stack(seqs)
        loss = asr.loss(units, torch.full((16,), units.shape[1]), tokens, torch.full((16,), 3), num_units)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return asr


def test_guidance_pulls_units_toward_the_text():
    asr = _train_unit_asr()
    target = torch.tensor([3, 5, 2])
    # Path B guessed [3, 6, 2]: the middle "word" is wrong (a look-alike).
    guess = torch.cat(
        [
            torch.zeros(4, dtype=torch.long),
            torch.tensor([3, 6, 2]).repeat_interleave(4),
            torch.zeros(4, dtype=torch.long),
        ]
    )
    logits = torch.nn.functional.one_hot(guess, 8).float() * 3.0
    res = guide_units(logits, target, asr)
    assert res.ctc_after < res.ctc_before
    assert (res.units[8:12] == 5).float().mean() >= 0.5  # middle segment fixed
    assert (res.units[:4] == 0).all()  # leading silence (timing) preserved
    assert res.changed_fraction < 0.5
