import json
from types import SimpleNamespace

from punarvaani.models.path_a.llm_correction import (
    ClaudeCorrector,
    CorrectionContext,
    LexiconCorrector,
    build_messages,
)

CORPUS = [
    "मुझे पानी चाहिए",
    "मुझे पानी दो",
    "पानी ठंडा है",
    "मुझे दवाई चाहिए",
    "बाल कटवाने हैं",
    "दर्द हो रहा है",
]


def test_lexicon_corrector_fixes_viseme_confusions_with_context():
    corr = LexiconCorrector(CORPUS)
    # बानी / भुझे look exactly like पानी / मुझे on the lips
    out = corr.correct([("भुझे बानी चाहिए", -3.0)], CorrectionContext())
    assert out.text == "मुझे पानी चाहिए" and out.source == "lexicon"


def test_lexicon_keeps_known_words():
    corr = LexiconCorrector(CORPUS)
    assert corr.correct_sentence("दर्द हो रहा है")[0] == "दर्द हो रहा है"


def test_prompt_contains_confusion_groups_context_and_nbest():
    ctx = CorrectionContext(setting="अस्पताल", phrases=["मुझे पानी चाहिए"], recent_turns=["नर्स: कैसा लग रहा है?"])
    system, user = build_messages([("भुझे बानी चाहिए", -3.2), ("मुझे बानी चाहिए", -3.5)], ctx)
    assert "प फ ब भ म" in system and "अस्पताल" in system and "मुझे पानी चाहिए" in system
    assert "0. भुझे बानी चाहिए" in user and "नर्स" in user


class _FakeMessages:
    def __init__(self, payload=None, error=None, stop_reason="end_turn"):
        self.payload, self.error, self.stop_reason, self.calls = payload, error, stop_reason, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        block = SimpleNamespace(type="text", text=json.dumps(self.payload, ensure_ascii=False))
        return SimpleNamespace(content=[block], stop_reason=self.stop_reason)


def _client(messages):
    return SimpleNamespace(beta=SimpleNamespace(messages=messages))


def test_claude_corrector_parses_structured_output():
    msgs = _FakeMessages({"text": "मुझे पानी चाहिए", "chosen_index": 1, "confidence": 0.9})
    corr = ClaudeCorrector(client=_client(msgs))
    out = corr.correct([("भुझे बानी चाहिए", -3.0), ("मुझे बानी चाहिए", -3.1)], CorrectionContext())
    assert (out.text, out.source, out.chosen_index) == ("मुझे पानी चाहिए", "llm", 1)
    call = msgs.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["output_config"]["format"]["type"] == "json_schema"


def test_claude_corrector_falls_back_offline_on_error_and_refusal():
    lexicon = LexiconCorrector(CORPUS)
    for msgs in (
        _FakeMessages(error=ConnectionError("no network")),
        _FakeMessages({"text": "x"}, stop_reason="refusal"),
    ):
        out = ClaudeCorrector(client=_client(msgs), fallback=lexicon).correct(
            [("भुझे बानी चाहिए", -3.0)], CorrectionContext()
        )
        assert out.source == "lexicon" and out.text == "मुझे पानी चाहिए"
