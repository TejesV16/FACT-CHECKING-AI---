import unittest

from ui_config import history_prompt, verdict_tone


class VerdictToneTests(unittest.TestCase):
    def test_known_verdict_has_readable_tone(self):
        tone = verdict_tone("True")
        self.assertEqual(tone["label"], "Supported")
        self.assertEqual(tone["accent"], "#0F766E")

    def test_unknown_verdict_uses_review_tone(self):
        tone = verdict_tone("unexpected")
        self.assertEqual(tone["label"], "Needs review")

    def test_history_prompt_returns_a_working_example(self):
        self.assertIn("Earth", history_prompt("Science"))


if __name__ == "__main__":
    unittest.main()
