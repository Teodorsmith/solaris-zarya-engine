# Copyright (C) 2026 Teodor Smith
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# For commercial licensing options without AGPLv3 network-copyleft obligations,
# contact: teosmith.studios@gmail.com

import unittest

from agent.engine.verifier import SRTVerifier
from agent.models import SRTTrace


class TestSRTVerifier(unittest.TestCase):
    def setUp(self):
        self.verifier = SRTVerifier()

    def test_modus_ponens_valid(self):
        srt = SRTTrace(
            conclusion="it is raining",
            premises=["if clouds then it is raining", "clouds"],
            inference_rule="modus_ponens",
            confidence=1.0,
        )
        res = self.verifier.verify(srt)
        self.assertTrue(res.verified, res.reason)

    def test_modus_tollens_valid(self):
        srt = SRTTrace(
            conclusion="not clouds",
            premises=["if clouds then rain", "not rain"],
            inference_rule="modus_tollens",
            confidence=1.0,
        )
        res = self.verifier.verify(srt)
        self.assertTrue(res.verified, res.reason)

    def test_transitive_valid(self):
        srt = SRTTrace(
            conclusion="A implies C",
            premises=["A implies B", "B implies C"],
            inference_rule="transitive_implication",
            confidence=1.0,
        )
        res = self.verifier.verify(srt)
        self.assertTrue(res.verified, res.reason)

    def test_invalid_rule(self):
        srt = SRTTrace(
            conclusion="C",
            premises=["A implies B", "B"],
            inference_rule="modus_ponens",
            confidence=1.0,
        )
        res = self.verifier.verify(srt)
        self.assertFalse(res.verified)

    def test_correct_answer_wrong_srt(self):
        # Conclusion is independent of premises
        srt = SRTTrace(
            conclusion="C",
            premises=["A implies B", "A"],
            inference_rule="modus_ponens",
            confidence=1.0,
        )
        res = self.verifier.verify(srt)
        self.assertFalse(res.verified)

    def test_unsupported_quantifier(self):
        srt = SRTTrace(
            conclusion="some apples are red",
            premises=["for all apples, they are red"],
            inference_rule="modus_ponens",
            confidence=1.0,
        )
        res = self.verifier.verify(srt)
        self.assertFalse(res.verified)
        self.assertIn("quantifier not supported", res.reason)


if __name__ == "__main__":
    unittest.main()
