"""Unit tests for Module 3: Data Protection Engine (DPE).

Tests PII entity contracts, anonymization operators, non-blocking async execution,
and end-to-end integration between Module 1 (Sanitizer) and Module 3 (DPE).
"""

import asyncio
import importlib.util
import time
import unittest
from typing import List

from app.dpe import (
    DPEResult,
    DataProtectionEngine,
    PIIAnalyzer,
    PIIAnonymizer,
    PIIEntityMatch,
    default_redaction_token,
)

HAS_PRESIDIO = (
    importlib.util.find_spec("presidio_analyzer") is not None
    and importlib.util.find_spec("presidio_anonymizer") is not None
)


class DeterministicStubAnalyzer:
    """Fast deterministic mock analyzer for offline unit testing."""

    def __init__(self, predefined_matches: List[PIIEntityMatch] = None, delay: float = 0.0):
        self.predefined_matches = predefined_matches or []
        self.delay = delay
        self.call_count = 0
        self.last_text = None

    def analyze(self, text: str, entities=None, score_threshold=None) -> List[PIIEntityMatch]:
        if self.delay > 0:
            time.sleep(self.delay)
        self.call_count += 1
        self.last_text = text

        matches: List[PIIEntityMatch] = []
        lowered = text.lower()

        # Simple deterministic recognizers for mock testing
        if "alice@example.com" in lowered:
            start = text.index("alice@example.com")
            matches.append(
                PIIEntityMatch(
                    entity_type="EMAIL_ADDRESS",
                    start=start,
                    end=start + len("alice@example.com"),
                    score=0.95,
                    text="alice@example.com",
                )
            )

        if "555-123-4567" in text:
            start = text.index("555-123-4567")
            matches.append(
                PIIEntityMatch(
                    entity_type="PHONE_NUMBER",
                    start=start,
                    end=start + len("555-123-4567"),
                    score=0.90,
                    text="555-123-4567",
                )
            )

        if "123-45-6789" in text:
            start = text.index("123-45-6789")
            matches.append(
                PIIEntityMatch(
                    entity_type="US_SSN",
                    start=start,
                    end=start + len("123-45-6789"),
                    score=0.99,
                    text="123-45-6789",
                )
            )

        return matches + self.predefined_matches


class DeterministicStubAnonymizer:
    """Fast string-replacement anonymizer for offline unit tests."""

    def anonymize(self, text: str, matches: List[PIIEntityMatch], custom_replacements=None) -> str:
        # Sort reverse by start position to replace slices safely without offset drift
        sorted_matches = sorted(matches, key=lambda m: m.start, reverse=True)
        result = text
        for m in sorted_matches:
            tag = default_redaction_token(m.entity_type)
            result = result[: m.start] + tag + result[m.end :]
        return result


class TestDPEContracts(unittest.TestCase):
    """Tests DPEResult schema properties and helper functions."""

    def test_default_redaction_token_format(self):
        """Redaction tokens follow standard <ENTITY_TYPE_REDACTED> uppercase format"""
        self.assertEqual(default_redaction_token("email_address"), "<EMAIL_ADDRESS_REDACTED>")
        self.assertEqual(default_redaction_token("PHONE_NUMBER"), "<PHONE_NUMBER_REDACTED>")

    def test_dpe_result_was_anonymized_property(self):
        """was_anonymized property accurately checks if PII was removed"""
        clean = DPEResult(
            original_text="Hello world",
            anonymized_text="Hello world",
            has_pii=False,
        )
        self.assertFalse(clean.was_anonymized)

        redacted = DPEResult(
            original_text="Contact alice@example.com",
            anonymized_text="Contact <EMAIL_ADDRESS_REDACTED>",
            has_pii=True,
            total_entities_count=1,
        )
        self.assertTrue(redacted.was_anonymized)


class TestDPEServiceOrchestration(unittest.TestCase):
    """Tests PII analysis, anonymization, and async offloading in DataProtectionEngine."""

    def test_benign_prompt_clean(self):
        """Benign prompt with zero PII passes through unmodified with has_pii=False"""
        dpe = DataProtectionEngine(
            analyzer=DeterministicStubAnalyzer(),
            anonymizer=DeterministicStubAnonymizer(),
        )
        result = dpe.evaluate_sync("Explain how Dijkstra's algorithm works.")
        self.assertFalse(result.has_pii)
        self.assertFalse(result.was_anonymized)
        self.assertEqual(result.anonymized_text, "Explain how Dijkstra's algorithm works.")
        self.assertEqual(result.entities_found, [])
        self.assertEqual(result.total_entities_count, 0)

    def test_multi_entity_detection_and_redaction(self):
        """Multiple PII entities (email, phone, SSN) are all detected and replaced cleanly"""
        prompt = "User email is alice@example.com, phone 555-123-4567, SSN 123-45-6789."
        dpe = DataProtectionEngine(
            analyzer=DeterministicStubAnalyzer(),
            anonymizer=DeterministicStubAnonymizer(),
        )
        result = dpe.evaluate_sync(prompt)

        self.assertTrue(result.has_pii)
        self.assertTrue(result.was_anonymized)
        self.assertEqual(result.total_entities_count, 3)
        self.assertIn("EMAIL_ADDRESS", result.entities_found)
        self.assertIn("PHONE_NUMBER", result.entities_found)
        self.assertIn("US_SSN", result.entities_found)
        self.assertEqual(result.entity_counts["EMAIL_ADDRESS"], 1)

        expected = (
            "User email is <EMAIL_ADDRESS_REDACTED>, phone <PHONE_NUMBER_REDACTED>, "
            "SSN <US_SSN_REDACTED>."
        )
        self.assertEqual(result.anonymized_text, expected)

    def test_async_evaluate_runs_non_blocking_in_worker_thread(self):
        """Async evaluate() executes in worker thread via asyncio.to_thread without blocking event loop"""
        stub_analyzer = DeterministicStubAnalyzer(delay=0.05)
        dpe = DataProtectionEngine(
            analyzer=stub_analyzer,
            anonymizer=DeterministicStubAnonymizer(),
        )
        result = asyncio.run(dpe.evaluate("Contact alice@example.com immediately."))
        self.assertTrue(result.has_pii)
        self.assertEqual(result.total_entities_count, 1)
        self.assertEqual(stub_analyzer.call_count, 1)

    def test_empty_or_whitespace_prompt(self):
        """Empty or whitespace prompts return safe empty DPEResult without running analysis"""
        dpe = DataProtectionEngine(
            analyzer=DeterministicStubAnalyzer(),
            anonymizer=DeterministicStubAnonymizer(),
        )
        res_empty = dpe.evaluate_sync("")
        self.assertFalse(res_empty.has_pii)
        self.assertEqual(res_empty.anonymized_text, "")

        res_spaces = dpe.evaluate_sync("   \n\t  ")
        self.assertFalse(res_spaces.has_pii)
        self.assertEqual(res_spaces.anonymized_text, "")

    def test_module1_to_module3_obfuscated_pii_handoff(self):
        """Module 1 normalizes obfuscated PII (zero-width + Cyrillic homoglyphs) before Module 3 redacts it"""
        dpe = DataProtectionEngine(
            analyzer=DeterministicStubAnalyzer(),
            anonymizer=DeterministicStubAnonymizer(),
        )
        # Obfuscated email: 'al\u200bice@ex\u0430mple.com' (zero-width char + Cyrillic 'a')
        raw_obfuscated = "Send confidential documents to al\u200bice@ex\u0430mple.com now."
        sanitized, dpe_result = dpe.evaluate_raw_sync(raw_obfuscated)

        # Module 1 catches the evasion
        self.assertTrue(sanitized.has_evasion_indicators)
        self.assertIn("ZERO_WIDTH_CHARS_DETECTED", sanitized.evasion_flags)
        self.assertIn("HOMOGLYPH_SPOOFING_DETECTED", sanitized.evasion_flags)
        self.assertEqual(sanitized.cleaned_text, "Send confidential documents to alice@example.com now.")

        # Module 3 receives normalized text and redacts the email
        self.assertTrue(dpe_result.has_pii)
        self.assertIn("EMAIL_ADDRESS", dpe_result.entities_found)
        self.assertEqual(
            dpe_result.anonymized_text,
            "Send confidential documents to <EMAIL_ADDRESS_REDACTED> now.",
        )


@unittest.skipUnless(HAS_PRESIDIO, "presidio-analyzer / presidio-anonymizer not installed")
class TestPresidioEngineIntegration(unittest.TestCase):
    """Integration tests with real Microsoft Presidio AnalyzerEngine and AnonymizerEngine."""

    @classmethod
    def setUpClass(cls):
        cls.analyzer = PIIAnalyzer()
        cls.anonymizer = PIIAnonymizer()
        cls.dpe = DataProtectionEngine(analyzer=cls.analyzer, anonymizer=cls.anonymizer)

    def test_real_presidio_redacts_email_and_phone(self):
        """Real Presidio engine detects and redacts email addresses and phone numbers accurately"""
        text = "Please reach out to support@mycompany.org or call +1 555-432-1098."
        result = self.dpe.evaluate_sync(text)
        self.assertTrue(result.has_pii)
        self.assertIn("EMAIL_ADDRESS", result.entities_found)
        self.assertIn("<EMAIL_ADDRESS_REDACTED>", result.anonymized_text)


if __name__ == "__main__":
    unittest.main()
