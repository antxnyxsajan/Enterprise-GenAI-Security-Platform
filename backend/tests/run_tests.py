"""Custom formatted test runner for Enterprise GenAI Security Platform.

Provides clear, categorized, and unambiguous security verification output.
Usage (from backend/):
    python tests/run_tests.py                 # Fast offline unit tests (default)
    python tests/run_tests.py --integration   # Real ML models & Qdrant integration tests
    python tests/run_tests.py --all           # Unit + Integration tests
"""

import argparse
import os
import sys
import time
import unittest

# Ensure backend root is in sys.path
tests_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.abspath(os.path.join(tests_dir, ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)


class SecurityTestResult(unittest.TestResult):
    """Custom TestResult formatter that groups tests by category and prints clear descriptions."""

    def __init__(self, suite_label: str = "Unit Test Suite", stream=None):
        super().__init__()
        self.suite_label = suite_label
        self.stream = stream or sys.stdout
        self.current_class = None
        self.start_time = 0.0
        self.test_count = 0
        self.pass_count = 0

        self.category_names = {
            # Module 1: Sanitizer
            "TestUnicodeNormalization": "[Module 1] Unicode NFKC Normalization & Compatibility",
            "TestConfusableHomoglyphs": "[Module 1] Cross-Script Confusable Homoglyphs (Cyrillic/Greek)",
            "TestInvisibleCharactersAndSmuggling": "[Module 1] Invisible Characters & Unicode Tag ASCII Smuggling",
            "TestWhitespaceAndSyntaxPreservation": "[Module 1] Syntax & Indentation Preservation (Code Protection)",
            "TestAdversarialSuffixStripping": "[Module 1] Adversarial Suffix & GCG Attack Stripper",
            "TestFullSanitizerPipeline": "[Module 1] End-to-End Sanitizer Pipeline (Compound Attacks)",
            # Module 2: Hybrid Prompt Firewall
            "TestDynamicLabelResolution": "[Module 2] Dynamic id2label Introspection (No Score Inversion)",
            "TestFirewallLazyLoading": "[Module 2] Lazy Model Loading & Device Resolution",
            "TestSeedSignatureLoader": "[Module 2] Attack Signature Seed Loader (JSONL / JBB-Behaviors)",
            "TestQdrantInMemoryStore": "[Module 2] Qdrant In-Memory Vector Store & Idempotent Seeding",
            "TestHybridFirewallOrchestration": "[Module 2] Parallel Non-Gated Firewall & Complementarity",
            "TestHybridFirewallIntegration": "[Module 2 Integration] Real DeBERTa-v3 + MiniLM + Qdrant",
            # Module 3: Data Protection Engine (DPE)
            "TestDPEContracts": "[Module 3] DPE Schema & Redaction Token Contracts",
            "TestDPEServiceOrchestration": "[Module 3] PII Detection & Async Redaction Engine",
            "TestPresidioEngineIntegration": "[Module 3 Integration] Real Presidio Analyzer & Anonymizer",
        }

    def startTestRun(self):
        self.start_time = time.perf_counter()
        self.stream.write("\n" + "=" * 84 + "\n")
        self.stream.write("  ENTERPRISE GENAI SECURITY PLATFORM — SECURITY TEST RUNNER\n")
        self.stream.write(f"  Suite: {self.suite_label}\n")
        self.stream.write("=" * 84 + "\n\n")

    def startTest(self, test):
        super().startTest(test)
        self.test_count += 1
        class_name = test.__class__.__name__

        if class_name != self.current_class:
            self.current_class = class_name
            category_title = self.category_names.get(class_name, class_name)
            self.stream.write(f"\n▸ {category_title}:\n")

    def addSuccess(self, test):
        super().addSuccess(test)
        self.pass_count += 1
        description = self._get_test_description(test)
        self.stream.write(f"   [PASS] {description}\n")

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        description = self._get_test_description(test)
        self.stream.write(f"   [SKIP] {description} ({reason})\n")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        description = self._get_test_description(test)
        self.stream.write(f"   [FAIL] {description}\n")

    def addError(self, test, err):
        super().addError(test, err)
        description = self._get_test_description(test)
        self.stream.write(f"   [ERROR] {description}\n")

    def _get_test_description(self, test):
        doc = getattr(test, "_testMethodDoc", None)
        if doc and doc.strip():
            return doc.strip().split("\n")[0]
        name = test._testMethodName
        if name.startswith("test_"):
            name = name[5:]
        return name.replace("_", " ").capitalize()

    def print_summary(self):
        elapsed = time.perf_counter() - self.start_time
        self.stream.write("\n" + "-" * 84 + "\n")
        self.stream.write("  VERIFICATION SUMMARY\n")
        self.stream.write("-" * 84 + "\n")
        self.stream.write(f"  Total Tests Run:  {self.test_count}\n")
        self.stream.write(f"  Passed:           {self.pass_count}\n")
        self.stream.write(f"  Skipped:          {len(self.skipped)}\n")
        self.stream.write(f"  Failures:         {len(self.failures)}\n")
        self.stream.write(f"  Errors:           {len(self.errors)}\n")
        self.stream.write(f"  Duration:         {elapsed:.4f}s\n")

        if self.wasSuccessful():
            status = (
                "ALL EXECUTED TESTS PASSED (WITH SKIPS)"
                if self.skipped
                else "ALL TESTS PASSED (100% SUCCESS)"
            )
            self.stream.write(f"  Status:           {status}\n")
        else:
            self.stream.write("  Status:           VERIFICATION FAILED\n")
            if self.failures:
                self.stream.write("\n" + "!" * 84 + "\n")
                self.stream.write("  FAILURE DETAILS:\n")
                self.stream.write("!" * 84 + "\n")
                for test, err in self.failures:
                    self.stream.write(f"\n[FAIL] {test.id()}\n{err}\n")
            if self.errors:
                self.stream.write("\n" + "!" * 84 + "\n")
                self.stream.write("  ERROR DETAILS:\n")
                self.stream.write("!" * 84 + "\n")
                for test, err in self.errors:
                    self.stream.write(f"\n[ERROR] {test.id()}\n{err}\n")

        self.stream.write("=" * 84 + "\n\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run categorized security tests for Enterprise GenAI Security Platform."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--unit",
        action="store_true",
        help="Run fast offline unit tests only (default).",
    )
    group.add_argument(
        "--integration",
        action="store_true",
        help="Run real-model integration tests (loads DeBERTa-v3, MiniLM, Qdrant).",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Run both unit and integration tests.",
    )
    args = parser.parse_args()

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    if args.integration:
        label = "Integration Tests (Real ML Models + Qdrant)"
        suite.addTests(
            loader.discover(os.path.join(tests_dir, "integration"), pattern="test_*.py")
        )
    elif args.all:
        label = "Full Verification (Unit + Integration Tests)"
        suite.addTests(loader.discover(os.path.join(tests_dir, "unit"), pattern="test_*.py"))
        suite.addTests(
            loader.discover(os.path.join(tests_dir, "integration"), pattern="test_*.py")
        )
    else:
        label = "Unit Tests (Modules 1 & 2 — Fast Offline)"
        suite.addTests(loader.discover(os.path.join(tests_dir, "unit"), pattern="test_*.py"))

    result = SecurityTestResult(suite_label=label)
    result.startTestRun()
    suite.run(result)
    result.print_summary()
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
