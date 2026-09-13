"""Custom formatted test runner for Enterprise GenAI Security Platform.

Provides clear, categorized, and unambiguous security verification output.
Usage (from backend/):
    python tests/run_tests.py
"""

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

    def __init__(self, stream=None, descriptions=None, verbosity=None):
        super().__init__(stream, descriptions, verbosity)
        self.stream = stream or sys.stdout
        self.current_class = None
        self.start_time = 0
        self.test_count = 0
        self.pass_count = 0

        self.category_names = {
            "TestUnicodeNormalization": "Unicode NFKC Normalization & Compatibility",
            "TestConfusableHomoglyphs": "Cross-Script Confusable Homoglyphs (Cyrillic/Greek)",
            "TestInvisibleCharactersAndSmuggling": "Invisible Characters & Unicode Tag ASCII Smuggling",
            "TestWhitespaceAndSyntaxPreservation": "Syntax & Indentation Preservation (Code Protection)",
            "TestAdversarialSuffixStripping": "Adversarial Suffix & GCG Attack Stripper",
            "TestFullSanitizerPipeline": "End-to-End Sanitizer Pipeline (Compound Attacks)",
        }

    def startTestRun(self):
        self.start_time = time.perf_counter()
        self.stream.write("\n" + "=" * 80 + "\n")
        self.stream.write("  ENTERPRISE GENAI SECURITY PLATFORM — SECURITY TEST RUNNER\n")
        self.stream.write("  Module 1: Input Pre-processing & Sanitizer Engine\n")
        self.stream.write("=" * 80 + "\n\n")

    def startTest(self, test):
        super().startTest(test)
        self.test_count += 1
        class_name = test.__class__.__name__

        if class_name != self.current_class:
            self.current_class = class_name
            category_title = self.category_names.get(class_name, class_name)
            self.stream.write(f"▸ {category_title}:\n")

    def addSuccess(self, test):
        super().addSuccess(test)
        self.pass_count += 1
        description = self._get_test_description(test)
        self.stream.write(f"   [PASS] {description}\n")

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
        self.stream.write("\n" + "-" * 80 + "\n")
        self.stream.write("  VERIFICATION SUMMARY\n")
        self.stream.write("-" * 80 + "\n")
        self.stream.write(f"  Total Tests Run:  {self.test_count}\n")
        self.stream.write(f"  Passed:           {self.pass_count}\n")
        self.stream.write(f"  Failures:         {len(self.failures)}\n")
        self.stream.write(f"  Errors:           {len(self.errors)}\n")
        self.stream.write(f"  Duration:         {elapsed:.4f}s\n")

        if self.wasSuccessful():
            self.stream.write("  Status:           ALL TESTS PASSED (100% SUCCESS)\n")
        else:
            self.stream.write("  Status:           VERIFICATION FAILED\n")
            if self.failures:
                self.stream.write("\n" + "!" * 80 + "\n")
                self.stream.write("  FAILURE DETAILS:\n")
                self.stream.write("!" * 80 + "\n")
                for test, err in self.failures:
                    self.stream.write(f"\n[FAIL] {test.id()}\n{err}\n")
            if self.errors:
                self.stream.write("\n" + "!" * 80 + "\n")
                self.stream.write("  ERROR DETAILS:\n")
                self.stream.write("!" * 80 + "\n")
                for test, err in self.errors:
                    self.stream.write(f"\n[ERROR] {test.id()}\n{err}\n")

        self.stream.write("=" * 80 + "\n\n")


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.discover(tests_dir, pattern="test_*.py")
    result = SecurityTestResult()
    result.startTestRun()
    suite.run(result)
    result.print_summary()
    sys.exit(0 if result.wasSuccessful() else 1)
