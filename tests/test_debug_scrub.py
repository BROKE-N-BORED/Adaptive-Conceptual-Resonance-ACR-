"""Debug log layer tests: secrets never reach the report."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from debug_log import DebugLogger  # noqa: E402


class ScrubTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.log = DebugLogger("acr", "0.7",
                               log_dir=self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_fake_key_and_token_scrubbed(self):
        fake_key = "sk-fake-key-abcdefghijklmnop123456"
        fake_token = "ghp_faketoken1234567890abcdefghij"
        self.log.set_setting("api_key", fake_key)
        self.log.set_setting("access_token", fake_token)
        self.log.info("saw token %s in a line", fake_token)
        self.log.info("key was %s", fake_key)
        path = self.log.generate_report()
        with open(path, encoding="utf-8") as fh:
            report = fh.read()
        self.assertNotIn(fake_key, report)
        self.assertNotIn(fake_token, report)
        self.assertIn("[REDACTED]", report)

    def test_auth_header_scrubbed(self):
        self.log.info("Authorization: Bearer secret-bearer-value-99")
        path = self.log.generate_report()
        with open(path, encoding="utf-8") as fh:
            report = fh.read()
        self.assertNotIn("secret-bearer-value-99", report)

    def test_traceback_captured(self):
        try:
            raise ValueError("boom for the report")
        except ValueError:
            self.log.exception("caught a test error")
        report = self.log.build_report()
        self.assertIn("Traceback", report)
        self.assertIn("boom for the report", report)

    def test_two_reports_do_not_clobber(self):
        first = self.log.generate_report()
        second = self.log.generate_report()
        self.assertNotEqual(first, second)
        self.assertTrue(os.path.exists(first))
        self.assertTrue(os.path.exists(second))

    def test_report_has_standard_sections(self):
        report = self.log.build_report()
        for section in ("== Context ==", "== Settings ==",
                        "== Services ==", "== Recent actions",
                        "== Errors ==", "== Log tail"):
            self.assertIn(section, report)


if __name__ == "__main__":
    unittest.main()
