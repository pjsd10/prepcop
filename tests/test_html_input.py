"""Local-only extraction checks; no provider or evaluator-answer inputs."""

import unittest
from pathlib import Path

from interview_prep.html_input import InputValidationError, parse_resume_html, validate_input_text


class HtmlInputTests(unittest.TestCase):
    def test_all_fictional_resume_reference_lines_are_preserved(self):
        root = Path(__file__).parent / "fixtures" / "fictional-resumes"
        cases = sorted(root.glob("*/resume.html"))
        self.assertEqual(len(cases), 5)
        for case in cases:
            with self.subTest(case=case.parent.name):
                extracted = parse_resume_html(case.read_bytes(), case.name).casefold()
                reference = case.with_name("resume-reference.txt").read_text(encoding="utf-8-sig")
                for line in reference.splitlines():
                    if line.strip():
                        self.assertIn(line.strip().casefold(), extracted)

    def test_inert_extraction_preserves_qualifiers_and_table_cells(self):
        markup = b'''<html><head><title>SECRET TITLE</title><style>SECRET STYLE</style></head>
        <body><h1>Candidate</h1><p>Python <strong>tutorial only</strong>; not used at work.</p>
        <table><tr><th>Skill</th><th>Exposure</th></tr><tr><td>SQL</td><td>Course project only</td></tr></table>
        <script src="https://example.invalid/script">SECRET SCRIPT</script>
        <iframe src="https://example.invalid/page">SECRET FRAME</iframe>
        <div hidden>SECRET HIDDEN</div><p style="display: none !important">SECRET DISPLAY</p>
        <span aria-hidden="true">SECRET ARIA</span><template>SECRET TEMPLATE</template>
        <img src="https://example.invalid/image" onerror="alert(1)"></body></html>'''
        result = parse_resume_html(markup, "candidate.HTML")
        self.assertIn("Python tutorial only; not used at work.", result)
        self.assertIn("Skill | Exposure", result)
        self.assertIn("SQL | Course project only", result)
        self.assertNotIn("SECRET", result)
        self.assertNotIn("https://", result)

    def test_utf8_bom_entities_and_inline_text(self):
        result = parse_resume_html(
            '\ufeff<p>Ren\u00e9 develops Python &amp; SQL projects; hands-on work only in a course.</p>'.encode(),
            "resume.htm",
        )
        self.assertIn("Ren\u00e9", result)
        self.assertIn("Python & SQL", result)

    def test_invalid_files_are_rejected(self):
        cases = [
            (b"", "resume.html", "empty"),
            (b"<p>Valid text but unsupported extension.</p>", "resume.pdf", "Unsupported file type"),
            (b"%PDF-1.4 renamed", "resume.html", "not static HTML"),
            (b"PK\x03\x04 renamed", "resume.html", "not static HTML"),
            (b"<p>Invalid encoding \xff</p>", "resume.html", "UTF-8"),
            (b"<p>Binary\x00</p>", "resume.html", "binary data"),
            (b"Plain text renamed to HTML with no markup or conversion at all.", "resume.html", "No supported HTML"),
            (b"<html><script>document.write('Resume generated dynamically')</script></html>", "resume.html", "readable resume text"),
            (b"<html><img src='resume.png'></html>", "resume.html", "readable resume text"),
        ]
        for data, filename, expected in cases:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(InputValidationError, expected):
                    parse_resume_html(data, filename)

    def test_size_limits_reject_without_silent_truncation(self):
        markup = b"<p>Course project only; no professional experience with SQL yet.</p>"
        with self.assertRaisesRegex(InputValidationError, "too large"):
            parse_resume_html(markup, "resume.html", max_bytes=10)
        with self.assertRaisesRegex(InputValidationError, "too long"):
            parse_resume_html(markup, "resume.html", max_chars=40)

    def test_deeply_nested_markup_is_bounded(self):
        markup = b"<div>" * 300 + b"Candidate worked on a course project with Python only." + b"</div>" * 300
        with self.assertRaisesRegex(InputValidationError, "too complex"):
            parse_resume_html(markup, "resume.html")

    def test_confirmed_text_validation_keeps_candidate_corrections(self):
        corrected = "I used Python in a tutorial only. I want advanced practice despite that."
        self.assertEqual(validate_input_text(corrected, "Resume"), corrected)
        with self.assertRaisesRegex(InputValidationError, "enough readable text"):
            validate_input_text("   ", "Job description")
        with self.assertRaisesRegex(InputValidationError, "control characters"):
            validate_input_text(corrected + "\x01", "Resume")


if __name__ == "__main__":
    unittest.main()
