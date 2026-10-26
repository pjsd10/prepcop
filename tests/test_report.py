"""Controlled renderer checks; no provider or browser requests are performed."""

from copy import deepcopy
from html.parser import HTMLParser
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from interview_prep.report import build_reviewed_export, render_report_html, safe_source_url


def example_state() -> dict:
    return {
        "run_id": "renderer-test-only",
        "approved_brief_version": 2,
        "report_reviewed_version": 3,
        "brief": {
            "version": 2, "role": "QA Engineer", "seniority": "", "summary": "Tutorial exposure only; no work use.",
            "interview_stage": "Technical screen", "preparation_minutes": 30,
            "skills": [{"id": "api", "name": "API testing", "exposure": "studied_tutorial",
                        "desired_depth": "advanced", "notes": "No production ownership", "confirmed": True},
                       {"id": "leadership", "name": "Leadership", "exposure": "used_at_work",
                        "desired_depth": "intermediate", "notes": "", "confirmed": True}],
            "selected_topic_ids": ["api"], "deferred_skills": ["Leadership"],
            "uncertainties": ["Project ownership is unclear"],
        },
        "report": {
            "version": 3, "brief_version": 2, "status": "partial", "title": "API practice",
            "summary": "One supported practice question.",
            "questions": [{"id": "q1", "topic_id": "api", "question": "How would you test an API response?",
                           "depth": "advanced", "explanation": "Check behavior and response data.",
                           "source_ids": ["s1"],
                           "key_points": [{"text": "Check status and required fields.", "source_ids": ["s1"]}],
                           "candidate_example_prompt": "Use your own tutorial example."}],
            "study_plan": [{"topic_id": "api", "activity": "Read and practise the checks.",
                            "minutes": 30, "source_ids": ["s1"]}],
            "gaps": ["Five more supported questions were not available."],
        },
        "evidence": [{"id": "s1", "topic_id": "api", "title": "API testing guide",
                      "url": "https://example.org/api?lang=en&mode=guide", "excerpt": "Check the status and response fields.",
                      "status": "accepted", "reason": "Relevant response checks.",
                      "retrieved_at": "2026-10-25T20:00:00Z"}],
    }


class TagCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []
        self.links = []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        if tag == "a":
            self.links.append(dict(attrs).get("href"))


class ReportTests(unittest.TestCase):
    def test_export_is_deterministic_and_does_not_mutate_state(self):
        state = example_state()
        untouched = deepcopy(state)
        first = build_reviewed_export(state)
        self.assertEqual(first, build_reviewed_export(state))
        self.assertEqual(state, untouched)
        self.assertIn(b"Report version 3", first)
        self.assertIn(b"Brief version 2", first)
        self.assertIn(b"1 of 6 practice questions", first)
        self.assertIn(b"Partial", first)

    def test_stale_or_missing_review_is_blocked(self):
        for reviewed in (None, 1, 2, 4):
            with self.subTest(reviewed=reviewed):
                state = example_state()
                state["report_reviewed_version"] = reviewed
                with self.assertRaisesRegex(ValueError, "current report version"):
                    build_reviewed_export(state)

    def test_current_approved_brief_must_match_report(self):
        state = example_state()
        state["brief"]["version"] = 4
        with self.assertRaisesRegex(ValueError, "earlier brief"):
            build_reviewed_export(state)
        state = example_state()
        state["approved_brief_version"] = None
        with self.assertRaisesRegex(ValueError, "approved"):
            build_reviewed_export(state)

    def test_no_report_and_invalid_version_cannot_export(self):
        with self.assertRaisesRegex(ValueError, "no saved report"):
            build_reviewed_export({})
        state = example_state()
        del state["report"]["version"]
        state["report_reviewed_version"] = None
        with self.assertRaisesRegex(ValueError, "valid review version"):
            build_reviewed_export(state)

    def test_every_untrusted_surface_is_escaped(self):
        state = example_state()
        hostile = '<script>alert("unsafe")</script><img src="https://tracker.invalid/pixel">'
        state["report"]["title"] = hostile
        state["report"]["summary"] = hostile
        state["report"]["questions"][0]["question"] = hostile
        state["report"]["questions"][0]["explanation"] = hostile
        state["report"]["questions"][0]["key_points"][0]["text"] = hostile
        state["report"]["gaps"] = [hostile]
        state["brief"]["skills"][0]["notes"] = hostile
        state["evidence"][0]["title"] = hostile
        state["evidence"][0]["excerpt"] = hostile
        state["evidence"][0]["reason"] = hostile
        rendered = build_reviewed_export(state).decode()
        parsed = TagCollector()
        parsed.feed(rendered)
        self.assertNotIn("script", parsed.tags)
        self.assertNotIn("img", parsed.tags)
        self.assertIn("&lt;script&gt;", rendered)
        self.assertNotIn(hostile, rendered)
        self.assertIn("Content-Security-Policy", rendered)

    def test_links_only_allow_plain_http_and_https(self):
        rejected = ["javascript:alert(1)", "data:text/html,hello", "file:///tmp/a", "//example.org",
                    "https://name:password@example.org", "https://example.org\n.bad", "https:///", "https://example.org:bad"]
        for url in rejected:
            with self.subTest(url=url):
                self.assertIsNone(safe_source_url(url))
        self.assertEqual(safe_source_url("https://example.org/page"), "https://example.org/page")
        self.assertEqual(safe_source_url("http://example.org/page"), "http://example.org/page")
        state = example_state()
        state["evidence"][0]["url"] = "javascript:alert(1)"
        output = build_reviewed_export(state).decode()
        self.assertNotIn('href="javascript:', output)
        self.assertIn("safe web link is unavailable", output)

    def test_missing_or_rejected_support_is_visible_and_not_linked(self):
        state = example_state()
        state["report"]["questions"][0]["source_ids"] = ["s1", "missing"]
        state["evidence"][0]["status"] = "rejected"
        state["evidence"][0]["reason"] = "Wrong topic."
        output = build_reviewed_export(state).decode()
        self.assertIn("missing: support unavailable", output)
        self.assertIn("s1: support unavailable", output)
        self.assertNotIn('href="#source-1"', output)
        self.assertIn("Wrong topic.", output)

    def test_exposure_and_depth_remain_separate(self):
        output = build_reviewed_export(example_state()).decode()
        self.assertIn("Self-reported exposure", output)
        self.assertIn("Requested depth", output)
        self.assertIn("studied tutorial", output)
        self.assertIn("advanced", output)
        self.assertIn("no work use", output)
        self.assertIn("No production ownership", output)
        self.assertIn("Deferred this run: Leadership", output)

    def test_failed_result_keeps_saved_evidence_without_padding(self):
        state = example_state()
        state["report"]["status"] = "failed"
        state["report"]["questions"] = []
        state["report"]["study_plan"] = []
        output = build_reviewed_export(state).decode()
        self.assertIn("Failed", output)
        self.assertIn("0 of 6 practice questions", output)
        self.assertIn("No supported practice questions", output)
        self.assertIn("Check the status and response fields.", output)
        self.assertNotIn('class="question"', output)

    def test_model_compatible_input(self):
        class SavedModel:
            def __init__(self, value):
                self.value = value

            def model_dump(self, *, mode):
                self.mode = mode
                return deepcopy(self.value)

        state = example_state()
        rendered = render_report_html(SavedModel(state["report"]), brief=SavedModel(state["brief"]),
                                      sources=[SavedModel(state["evidence"][0])])
        self.assertIn("API practice", rendered)
        self.assertIn('href="#source-1"', rendered)


def controlled_ui(view: str, initial: dict) -> None:
    """Executed by Streamlit AppTest; the fake only records explicit actions."""
    import streamlit as st
    from interview_prep.report import build_reviewed_export
    from interview_prep.ui import _brief_review, _progress, _report

    if "saved" not in st.session_state:
        st.session_state["saved"] = initial
        st.session_state["submitted"] = []
        st.session_state["exports"] = 0

    class ControlledEngine:
        def submit(self, run_id, event, event_id):
            st.session_state["submitted"].append(event)
            if event["action"] == "review_report":
                st.session_state["saved"]["report_reviewed_version"] = event["version"]
            return st.session_state["saved"]

        def export_report(self, run_id):
            st.session_state["exports"] += 1
            return build_reviewed_export(st.session_state["saved"])

    if view == "brief":
        _brief_review(ControlledEngine(), st.session_state["saved"], [])
    elif view == "progress":
        _progress(st.session_state["saved"])
    else:
        _report(ControlledEngine(), st.session_state["saved"])


class StreamlitInteractionTests(unittest.TestCase):
    def _app(self, view, state=None):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_function(controlled_ui, args=(view, state or example_state()), default_timeout=10).run()
        self.assertEqual(len(app.exception), 0)
        return app

    def test_idle_brief_reruns_do_not_submit_actions(self):
        app = self._app("brief")
        self.assertEqual(app.session_state["submitted"], [])
        self.assertEqual(app.multiselect[0].proto.max_selections, 3)
        app.run()
        self.assertEqual(app.session_state["submitted"], [])

    def test_edited_brief_is_saved_before_a_new_approval(self):
        app = self._app("brief")
        app.text_area[0].set_value("Corrected: I used this only in a course.")
        self.assertTrue(app.button[1].disabled)
        app.checkbox[0].check().run()
        self.assertFalse(app.button[1].disabled)
        app.button[1].click().run()
        self.assertEqual(len(app.exception), 0)
        events = app.session_state["submitted"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["action"], "revise_brief")
        self.assertEqual(events[0]["brief"]["summary"], "Corrected: I used this only in a course.")

    def test_unchanged_approved_brief_submits_its_exact_version_once(self):
        app = self._app("brief")
        self.assertTrue(app.button[1].disabled)
        app.checkbox[0].check().run()
        self.assertFalse(app.button[1].disabled)
        app.button[1].click().run()
        self.assertEqual(app.session_state["submitted"], [{"action": "approve_brief", "version": 2}])
        app.run()
        self.assertEqual(len(app.session_state["submitted"]), 1)

    def test_report_review_is_required_and_download_is_cached(self):
        state = example_state()
        state["report_reviewed_version"] = None
        app = self._app("report", state)
        self.assertEqual(app.session_state["exports"], 0)
        review = next(item for item in app.button if item.label == "Mark reviewed and prepare download")
        self.assertTrue(review.disabled)
        self.assertEqual(app.session_state["submitted"], [])
        self.assertEqual(app.session_state["exports"], 0)
        app.checkbox[0].check().run()
        review = next(item for item in app.button if item.label == "Mark reviewed and prepare download")
        self.assertFalse(review.disabled)
        review.click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state["submitted"], [{"action": "review_report", "version": 3}])
        self.assertEqual(app.session_state["exports"], 1)
        app.run()
        self.assertEqual(app.session_state["exports"], 1)

    def test_saved_report_change_cannot_reuse_previous_cached_bytes(self):
        app = self._app("report")
        old_export = app.session_state["report_download"]["data"]
        saved = app.session_state["saved"]
        saved["report"]["version"] = 4
        saved["report"]["title"] = "Updated report"
        saved["report_reviewed_version"] = 4
        app.run()
        self.assertEqual(app.session_state["exports"], 2)
        self.assertNotEqual(app.session_state["report_download"]["data"], old_export)

    def test_progress_uses_factual_allowlist_not_arbitrary_event_text(self):
        from interview_prep.ui import progress_message
        hostile = "private provider payload <script>run()</script>"
        self.assertIsNone(progress_message("unrecognized", {"message": hostile}))
        message = progress_message("research_decision", {"action": "extract_page", "reason": hostile})
        self.assertEqual(message, "Reading a selected source…")
        self.assertNotIn(hostile, message)
        self.assertIn("partial", progress_message("report_drafted", {"status": "partial"}))

    def test_failed_extraction_and_unassessed_success_have_distinct_visible_outcomes(self):
        state = example_state()
        state["report_reviewed_version"] = None
        failed = {**state["evidence"][0], "id": "failed", "status": "rejected", "readable": False,
                  "text": "", "excerpt": "", "reason": "Page extraction failed or access was blocked."}
        saved_text = '<script>unsafe()</script> ![tracking](https://example.invalid/pixel) ' + "saved page text " * 1500
        unassessed = {**state["evidence"][0], "id": "unassessed", "status": "uncertain", "readable": None,
                      "text": saved_text, "excerpt": "", "reason": "Readability and support assessment could not finish."}
        state["evidence"] = [failed, unassessed]
        app = self._app("report", state)
        captions = "\n".join(element.value for element in app.caption)
        self.assertIn("Extraction unsuccessful · No text available for assessment", captions)
        self.assertIn("Extraction succeeded · Assessment incomplete or uncertain", captions)
        self.assertNotIn("Sources were assessed by the Research Coach", captions)
        self.assertEqual(len(app.text), 1)
        self.assertEqual(app.text[0].value, saved_text[:500])
        self.assertEqual(len(app.text[0].value), 500)
        self.assertNotIn(saved_text[:500], "\n".join(element.value for element in app.markdown))
        self.assertEqual(app.session_state["saved"], state)
        self.assertEqual(app.session_state["submitted"], [])
        self.assertEqual(app.session_state["exports"], 0)

    def test_pending_success_is_visible_without_implying_completed_assessment(self):
        state = example_state()
        state["report_reviewed_version"] = None
        state["evidence"][0].update(status="pending", readable=None, text="Saved page awaiting assessment.", excerpt="")
        app = self._app("report", state)
        captions = "\n".join(element.value for element in app.caption)
        self.assertIn("Extraction succeeded · Assessment pending", captions)
        self.assertEqual(app.text[0].value, "Saved page awaiting assessment.")

    def test_zero_readable_counter_explains_unconfirmed_assessment(self):
        state = example_state()
        state["counters"] = {"extract_attempts": {"api": 4}, "readable_pages": {}}
        app = self._app("progress", state)
        visible = "\n".join(element.value for element in app.markdown)
        captions = "\n".join(element.value for element in app.caption)
        self.assertIn("0 of 4 pages confirmed readable by the Coach", visible)
        self.assertIn("Extraction can succeed before readability is confirmed", captions)
        self.assertIn("no page has yet been confirmed readable", captions)


if __name__ == "__main__":
    unittest.main()
