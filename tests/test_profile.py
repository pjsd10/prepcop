"""Controlled-response policy tests. These do not establish live model quality."""

import copy
import unittest

from pydantic import ValidationError

from interview_prep.profile import (
    allowed_probe_skill_ids,
    make_brief,
    map_profile,
    next_probe,
    validate_selected_topics,
)


class ControlledContext:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []
        self.completed = {}

    def structured_call(self, key, messages, schema):
        self.calls.append((key, messages, schema))
        if key not in self.completed:
            self.completed[key] = schema.model_validate(self.responses.pop(0))
        return self.completed[key]


def skill(skill_id, exposure="not_sure", depth="not_sure"):
    return {
        "id": skill_id,
        "name": skill_id.title(),
        "requirement": f"Apply {skill_id} to the target role.",
        "resume_evidence": "",
        "exposure": exposure,
        "desired_depth": depth,
        "notes": "Candidate correction: limited ownership.",
        "confirmed": True,
    }


def probe(skill_id, question=None):
    return {
        "skill_id": skill_id,
        "question": question or f"What have you personally done with {skill_id}?",
        "selection_reason": "Clarifies ownership for relevant practice.",
        "addresses_unknown": f"Actual personal ownership of {skill_id} work",
    }


def answered(skill_id, text="A tutorial only; I want advanced practice.", skipped=False, question=None):
    return {"probe": probe(skill_id, question), "answer": "" if skipped else text, "skipped": skipped}


def state(selected=None):
    return {
        "run_id": "isolated-controlled-profile-test",
        "input_version": 1,
        "profile_version": 3,
        "resume_text": "Candidate has used Python in a course project and assisted SQL testing at work.",
        "jd_text": "Software engineer role requiring Python automation, SQL analysis and Kubernetes deployments.",
        "profile": {
            "role": "Software engineer",
            "seniority": "Mid-level",
            "skills": [skill("python", "personal_course_project", "advanced"), skill("sql"), skill("kubernetes"), skill("leadership")],
            "uncertainties": ["Ownership of Kubernetes deployment is unclear."],
        },
        "selected_topic_ids": selected or ["python", "sql", "kubernetes"],
        "preparation_minutes": 75,
        "interview_stage": "Technical interview",
        "answers": [],
        "probe_count": 0,
        "pending_probe": None,
        "finish_probing": False,
    }


class ProfileTests(unittest.TestCase):
    def test_selection_rejects_zero_duplicates_unknown_and_four_topics(self):
        profile = state()["profile"]
        for selected in ([], ["python", "python"], ["missing"], ["python", "sql", "kubernetes", "leadership"], [""]):
            with self.subTest(selected=selected), self.assertRaises(ValueError):
                validate_selected_topics(profile, selected)
        for selected in (["python"], ["python", "sql"], ["python", "sql", "kubernetes"]):
            self.assertEqual(validate_selected_topics(profile, selected), selected)

    def test_mapping_preserves_all_major_skills_and_requires_no_human_confirmation(self):
        current = state()
        proposed = copy.deepcopy(current["profile"])
        for item in proposed["skills"]:
            item["confirmed"] = False
        proposed["skills"][0]["resume_evidence"] = "used Python in a course project"
        ctx = ControlledContext(proposed)
        result = map_profile(current, ctx)
        self.assertEqual(len(result["profile"]["skills"]), 4)
        self.assertEqual(result["stage"], "profile_review")
        self.assertIsNone(result["approved_brief_version"])
        self.assertEqual(len(ctx.calls), 1)
        self.assertNotIn("evaluator", ctx.calls[0][1][1]["content"].lower())
        proposed["skills"][0]["confirmed"] = True
        with self.assertRaises(ValidationError):
            map_profile(current, ControlledContext(proposed))

    def test_mapping_rejects_invented_resume_excerpt_and_empty_inputs_before_call(self):
        current = state()
        proposed = copy.deepcopy(current["profile"])
        for item in proposed["skills"]:
            item["confirmed"] = False
        proposed["skills"][0]["resume_evidence"] = "Owned production Python systems for 20 years."
        with self.assertRaises(ValidationError):
            map_profile(current, ControlledContext(proposed))
        current["jd_text"] = ""
        ctx = ControlledContext()
        with self.assertRaises(ValueError):
            map_profile(current, ctx)
        self.assertEqual(ctx.calls, [])

    def test_initial_coverage_is_required_before_followups_or_model_stop(self):
        current = state()
        current["answers"] = [answered("python")]
        current["probe_count"] = 1
        self.assertEqual(allowed_probe_skill_ids(current), ["sql", "kubernetes"])
        with self.assertRaises(ValidationError):
            next_probe(current, ControlledContext({"probe": probe("python", "What Python project did you own? ")}))
        with self.assertRaises(ValidationError):
            next_probe(current, ControlledContext({"probe": None, "stop_reason": "Enough context."}))
        ctx = ControlledContext({"probe": probe("sql")})
        result = next_probe(current, ctx)
        self.assertEqual(result["pending_probe"]["skill_id"], "sql")
        self.assertEqual(result["probe_count"], 2)

    def test_skipped_question_counts_as_initial_opportunity_but_remains_unknown(self):
        current = state()
        current["answers"] = [answered("python", skipped=True), answered("sql", skipped=True)]
        current["probe_count"] = 2
        self.assertEqual(allowed_probe_skill_ids(current), ["kubernetes"])
        result = next_probe(current, ControlledContext({"probe": probe("kubernetes")}))
        self.assertEqual(result["probe_count"], 3)

    def test_adaptive_followup_sees_answers_and_rejects_repeat_question(self):
        current = state()
        current["answers"] = [answered("python"), answered("sql", "Mostly joins."), answered("kubernetes", "I deployed an app.")]
        current["probe_count"] = 3
        with self.assertRaises(ValidationError):
            next_probe(current, ControlledContext({"probe": probe("sql")}))
        ctx = ControlledContext({"probe": probe("kubernetes", "Was that deployment independent work or a guided tutorial?")})
        result = next_probe(current, ctx)
        self.assertEqual(result["probe_count"], 4)
        self.assertIn("I deployed an app.", ctx.calls[0][1][1]["content"])
        self.assertEqual(result["pending_probe"]["skill_id"], "kubernetes")

    def test_cap_finish_and_existing_pending_do_not_make_another_call(self):
        capped = state(["python"])
        capped["probe_count"] = 5
        finish = state(["python"])
        finish["finish_probing"] = True
        pending = state(["python"])
        pending["pending_probe"] = probe("python")
        pending["probe_count"] = 1
        for current in (capped, finish, pending):
            with self.subTest(current=current):
                ctx = ControlledContext()
                result = next_probe(current, ctx)
                self.assertEqual(ctx.calls, [])
                self.assertLessEqual(result["probe_count"], 5)
        self.assertTrue(next_probe(capped, ControlledContext())["finish_probing"])
        self.assertTrue(next_probe(finish, ControlledContext())["finish_probing"])
        self.assertEqual(next_probe(pending, ControlledContext())["pending_probe"], pending["pending_probe"])

    def test_model_may_stop_after_initial_coverage(self):
        current = state(["python"])
        current["answers"] = [answered("python")]
        current["probe_count"] = 1
        result = next_probe(current, ControlledContext({"probe": None, "stop_reason": "Exposure and desired practice are explicit."}))
        self.assertTrue(result["finish_probing"])
        self.assertEqual(result["probe_count"], 1)

    def test_brief_keeps_authoritative_fields_and_separates_exposure_from_depth(self):
        current = state(["python", "sql"])
        current["answers"] = [answered("python"), answered("sql", skipped=True)]
        current["probe_count"] = 2
        synthesis = {
            "summary": "Advanced Python preparation with scaffolding; SQL exposure remains unclear.",
            "clarifications": [{
                "skill_id": "python", "answer_index": 0, "answer_excerpt": "A tutorial only",
                "exposure": "studied_tutorial", "desired_depth": None,
                "notes": "Candidate corrected project experience to guided tutorial exposure.",
            }],
            "uncertainties": [],
        }
        result = make_brief(current, ControlledContext(synthesis))
        brief = result["brief"]
        self.assertEqual(brief["version"], 3)
        self.assertEqual(brief["selected_topic_ids"], ["python", "sql"])
        self.assertEqual(brief["preparation_minutes"], 75)
        self.assertEqual(brief["interview_stage"], "Technical interview")
        self.assertEqual(brief["deferred_skills"], ["Kubernetes", "Leadership"])
        python = brief["skills"][0]
        self.assertEqual(python["exposure"], "studied_tutorial")
        self.assertEqual(python["desired_depth"], "advanced")
        self.assertFalse(python["confirmed"])
        self.assertIn("Candidate correction: limited ownership.", python["notes"])
        self.assertIn("A tutorial only; I want advanced practice.", python["notes"])
        self.assertTrue(any("skipped clarification" in item for item in brief["uncertainties"]))
        self.assertEqual(result["stage"], "brief_review")
        self.assertIsNone(result["approved_brief_version"])

    def test_brief_rejects_skipped_unrelated_or_fabricated_answer_support(self):
        current = state(["python", "sql"])
        current["answers"] = [answered("python"), answered("sql", skipped=True)]
        for update in (
            {"skill_id": "sql", "answer_index": 1, "answer_excerpt": "Skipped", "exposure": "used_at_work"},
            {"skill_id": "sql", "answer_index": 0, "answer_excerpt": "tutorial only", "exposure": "used_at_work"},
            {"skill_id": "python", "answer_index": 0, "answer_excerpt": "Owned production Python", "exposure": "used_at_work"},
        ):
            with self.subTest(update=update), self.assertRaises(ValidationError):
                make_brief(current, ControlledContext({"summary": "Proposed brief", "clarifications": [update]}))

    def test_early_finish_preserves_unknowns_without_manufactured_replies(self):
        current = state(["python"])
        current["finish_probing"] = True
        result = make_brief(current, ControlledContext({"summary": "Prepare advanced Python, with limited course exposure."}))
        self.assertTrue(any("no clarification answer" in item for item in result["brief"]["uncertainties"]))
        self.assertEqual(current["answers"], [])
        self.assertEqual(result["brief"]["skills"][0]["desired_depth"], "advanced")

    def test_resolution_requires_stored_human_evidence(self):
        current = state(["kubernetes"])
        current["answers"] = [answered("kubernetes", "A guided tutorial only; no independent deployment.")]
        unknown = current["profile"]["uncertainties"][0]
        result = make_brief(current, ControlledContext({
            "summary": "Kubernetes practice should acknowledge guided tutorial exposure.",
            "resolved_uncertainties": [{"uncertainty": unknown, "answer_index": 0, "answer_excerpt": "A guided tutorial only"}],
        }))
        self.assertNotIn(unknown, result["brief"]["uncertainties"])
        current["answers"][0]["skipped"] = True
        with self.assertRaises(ValidationError):
            make_brief(current, ControlledContext({
                "summary": "Proposed brief",
                "resolved_uncertainties": [{"uncertainty": unknown, "answer_index": 0, "answer_excerpt": "A guided tutorial only"}],
            }))

    def test_operation_identity_reuses_same_input_but_changes_with_new_answer(self):
        current = state(["python"])
        current["answers"] = [answered("python")]
        ctx = ControlledContext({"summary": "Preserve advanced preference with tutorial scaffolding."}, {"summary": "Updated direct answer."})
        first = make_brief(current, ctx)
        second = make_brief(current, ctx)
        self.assertEqual(first, second)
        self.assertEqual(ctx.calls[0][0], ctx.calls[1][0])
        current["answers"][0]["answer"] = "I now want intermediate practice from this course project."
        make_brief(current, ctx)
        self.assertNotEqual(ctx.calls[1][0], ctx.calls[2][0])


if __name__ == "__main__":
    unittest.main()
