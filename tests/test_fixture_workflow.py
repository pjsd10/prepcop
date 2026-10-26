"""Five-role controlled workflow checks using the real durable Engine.

Only resume.html and job-description.txt are loaded as source input. Candidate
answers below are test-driver actions submitted AFTER a matching saved probe;
evaluator documents and unasked answer scripts never enter model messages.
The model and retrieval replies are deliberately scripted. This suite proves
workflow controls and preservation, not live model quality or web grounding.
"""

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from uuid import uuid4

import pytest

from interview_prep.config import Settings
from interview_prep.engine import Engine
from interview_prep.html_input import parse_resume_html
from test_engine import ScriptedModel, call


FIXTURES = Path(__file__).parent / "fixtures" / "fictional-resumes"


def topic(identity, name, evidence="", exposure="not_sure"):
    return dict(id=identity, name=name, requirement=f"Prepare the JD requirement: {name}.",
                resume_evidence=evidence, exposure=exposure, desired_depth="not_sure",
                notes="", confirmed=False)


CASES = [
    dict(
        case_id="01-software-engineer", role="Backend Software Engineer",
        skills=[
            topic("python", "Python", "Implemented Python endpoints", "used_at_work"),
            topic("api", "REST API design and testing", "including input validation and database queries", "used_at_work"),
            topic("sql", "PostgreSQL and transactions", "investigated slow PostgreSQL queries with a senior engineer", "used_at_work"),
            topic("delivery", "Docker and delivery", "updated an existing GitHub Actions workflow", "used_at_work"),
            topic("kubernetes", "Kubernetes", "a local Kubernetes cluster following a tutorial", "studied_tutorial"),
            topic("aws", "AWS"),
        ],
        selected=["api", "sql", "kubernetes"], depths=["intermediate", "intermediate", "beginner"],
        probe_topics=["api", "sql", "kubernetes", "api", "kubernetes"],
        answers=[
            "I implemented endpoints and validation. I have not designed idempotency keys for a production service.",
            "I fixed queries with guidance. I need intermediate practice on indexes and transactions.",
            "Only a local tutorial. Please use beginner questions, not production-cluster troubleshooting.",
            None, None,
        ],
        unasked=["I listed AWS after a course. I have not provisioned or operated AWS services."],
        final_exposure={"aws": "not_sure", "kubernetes": "studied_tutorial"},
        updates=[], finish_last=True, question_stem="How would you reason about",
        correction="Keep infrastructure ownership with the platform team; no production Kubernetes administration is claimed.",
    ),
    dict(
        case_id="02-qa-engineer", role="API Quality Engineer",
        skills=[
            topic("python", "Python and pytest", "Maintained Selenium and pytest checks", "used_at_work"),
            topic("api", "REST API testing", "used Postman and Python requests against a local demonstration service", "personal_course_project"),
            topic("negative", "Authentication, negative testing and stateful CRUD"),
            topic("sql", "SQL", "Used SQL queries to compare application results with database records", "used_at_work"),
            topic("ci", "Jenkins and CI", "Updated stages in an existing Jenkins pipeline", "used_at_work"),
            topic("docker", "Docker", "ran the demonstration service using an existing compose file", "studied_tutorial"),
            topic("performance", "Performance-testing fundamentals"),
        ],
        selected=["python", "api"], depths=["intermediate", "beginner"],
        probe_topics=["python", "api"],
        answers=[
            "I maintain pytest fixtures and parametrized tests at work. Give me intermediate questions.",
            "My work experience is mostly UI testing. API testing is from a local practice project.",
        ],
        unasked=["I tested a basic token in the lab. I need beginner coverage of auth failure cases."],
        final_exposure={"python": "used_at_work", "api": "personal_course_project"},
        updates=[dict(skill_id="api", answer_index=1, answer_excerpt="API testing is from a local practice project.",
                      exposure="personal_course_project", notes="API practice does not erase professional Python automation.")],
        finish_last=False, question_stem="How would you test",
        correction="Keep Python work experience separate from the local API practice project and existing Jenkins stages.",
    ),
    dict(
        case_id="03-engineering-manager", role="Engineering Manager",
        skills=[
            topic("people", "People management", "Led a five-engineer service team"),
            topic("mentoring", "Mentoring and feedback", "Mentored two engineers through regular technical feedback and pairing", "used_at_work"),
            topic("hiring", "Hiring", "Participated in hiring interviews and provided technical feedback", "used_at_work"),
            topic("delivery", "Delivery planning", "roadmap planning and technical delivery", "used_at_work"),
            topic("technical", "Technical leadership", "Contributed to design reviews", "used_at_work"),
            topic("incidents", "Incident learning", "Coordinated an incident review", "used_at_work"),
        ],
        selected=["people"], depths=["intermediate"], probe_topics=["people"],
        answers=["The five engineers were not my formal direct reports. Their performance reviews were owned by my manager."],
        unasked=["I conducted technical interviews but did not approve offers or headcount."],
        final_exposure={"people": "not_used", "mentoring": "used_at_work"},
        updates=[dict(skill_id="people", answer_index=0, answer_excerpt="The five engineers were not my formal direct reports.",
                      exposure="not_used", notes="Formal line-management authority is distinct from technical leadership.")],
        finish_last=False, question_stem="How would you approach a leadership scenario about",
        correction="Technical mentoring is real work experience; formal direct-report authority was not held.",
    ),
    dict(
        case_id="04-program-manager", role="Technical Program Manager",
        skills=[
            topic("dependencies", "Dependency management", "Maintained dependency maps, decision logs", "used_at_work"),
            topic("risk", "Risk and escalation", "Escalated delivery risks", "used_at_work"),
            topic("stakeholders", "Stakeholder conflict", "facilitated decisions between engineering and operations", "used_at_work"),
            topic("readiness", "Release readiness", "a shared release-readiness checklist", "used_at_work"),
            topic("communication", "Technical communication", "Prepared concise progress updates", "used_at_work"),
            topic("security", "Security review coordination", "Partnered with security specialists to track required review milestones", "used_at_work"),
            topic("sql", "SQL", "queried a synthetic delivery-events dataset", "personal_course_project"),
        ],
        selected=["dependencies", "stakeholders", "security"], depths=["advanced", "advanced", "intermediate"],
        probe_topics=["dependencies", "stakeholders", "security"],
        answers=[
            "I coordinated workstream leads but had no direct reports or final authority over their priorities.",
            "I presented alternatives and escalated conflicts; the sponsor made final scope decisions.",
            "I tracked security reviews. I did not implement identity protocols or perform security assessments.",
        ],
        unasked=["I tracked milestone readiness and blocker age. I cannot claim a measured percentage reduction in delivery time."],
        final_exposure={"security": "used_at_work", "sql": "personal_course_project"},
        updates=[], finish_last=False, question_stem="How would you coordinate a program scenario about",
        correction="Security specialists own implementation; coordination does not establish security-engineering expertise.",
    ),
    dict(
        case_id="05-product-manager", role="Product Manager for Workflow Tools",
        skills=[
            topic("discovery", "User discovery", "Conducted user interviews with internal operations teams", "used_at_work"),
            topic("priorities", "Prioritization", "prioritized usability improvements with design and engineering", "used_at_work"),
            topic("metrics", "Product analytics", "Reviewed adoption dashboards with an analyst", "used_at_work"),
            topic("experiments", "Experiment design", "no completed live experiment is described", "personal_course_project"),
            topic("engineering", "Engineering collaboration", "Helped define an onboarding change", "used_at_work"),
            topic("sql", "SQL", "SQL course project: queried a synthetic event dataset", "personal_course_project"),
        ],
        selected=["experiments", "metrics"], depths=["beginner", "intermediate"],
        probe_topics=["experiments", "metrics"],
        answers=[
            "I proposed an experiment, but it has not launched. I have not analyzed a completed live A/B test.",
            "I reviewed dashboards, but I cannot attribute a measured retention improvement to my work.",
        ],
        unasked=["SQL is from a course. At work an analyst prepared the queries."],
        final_exposure={"experiments": "personal_course_project", "sql": "personal_course_project"},
        updates=[], finish_last=False, question_stem="How would you evaluate product evidence about",
        correction="Keep the proposed experiment distinct from completed results; no retention improvement is attributable here.",
    ),
]


class ControlledRetrieval:
    """Tavily-shaped results with explicitly fictional sources; no HTTP code."""

    def __init__(self, case):
        self.case = case
        self.calls = []

    @staticmethod
    def url(topic_id):
        return f"https://controlled.example/{topic_id}"

    @staticmethod
    def points(topic_id):
        return [f"Clarify the stated assumptions for {topic_id}.",
                f"Identify the relevant constraints for {topic_id}.",
                f"Explain the tradeoffs and remaining uncertainty for {topic_id}."]

    @classmethod
    def text(cls, topic_id):
        return "Controlled test reference. " + " ".join(cls.points(topic_id))

    def search_sources(self, query, timeout=20):
        assert 0 < timeout <= 20
        topic_id = next(topic_id for topic_id in self.case["selected"] if query.startswith(topic_id + " "))
        self.calls.append(("search", topic_id, query))
        return dict(ok=True, status="success", results=[dict(url=self.url(topic_id),
                    title=f"Controlled reference: {topic_id}", content="Controlled search lead, not an independently fetched source.")])

    def extract_page(self, url, timeout=20):
        assert 0 < timeout <= 20
        topic_id = url.rsplit("/", 1)[-1]
        assert topic_id in self.case["selected"]
        self.calls.append(("extract", topic_id, url))
        return dict(ok=True, status="success", content_available=True, results=[dict(url=url,
                    title=f"Controlled reference: {topic_id}", raw_content=self.text(topic_id))])


def _probe(case, index):
    topic_id = case["probe_topics"][index]
    name = next(skill["name"] for skill in case["skills"] if skill["id"] == topic_id)
    question = (f"What did you personally own in {name}?" if index < len(case["selected"])
                else f"Which remaining {name} uncertainty should shape practice in follow-up {index + 1}?")
    return dict(probe=dict(skill_id=topic_id, question=question,
                          selection_reason="Clarify ownership and preparation needs for a selected JD requirement.",
                          addresses_unknown=f"{name}: ownership and limits, probe {index + 1}"))


def _assessment(topic_id, source_id):
    return dict(source_id=source_id, readable=True, accepted=True,
                reason="This controlled reference supports the scripted practice checkpoints for this topic.",
                excerpts=[ControlledRetrieval.text(topic_id)])


def _draft(case):
    questions = []
    count = 6 // len(case["selected"])
    skill_names = {item["id"]: item["name"] for item in case["skills"]}
    for index, topic_id in enumerate(case["selected"]):
        source_id = f"S{2 * (index + 1)}"
        for number in range(count):
            questions.append(dict(id=f"{topic_id}-{number + 1}", topic_id=topic_id,
                question=f"{case['question_stem']} {skill_names[topic_id]}, case {number + 1}?",
                depth=case["depths"][index], explanation="Use the controlled reference to examine assumptions, constraints and tradeoffs.",
                key_points=[dict(text=point, source_ids=[source_id]) for point in ControlledRetrieval.points(topic_id)],
                source_ids=[source_id], candidate_example_prompt="Use a real example only if you personally have one."))
    return dict(assessments=[], report=dict(version=1, brief_version=2, status="complete",
        title=f"Controlled {case['role']} practice", summary="Scripted workflow result; no live model or web evidence was evaluated.",
        questions=questions, study_plan=[dict(topic_id=topic_id,
            activity=f"Review the controlled {topic_id} reference and practise the corresponding questions.",
            minutes=60 // len(case["selected"]), source_ids=[f"S{2 * (index + 1)}"])
            for index, topic_id in enumerate(case["selected"])], gaps=[]))


def _responses(case):
    responses = [dict(role=case["role"], seniority="Candidate-reviewed target", skills=deepcopy(case["skills"]),
                      uncertainties=["Personal ownership beyond the supplied resume remains unconfirmed."])]
    responses.extend(_probe(case, index) for index in range(len(case["probe_topics"])))
    if not case["finish_last"]:
        responses.append(dict(probe=None, stop_reason="Every selected topic has an initial answer; no further scripted clarification is required."))
    responses.append(dict(summary=f"Preparation for {case['role']} using candidate-reviewed exposure and depth.",
                          clarifications=case["updates"], uncertainties=[], resolved_uncertainties=[]))
    for index, topic_id in enumerate(case["selected"]):
        search = dict(topic_id=topic_id, query=f"{topic_id} {case['role']} preparation reference",
                      reason="Find a suitable reference for the approved topic.")
        if index:
            search["assessment"] = _assessment(case["selected"][index - 1], f"S{index * 2}")
        responses.append(call("search_sources", search, f"search-{topic_id}"))
        responses.append(call("extract_page", dict(topic_id=topic_id, url=ControlledRetrieval.url(topic_id),
                         reason="Inspect the controlled reference before accepting its support."), f"extract-{topic_id}"))
    last = case["selected"][-1]
    responses.append(dict(action="draft", reason="The selected topics have enough controlled support for this test.",
                          assessment=_assessment(last, f"S{len(case['selected']) * 2}")))
    responses.append(_draft(case))
    return responses


def _messages_text(model):
    return "\n".join(str(message) for messages, _ in model.calls for message in messages)


def _submit(engine, run_id, action, **fields):
    return engine.submit(run_id, dict(action=action, **fields), uuid4().hex)


def _start(case, folder):
    fixture = FIXTURES / case["case_id"]
    resume = parse_resume_html((fixture / "resume.html").read_bytes(), "resume.html")
    jd = (fixture / "job-description.txt").read_text(encoding="utf-8-sig")
    model = ScriptedModel(_responses(case))
    retrieval = ControlledRetrieval(case)
    settings = Settings(runtime_dir=folder)
    engine = Engine(settings, model=model, retrieval=retrieval)
    run_id = engine.create_run(resume, jd, preparation_minutes=60, interview_stage="Hiring manager")
    state = engine.advance(run_id)
    assert state["pending"]["kind"] == "profile_review"
    assert state["resume_text"] == resume and state["jd_text"] == jd.strip()
    assert len(state["profile"]["skills"]) == len(case["skills"])
    assert retrieval.calls == []
    return engine, run_id, state, model, retrieval, settings


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["case_id"])
def test_five_role_controlled_full_workflow_preserves_matrix_edges(case, tmp_path):
    engine, run_id, state, model, retrieval, settings = _start(case, tmp_path)
    expected_calls = len(model.responses) + 1
    profile = deepcopy(state["profile"])
    for identity, depth in zip(case["selected"], case["depths"]):
        next(item for item in profile["skills"] if item["id"] == identity)["desired_depth"] = depth
    # No fourth priority is silently dropped, even with a longer editable table.
    with pytest.raises(ValueError):
        _submit(engine, run_id, "save_profile", profile=profile,
                selected_topic_ids=[item["id"] for item in profile["skills"][:4]])
    assert len(model.calls) == 1
    state = _submit(engine, run_id, "save_profile", profile=profile, selected_topic_ids=case["selected"],
                    preparation_minutes=60, interview_stage="Hiring manager")
    for index, topic_id in enumerate(case["probe_topics"]):
        assert state["pending"]["kind"] == "clarification"
        assert state["pending_probe"]["skill_id"] == topic_id
        assert state["probe_count"] == index + 1 <= 5
        captured = _messages_text(model)
        for not_yet_answered in case["answers"][index:]:
            if not_yet_answered:
                assert not_yet_answered not in captured
        reply = case["answers"][index]
        if index == len(case["probe_topics"]) - 1 and case["finish_last"]:
            state = _submit(engine, run_id, "finish")
        elif reply is None:
            state = _submit(engine, run_id, "skip")
        else:
            state = _submit(engine, run_id, "answer", answer=reply)
    assert state["pending"]["kind"] == "brief_review"
    assert state["probe_count"] == len(case["probe_topics"])
    assert [item["probe"]["skill_id"] for item in state["answers"][:len(case["selected"])]] == case["selected"]
    skills = {item["id"]: item for item in state["brief"]["skills"]}
    for identity, expected in case["final_exposure"].items():
        assert skills[identity]["exposure"] == expected
    for identity, depth in zip(case["selected"], case["depths"]):
        assert skills[identity]["desired_depth"] == depth
    for index, reply in enumerate(case["answers"]):
        saved = state["answers"][index]
        if reply is None:
            assert saved["skipped"] and saved["answer"] == ""
            assert saved["probe"]["addresses_unknown"] in " ".join(state["brief"]["uncertainties"])
        else:
            assert saved["answer"] == reply
            assert reply in skills[saved["probe"]["skill_id"]]["notes"]
    assert retrieval.calls == []
    assert state["brief"]["deferred_skills"] == [item["name"] for item in case["skills"] if item["id"] not in case["selected"]]
    before_correction_calls = len(model.calls)
    revision = deepcopy(state["brief"])
    revision["summary"] = case["correction"]
    state = _submit(engine, run_id, "revise_brief", brief=revision)
    assert state["brief"]["version"] == 2 and state["approved_brief_version"] is None
    assert len(model.calls) == before_correction_calls
    with pytest.raises(ValueError, match="stale"):
        _submit(engine, run_id, "approve_brief", version=1)
    with pytest.raises(ValueError):
        engine.export_report(run_id)
    restarted = Engine(settings, model=model, retrieval=retrieval)
    assert restarted.get_run(run_id)["brief"] == state["brief"]
    state = _submit(restarted, run_id, "approve_brief", version=2)
    assert state["pending"]["kind"] == "report_review"
    report = state["report"]
    assert report["status"] == "complete" and report["brief_version"] == 2
    assert Counter(item["topic_id"] for item in report["questions"]) == {
        identity: 6 // len(case["selected"]) for identity in case["selected"]}
    assert len({item["question"] for item in report["questions"]}) == 6
    assert sum(item["minutes"] for item in report["study_plan"]) == 60
    if "Manager" in case["role"]:
        assert all(case["question_stem"] in item["question"] for item in report["questions"])
        assert all("Python" not in item["question"] for item in report["questions"])
    counters = state["counters"]
    assert counters["model_attempts"] == expected_calls <= 32
    assert counters["search_attempts"] == len(case["selected"]) <= 4
    assert counters["extract_attempts"] == {identity: 1 for identity in case["selected"]}
    assert counters["readable_pages"] == {identity: 1 for identity in case["selected"]}
    assert counters["repairs"] == counters["transient_retries"] == 0
    assert len(retrieval.calls) == 2 * len(case["selected"])
    assert all(item["status"] == "accepted" and item["excerpt"] in item["text"] for item in state["evidence"])
    # Prove tool results actually enter a later model request, not just the ledger.
    tool_results = [message for messages, _ in model.calls for message in messages
                    if getattr(message, "type", None) == "tool"]
    for identity in case["selected"]:
        assert any(message.tool_call_id == f"extract-{identity}" and ControlledRetrieval.text(identity) in message.content
                   for message in tool_results)
    captured = _messages_text(model)
    for forbidden in ["evaluator-only", "candidate_reply_script", "must_not_infer", *case["unasked"]]:
        assert forbidden not in captured
    with pytest.raises(ValueError):
        restarted.export_report(run_id)
    review_id = uuid4().hex
    state = restarted.submit(run_id, dict(action="review_report", version=report["version"]), review_id)
    assert state["stage"] == "complete"
    export = restarted.export_report(run_id).decode("utf-8")
    assert case["role"] in export and case["correction"] in export
    assert export.count('<article class="question">') == 6
    completed_calls = len(model.calls)
    second_restart = Engine(settings, model=model, retrieval=retrieval)
    second_restart.submit(run_id, dict(action="review_report", version=report["version"]), review_id)
    second_restart.get_run(run_id)
    second_restart.advance(run_id)
    assert second_restart.export_report(run_id).decode("utf-8") == export
    assert len(model.calls) == completed_calls and not model.responses


def test_distinct_fictional_runs_share_storage_without_crossing_candidate_inputs(tmp_path):
    first = _start(CASES[0], tmp_path)
    second = _start(CASES[-1], tmp_path)
    first_engine, first_id, first_state, first_model, _, _ = first
    second_engine, second_id, second_state, second_model, _, _ = second
    assert first_id != second_id
    assert first_engine.get_run(first_id)["profile"]["role"] == CASES[0]["role"]
    assert second_engine.get_run(second_id)["profile"]["role"] == CASES[-1]["role"]
    assert first_state["resume_text"] not in _messages_text(second_model)
    assert second_state["resume_text"] not in _messages_text(first_model)
    assert first_engine.store.counters(first_id)["model_attempts"] == 1
    assert second_engine.store.counters(second_id)["model_attempts"] == 1
    assert first_engine.store.events(first_id) != second_engine.store.events(second_id)
