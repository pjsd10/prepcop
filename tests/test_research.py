"""Coach tests using controlled model/tool responses, never live providers."""
import json

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from interview_prep.errors import ProviderFailure, UnknownOperation
from interview_prep.models import Brief, Evidence, PrepReport, Skill
from interview_prep.research import DraftResult, _apply_assessment, _assess_draft, _validate_report, draft_pack, research_decide, research_execute, SourceAssessment


TEXT = ("Fixtures provide a reusable setup context for tests. Fixture dependencies can be composed. "
        "Teardown releases resources after a test. Scope controls fixture reuse. "
        "Assertions compare actual behavior with expected results.")


def state_for(count=1):
    skills = [Skill(id=f"t{i}", name=f"Practice topic {i}", requirement="Explain choices and limitations",
                    exposure="studied_tutorial", desired_depth="advanced") for i in range(count)]
    brief = Brief(version=2, role="QA Engineer", skills=skills, selected_topic_ids=[skill.id for skill in skills],
                  preparation_minutes=60)
    return {"run_id": "controlled-research-run", "brief": brief.model_dump(), "approved_brief_version": 2,
            "selected_topic_ids": brief.selected_topic_ids, "research_step": 0, "evidence": [], "candidates": []}


def evidence_for(topic="t0", source_id="S2", status="accepted"):
    return Evidence(id=source_id, topic_id=topic, title="Primary guide", url="https://docs.example.com/guide",
                    text=TEXT, excerpt=TEXT if status == "accepted" else "", retrieved_at="2026-10-25T00:00:00Z",
                    status=status, readable=True if status in ("accepted", "rejected") else None,
                    reason="Supports explanations of setup, teardown and reuse.").model_dump()


def assessment(source_id="S2", accepted=True, excerpts=None, readable=True):
    return {"source_id": source_id, "readable": readable, "accepted": accepted,
            "reason": "The extracted guide explains reusable setup and teardown with explicit limitations.",
            "excerpts": [TEXT] if excerpts is None and accepted else (excerpts or [])}


def tool_message(name, arg, topic="t0", source_assessment=None, call_id="call-1"):
    field = "query" if name == "search_sources" else "url"
    return AIMessage(content="", tool_calls=[{"id": call_id, "name": name, "args": {
        "topic_id": topic, field: arg, "reason": "Find appropriate primary guidance.", "assessment": source_assessment}}])


def report_for(state):
    brief = Brief.model_validate(state["brief"])
    questions, plan = [], []
    for topic in brief.selected_topic_ids:
        source = next(item for item in state["evidence"] if item["topic_id"] == topic)
        for index in range(6 // len(brief.selected_topic_ids)):
            questions.append({"id": f"{topic}-Q{index}", "topic_id": topic,
                              "question": f"How would you explain design choice {index} for {topic}?",
                              "depth": "advanced", "explanation": "Reusable setup makes test dependencies explicit.",
                              "source_ids": [source["id"]], "key_points": [
                                  {"text": "Describe reusable setup.", "source_ids": [source["id"]]},
                                  {"text": "Explain teardown.", "source_ids": [source["id"]]},
                                  {"text": "Discuss scope.", "source_ids": [source["id"]]}]})
        plan.append({"topic_id": topic, "activity": "Read the guide and practice explaining design choices.",
                     "minutes": 60 // len(brief.selected_topic_ids), "source_ids": [source["id"]]})
    return {"version": 1, "brief_version": 2, "status": "complete", "title": "Practice pack",
            "questions": questions, "study_plan": plan, "gaps": []}


class FakeRetrieval:
    def search_sources(self, query, timeout):
        return {"ok": True, "status": "success", "results": [{"url": "https://docs.example.com/guide",
                "title": "Primary guide", "content": "A search lead, not accepted evidence."}]}

    def extract_page(self, url, timeout):
        return {"ok": True, "status": "success", "content_available": True,
                "results": [{"url": url, "raw_content": TEXT}]}


class FakeContext:
    def __init__(self, responses=(), drafts=()):
        self.responses, self.drafts = list(responses), list(drafts)
        self.retrieval = FakeRetrieval()
        self.model_calls, self.draft_calls, self.tool_calls, self.readable, self.events = [], [], [], [], []
        self.remaining, self.models, self.retrieve, self.repair_used = 300, True, True, False
        self.repair_key = None
        self.completed = set()

    def completed_operation(self, key):
        return key in self.completed

    def can_model(self, reserve=0):
        return self.models

    def can_retrieve(self, kind, topic):
        return self.retrieve

    def remaining_stage_seconds(self):
        return self.remaining

    def consume_repair(self, key=None):
        if key and key == self.repair_key:
            return True
        if self.repair_used or not self.models:
            return False
        self.repair_used = True
        self.repair_key = key
        return True

    def model_call(self, key, messages, tools=None, max_tokens=None):
        self.model_calls.append((key, messages, tools))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def structured_call(self, key, messages, schema):
        self.draft_calls.append((key, messages, schema))
        response = self.drafts.pop(0)
        if isinstance(response, Exception):
            raise response
        return schema.model_validate(response)

    def execute_tool(self, key, kind, topic, args, function):
        self.tool_calls.append((key, kind, topic, args))
        return function(20)

    def record_readable(self, topic, source):
        if (topic, source) not in self.readable:
            self.readable.append((topic, source))

    def event(self, kind, detail):
        self.events.append((kind, detail))


def test_real_native_tool_protocol_roundtrip_with_controlled_responses():
    state = state_for()
    ctx = FakeContext([
        tool_message("search_sources", "fixture lifecycle primary documentation", call_id="search"),
        tool_message("extract_page", "https://docs.example.com/guide", call_id="extract"),
        AIMessage(content=json.dumps({"action": "draft", "reason": "There is sufficient substantive support.",
                                      "assessment": assessment()})),
    ])
    state.update(research_decide(state, ctx))
    assert not ctx.tool_calls  # Deciding does not execute a tool.
    state.update(research_execute(state, ctx))
    assert not state["evidence"]  # Search snippets are never accepted evidence.
    state.update(research_decide(state, ctx))
    assert isinstance(ctx.model_calls[1][1][-1], ToolMessage)
    assert "search lead" in ctx.model_calls[1][1][-1].content
    state.update(research_execute(state, ctx))
    assert state["evidence"][0]["status"] == "pending"
    assert state["evidence"][0]["readable"] is None and not ctx.readable
    state.update(research_decide(state, ctx))
    assert isinstance(ctx.model_calls[2][1][-1], ToolMessage)
    assert TEXT in ctx.model_calls[2][1][-1].content
    assert state["research_done"] and state["evidence"][0]["status"] == "accepted"
    assert ctx.readable == [("t0", "S2")]
    assert [item[1] for item in ctx.tool_calls] == ["search", "extract"]
    assert {tool["function"]["name"] for tool in ctx.model_calls[0][2]} == {"search_sources", "extract_page"}
    assert all("research:2:" in item[0] for item in ctx.model_calls + ctx.tool_calls)


@pytest.mark.parametrize("topic_count", [1, 2, 3])
def test_complete_pack_distribution_and_time_budget(topic_count):
    state = state_for(topic_count)
    state["evidence"] = [evidence_for(topic, f"S{i}") for i, topic in enumerate(state["selected_topic_ids"])]
    ctx = FakeContext(drafts=[{"assessments": [], "report": report_for(state)}])
    output = draft_pack(state, ctx)
    report = output["report"]
    assert report["status"] == "complete"
    assert len(report["questions"]) == 6
    assert all(sum(q["topic_id"] == topic for q in report["questions"]) == 6 // topic_count
               for topic in state["selected_topic_ids"])
    assert sum(item["minutes"] for item in report["study_plan"]) == 60
    assert not ctx.model_calls and not ctx.tool_calls  # Structured drafting binds no retrieval tools.
    assert ctx.draft_calls[0][2] is DraftResult


def test_final_draft_can_assess_last_page_without_an_extra_retrieval_decision():
    state = state_for()
    state["evidence"] = [evidence_for(status="pending")]
    ctx = FakeContext(drafts=[{"assessments": [assessment()], "report": report_for(state)}])
    ctx.retrieve = False
    state.update(research_decide(state, ctx))
    assert not ctx.model_calls and state["research_done"]
    output = draft_pack(state, ctx)
    assert output["report"]["status"] == "complete"
    assert output["evidence"][0]["status"] == "accepted"
    assert ctx.readable == [("t0", "S2")]


def test_wrong_topic_readable_page_is_rejected_then_alternative_is_chosen():
    state = state_for()
    state["evidence"] = [evidence_for(status="pending")]
    ctx = FakeContext([tool_message("search_sources", "better focused documentation", source_assessment=assessment(accepted=False))])
    output = research_decide(state, ctx)
    assert output["evidence"][0]["status"] == "rejected"
    assert output["evidence"][0]["reason"]
    assert output["pending_tool"]["name"] == "search_sources"
    assert output["evidence"][0]["readable"] is True and ctx.readable == [("t0", "S2")]


@pytest.mark.parametrize("readability", [None, "true", 1, False])
def test_readability_must_be_explicit_boolean_and_true_for_accepted_evidence(readability):
    state = state_for()
    state["evidence"] = [evidence_for(status="pending")]
    source_assessment = assessment(readable=readability)
    if readability is None:
        source_assessment.pop("readable")
    ctx = FakeContext([tool_message("search_sources", "primary topic guide", source_assessment=source_assessment)])
    ctx.repair_used = True
    output = research_decide(state, ctx)
    assert output["research_done"] and output["pending_tool"] is None and not ctx.readable
    assert state["evidence"][0]["status"] == "pending"


@pytest.mark.parametrize("bad_assessment", [None, assessment(excerpts=["Invented quote not in the actual source."])])
def test_pending_source_cannot_be_bypassed_or_given_fabricated_excerpt(bad_assessment):
    state = state_for()
    state["evidence"] = [evidence_for(status="pending")]
    ctx = FakeContext([tool_message("search_sources", "another source", source_assessment=bad_assessment)])
    ctx.repair_used = True
    output = research_decide(state, ctx)
    assert output["research_done"] and output["pending_tool"] is None
    assert state["evidence"][0]["status"] == "pending"


def test_mixed_exact_and_unmatched_excerpts_keep_only_grounded_support():
    state = state_for()
    state["evidence"] = [evidence_for(status="pending")]
    exact = TEXT[:100]
    decision = tool_message("search_sources", "another primary guide",
                            source_assessment=assessment(excerpts=[exact, "A fabricated claim that is not in the extracted page."]))
    ctx = FakeContext([decision])
    output = research_decide(state, ctx)
    source = output["evidence"][0]
    assert source["status"] == "accepted" and source["readable"] is True
    assert source["excerpt"] == exact.strip()
    assert "1 proposed excerpt(s) omitted" in source["reason"]
    assert ctx.readable == [("t0", "S2")]


def test_quote_matches_visible_words_across_markdown_link_markup():
    quote = "The simplest way to validate an instance is to use the validate function."
    source = Evidence.model_validate(evidence_for(status="pending"))
    source.text = "The simplest way to validate an instance is to use the [`validate`](../api/validate " \
                  '"validation docs") function. An unrelated note follows.'
    check = SourceAssessment(**assessment(excerpts=[quote, "A rewritten claim with different words is not present."]))
    _apply_assessment([source], check)
    assert source.status == "accepted"
    assert source.excerpt == quote
    assert "1 proposed excerpt(s) omitted" in source.reason


def test_saved_accepted_page_can_gain_only_verbatim_support_on_retry():
    state = state_for()
    source = Evidence.model_validate(evidence_for())
    source.excerpt = "Fixtures provide a reusable setup context for tests."
    exact = "Teardown releases resources after a test."
    proposed = SourceAssessment(**assessment(excerpts=[exact, "An unsupported invented sentence is absent."]))
    draft = DraftResult(assessments=[proposed], report=PrepReport.model_validate(report_for({
        **state, "evidence": [source.model_dump()]})))
    assessed = _assess_draft(draft, [source])[0]
    assert "Fixtures provide a reusable setup" in assessed.excerpt
    assert exact in assessed.excerpt
    assert "unsupported invented" not in assessed.excerpt
    assert "1 additional proposed excerpt(s) omitted" in assessed.reason


def test_saved_rejection_is_idempotent_but_cannot_be_revived():
    state = state_for()
    rejected = Evidence.model_validate(evidence_for(status="rejected"))
    rejected.readable = False
    rejected.excerpt = ""
    report = PrepReport.model_validate(report_for({**state, "evidence": [rejected.model_dump()]}))
    repeat = SourceAssessment(**assessment(accepted=False, readable=False, excerpts=[]))
    draft = DraftResult(assessments=[repeat], report=report)
    assert _assess_draft(draft, [rejected])[0].status == "rejected"
    draft.assessments[0].accepted = True
    with pytest.raises(ValueError, match="cannot be revived"):
        _assess_draft(draft, [rejected])


@pytest.mark.parametrize("saved_url,requested_url", [
    ("https://docs.example.com/guide", "https://docs.example.com/guide"),
    ("https://docs.example.com/guide", "https://docs.example.com/guide/#auth"),
    ("https://docs.example.com/guide/#setup", "https://docs.example.com/guide"),
    ("https://DOCS.EXAMPLE.COM:443/guide/", "https://docs.example.com/guide#auth"),
    ("https://docs.example.com./guide", "https://docs.example.com/guide/"),
    ("https://docs.example.com", "https://docs.example.com/#overview"),
])
def test_already_saved_page_is_not_extracted_or_charged_again(saved_url, requested_url):
    state = state_for()
    state["evidence"] = [evidence_for()]
    state["evidence"][0]["url"] = saved_url
    state["pending_tool"] = tool_message("extract_page", requested_url).tool_calls[0]
    ctx = FakeContext()
    output = research_execute(state, ctx)
    assert not ctx.tool_calls
    assert output["evidence"] == state["evidence"]
    assert "already extracted and saved" in output["research_messages"][-1].content
    assert ctx.events[-1][0] == "tool_skipped_duplicate"


@pytest.mark.parametrize("requested_url", [
    "https://docs.example.com/guide?version=2",
    "https://docs.example.com/Guide",
    "http://docs.example.com/guide",
    "https://other.example.com/guide",
])
def test_distinct_page_urls_are_not_collapsed_by_duplicate_protection(requested_url):
    state = state_for()
    state["evidence"] = [evidence_for()]
    state["pending_tool"] = tool_message("extract_page", requested_url).tool_calls[0]
    ctx = FakeContext()
    output = research_execute(state, ctx)
    assert len(ctx.tool_calls) == 1
    assert len(output["evidence"]) == 2 and output["evidence"][-1]["url"] == requested_url
    assert output["evidence"][-1]["status"] == "pending"


def test_normalized_duplicate_after_acceptance_uses_one_extraction_and_readable_charge(tmp_path):
    from interview_prep.config import Settings
    from interview_prep.context import RunContext
    from interview_prep.store import Store

    state = state_for()
    store = Store(tmp_path / "duplicate-source.sqlite")
    store.create(state["run_id"])
    calls = []
    saved_url = "https://DOCS.EXAMPLE.COM:443/guide/#setup"
    responses = [
        tool_message("extract_page", saved_url, call_id="original"),
        tool_message("extract_page", "https://docs.example.com/guide", call_id="duplicate",
                     source_assessment=assessment("S1")),
        AIMessage(content=json.dumps({"action": "draft", "reason": "Use the accepted saved source without repeating extraction."})),
    ]

    class CountedRetrieval(FakeRetrieval):
        def extract_page(self, url, timeout):
            calls.append(url)
            return super().extract_page(url, timeout)

    ctx = RunContext(store, state["run_id"], Settings(runtime_dir=tmp_path),
                     model=lambda messages, **kwargs: responses.pop(0), retrieval=CountedRetrieval())
    ctx.bind_state(state)
    ctx.begin_stage()
    state.update(research_decide(state, ctx))
    state.update(research_execute(state, ctx))
    state.update(research_decide(state, ctx))
    first = research_execute(state, ctx)
    replay = research_execute(state, ctx)  # Resume before the duplicate skip was checkpointed.
    assert first["evidence"] == replay["evidence"]
    state.update(replay)
    assert json.loads(state["research_messages"][-1].content)["status"] == "skipped"
    state.update(research_decide(state, ctx))
    counters = store.counters(state["run_id"])
    assert calls == [saved_url] and counters["extract_attempts"] == {"t0": 1}
    assert counters["readable_pages"] == {"t0": 1} and counters["transient_retries"] == 0
    assert len(state["evidence"]) == 1 and state["research_done"] is True
    source = state["evidence"][0]
    assert source["id"] == "S1" and source["status"] == "accepted"
    assert source["url"] == saved_url and source["excerpt"] == TEXT
    ctx.end_stage()


def test_same_page_for_another_topic_is_allowed_to_receive_its_own_assessment():
    state = state_for(2)
    state["evidence"] = [evidence_for(topic="t0")]
    state["pending_tool"] = tool_message("extract_page", "https://docs.example.com/guide#second-topic",
                                           topic="t1").tool_calls[0]
    ctx = FakeContext()
    output = research_execute(state, ctx)
    assert len(ctx.tool_calls) == 1
    assert output["evidence"][-1]["topic_id"] == "t1"
    assert output["evidence"][-1]["status"] == "pending"


def test_native_malformed_output_gets_only_shared_repair():
    state = state_for()
    ctx = FakeContext([AIMessage(content="not valid JSON"), tool_message("search_sources", "public documentation")])
    output = research_decide(state, ctx)
    assert ctx.repair_used and len(ctx.model_calls) == 2
    assert ctx.model_calls[1][0].endswith(":repair")
    assert output["pending_tool"]["name"] == "search_sources"


def test_failed_extraction_is_recorded_without_readable_count():
    state = state_for()
    ctx = FakeContext([tool_message("extract_page", "https://example.com/blocked")])
    ctx.retrieval.extract_page = lambda *args, **kwargs: {"ok": False, "status": "failed", "content_available": False, "error": "Blocked page"}
    state.update(research_decide(state, ctx))
    output = research_execute(state, ctx)
    assert len(ctx.tool_calls) == 1 and not ctx.readable
    assert output["evidence"][0]["status"] == "rejected"
    assert output["evidence"][0]["reason"] == "Blocked page"
    assert "Blocked page" in output["research_messages"][-1].content


@pytest.mark.parametrize("step", [research_decide, research_execute, draft_pack])
def test_stale_approval_blocks_every_research_step(step):
    state = state_for()
    state["approved_brief_version"] = 1
    ctx = FakeContext()
    with pytest.raises(ValueError, match="approval"):
        step(state, ctx)
    assert not ctx.model_calls and not ctx.tool_calls and not ctx.draft_calls


def test_changed_topic_selection_cannot_reuse_approval():
    state = state_for(2)
    state["selected_topic_ids"] = ["t0"]
    with pytest.raises(ValueError, match="selection changed"):
        research_decide(state, FakeContext())


def test_raw_resume_and_contacts_are_absent_from_research_context():
    state = state_for()
    state["resume_text"] = "private complete resume never send"
    state["jd_text"] = "unrelated complete job description never send"
    state["brief"]["skills"][0]["notes"] = "Contact person@example.com or 212-555-1234. Tutorial only."
    state["brief"]["skills"][0]["resume_evidence"] = "private candidate name and resume"
    ctx = FakeContext([tool_message("search_sources", "public topic guide")])
    research_decide(state, ctx)
    prompt = " ".join(str(message.content) for message in ctx.model_calls[0][1])
    assert "person@example.com" not in prompt and "212-555-1234" not in prompt
    assert "private complete resume" not in prompt and "private candidate name" not in prompt
    assert "Tutorial only" in prompt and "studied_tutorial" in prompt and "advanced" in prompt


@pytest.mark.parametrize("phase", ["research", "draft"])
def test_approved_summary_corrections_reach_both_coach_steps_with_contacts_redacted(phase):
    state = state_for()
    state["brief"]["summary"] = "Candidate correction: focus on fixture teardown rather than browser automation. Contact person@example.com or 212-555-1234."
    state["evidence"] = [evidence_for()]
    ctx = FakeContext(
        responses=[AIMessage(content=json.dumps({"action": "draft", "reason": "Saved evidence is sufficient for preparation."}))],
        drafts=[{"assessments": [], "report": report_for(state)}],
    )
    if phase == "research":
        research_decide(state, ctx)
        messages = ctx.model_calls[0][1]
    else:
        draft_pack(state, ctx)
        messages = ctx.draft_calls[0][1]
    prompt = " ".join(str(message.content) for message in messages)
    assert "Candidate correction: focus on fixture teardown rather than browser automation." in prompt
    assert "person@example.com" not in prompt and "212-555-1234" not in prompt


@pytest.mark.parametrize("defect", ["invented_source", "rejected_source", "wrong_topic", "missing_point_citation", "duplicate", "wrong_depth"])
def test_invalid_questions_are_omitted_without_padding(defect):
    state = state_for(2)
    state["evidence"] = [evidence_for("t0", "S0"), evidence_for("t1", "S1")]
    report = report_for(state)
    first = report["questions"][0]
    if defect == "invented_source":
        first["source_ids"] = ["invented"]
    elif defect == "rejected_source":
        state["evidence"][0]["status"] = "rejected"
    elif defect == "wrong_topic":
        first["source_ids"] = ["S1"]
    elif defect == "missing_point_citation":
        first["key_points"][0]["source_ids"] = []
    elif defect == "duplicate":
        report["questions"][1]["question"] = first["question"]
    else:
        first["depth"] = "beginner"
    clean, issues = _validate_report(PrepReport.model_validate(report), Brief.model_validate(state["brief"]),
                                     [Evidence.model_validate(item) for item in state["evidence"]])
    assert clean.status == "partial" and len(clean.questions) < 6 and issues


def test_excess_study_time_is_rejected():
    state = state_for()
    state["evidence"] = [evidence_for()]
    report = report_for(state)
    report["study_plan"][0]["minutes"] = 61
    clean, issues = _validate_report(PrepReport.model_validate(report), Brief.model_validate(state["brief"]),
                                     [Evidence.model_validate(item) for item in state["evidence"]])
    assert clean.status == "partial" and not clean.study_plan and len(clean.questions) == 6


def test_draft_failure_keeps_actual_evidence_and_explicit_partial_status():
    state = state_for()
    state["evidence"] = [evidence_for(status="pending")]
    ctx = FakeContext(drafts=[ProviderFailure("controlled failure")])
    output = draft_pack(state, ctx)
    assert output["report"]["status"] == "partial" and not output["report"]["questions"]
    assert output["evidence"][0]["text"] == TEXT
    assert output["evidence"][0]["status"] == "uncertain"
    assert output["report"]["gaps"]


def test_no_evidence_and_no_budget_is_failed_without_a_model_call():
    ctx = FakeContext()
    ctx.models = False
    output = draft_pack(state_for(), ctx)
    assert output["report"]["status"] == "failed"
    assert not ctx.draft_calls


def test_unknown_operation_propagates_for_deliberate_recovery():
    ctx = FakeContext([UnknownOperation("request outcome unknown")])
    with pytest.raises(UnknownOperation):
        research_decide(state_for(), ctx)


def test_honest_partial_for_missing_evidence_does_not_spend_repair():
    state = state_for()
    state["evidence"] = [evidence_for()]
    report = report_for(state)
    report["questions"] = report["questions"][:2]
    report["status"] = "partial"
    report["gaps"] = ["Only two sufficiently distinct questions are supported by these excerpts."]
    ctx = FakeContext(drafts=[{"assessments": [], "report": report}])
    output = draft_pack(state, ctx)
    assert output["report"]["status"] == "partial" and len(output["report"]["questions"]) == 2
    assert len(ctx.draft_calls) == 1 and not ctx.repair_used


def test_final_source_assessment_can_use_shared_semantic_repair():
    state = state_for()
    state["evidence"] = [evidence_for(status="pending")]
    report = report_for(state)
    ctx = FakeContext(drafts=[
        {"assessments": [assessment(excerpts=["This quotation is fabricated and should fail."])], "report": report},
        {"assessments": [assessment()], "report": report},
    ])
    output = draft_pack(state, ctx)
    assert output["report"]["status"] == "complete"
    assert ctx.repair_used and ctx.draft_calls[1][0].endswith(":semantic-repair")
    assert output["evidence"][0]["excerpt"] == TEXT


@pytest.mark.parametrize("readable,accepted", [(False, False), (True, False), (True, True)])
def test_final_draft_counts_only_explicit_readable_assessments(readable, accepted):
    state = state_for()
    state["evidence"] = [evidence_for(status="pending")]
    report = report_for(state)
    if not accepted:
        report.update(status="failed", questions=[], study_plan=[], gaps=["The source cannot support this topic."])
    ctx = FakeContext(drafts=[{"assessments": [assessment(readable=readable, accepted=accepted)], "report": report}])
    output = draft_pack(state, ctx)
    assert output["evidence"][0]["readable"] is readable
    assert ctx.readable == ([("t0", "S2")] if readable else [])
    assert output["report"]["status"] == ("complete" if accepted else "failed")


def test_draft_repair_charges_only_the_selected_final_readability_judgment():
    state = state_for()
    state["evidence"] = [evidence_for(status="pending")]
    unsupported = report_for(state)
    failed = {**unsupported, "status": "failed", "questions": [], "study_plan": [], "gaps": ["Only a blocked page is available."]}
    ctx = FakeContext(drafts=[
        {"assessments": [assessment(accepted=False, readable=True)], "report": unsupported},
        {"assessments": [assessment(accepted=False, readable=False)], "report": failed},
    ])
    output = draft_pack(state, ctx)
    assert len(ctx.draft_calls) == 2 and ctx.repair_used
    assert output["evidence"][0]["readable"] is False and not ctx.readable
    assert output["report"]["status"] == "failed"


def test_rejected_only_evidence_preserved_but_pack_is_failed():
    state = state_for()
    state["evidence"] = [evidence_for(status="rejected")]
    ctx = FakeContext()
    ctx.models = False
    output = draft_pack(state, ctx)
    assert output["report"]["status"] == "failed"
    assert output["evidence"][0]["status"] == "rejected"


def test_multiple_pending_sources_are_assessed_in_final_draft():
    state = state_for(2)
    state["evidence"] = [evidence_for("t0", "S0", "pending"), evidence_for("t1", "S1", "pending")]
    ctx = FakeContext(drafts=[{"assessments": [assessment("S0"), assessment("S1")], "report": report_for(state)}])
    output = draft_pack(state, ctx)
    assert output["report"]["status"] == "complete"
    assert all(item["status"] == "accepted" for item in output["evidence"])


def test_failed_semantic_repair_preserves_supported_first_draft():
    state = state_for()
    state["evidence"] = [evidence_for()]
    report = report_for(state)
    report["questions"][0]["source_ids"] = ["invented"]
    ctx = FakeContext(drafts=[{"assessments": [], "report": report}, ProviderFailure("controlled failure")])
    output = draft_pack(state, ctx)
    assert output["report"]["status"] == "partial"
    assert len(output["report"]["questions"]) == 5
    assert output["evidence"][0]["status"] == "accepted"


@pytest.mark.parametrize("saved_report", [False, True])
def test_failed_repair_or_saved_fallback_with_rejected_only_evidence_stays_failed(saved_report):
    state = state_for()
    state["evidence"] = [evidence_for(status="rejected")]
    report = report_for(state)
    ctx = FakeContext(drafts=[{"assessments": [], "report": report}, ProviderFailure("controlled repair failure")])
    if saved_report:
        state["report"] = report
        ctx.models = False
    output = draft_pack(state, ctx)
    assert output["report"]["status"] == "failed"
    assert not output["report"]["questions"] and not output["report"]["study_plan"]
    assert output["evidence"][0]["status"] == "rejected"
    assert output["evidence"][0]["text"] == TEXT
    assert output["report"]["gaps"]
    assert len(ctx.draft_calls) == (0 if saved_report else 2)


def test_completed_decision_is_reused_even_after_limits_reached():
    ctx = FakeContext([tool_message("search_sources", "public documentation")])
    ctx.models, ctx.retrieve, ctx.remaining = False, False, 0
    ctx.completed.add("research:2:decision:0")
    output = research_decide(state_for(), ctx)
    assert output["pending_tool"]["name"] == "search_sources"
    assert len(ctx.model_calls) == 1


def test_completed_draft_is_reused_after_stage_deadline():
    state = state_for()
    state["evidence"] = [evidence_for()]
    ctx = FakeContext(drafts=[{"assessments": [], "report": report_for(state)}])
    ctx.models, ctx.remaining = False, 0
    ctx.completed.add("research:2:draft:1")
    output = draft_pack(state, ctx)
    assert output["report"]["status"] == "complete"
    assert len(ctx.draft_calls) == 1


def test_completed_native_repair_uses_same_allowance_on_replay():
    ctx = FakeContext([AIMessage(content="invalid original response"), tool_message("search_sources", "public documentation")])
    ctx.completed.add("research:2:decision:0")
    ctx.models = False
    ctx.repair_used = True
    ctx.repair_key = "research:2:decision:0:repair"
    output = research_decide(state_for(), ctx)
    assert output["pending_tool"]["name"] == "search_sources"
    assert len(ctx.model_calls) == 2


def test_first_draft_after_brief_revision_handles_cleared_report():
    state = state_for()
    state["report"] = None
    state["evidence"] = [evidence_for()]
    ctx = FakeContext(drafts=[{"assessments": [], "report": report_for(state)}])
    assert draft_pack(state, ctx)["report"]["version"] == 1


def test_real_runtime_replays_completed_extraction_without_new_request_or_readable_charge(tmp_path):
    from interview_prep.config import Settings
    from interview_prep.context import RunContext
    from interview_prep.store import Store

    state = state_for()
    store = Store(tmp_path / "controlled.sqlite")
    store.create(state["run_id"])
    calls = []

    class CountedRetrieval(FakeRetrieval):
        def extract_page(self, url, timeout):
            calls.append(url)
            return super().extract_page(url, timeout)

    def model(messages, **kwargs):
        return tool_message("extract_page", "https://docs.example.com/guide")

    ctx = RunContext(store, state["run_id"], Settings(runtime_dir=tmp_path), model=model, retrieval=CountedRetrieval())
    ctx.bind_state(state)
    ctx.begin_stage()
    state.update(research_decide(state, ctx))
    ctx.bind_state(state)
    first = research_execute(state, ctx)
    store.add_elapsed(state["run_id"], 481)
    second = research_execute(state, ctx)  # Replay from the pre-tool graph checkpoint.
    counters = store.counters(state["run_id"])
    assert len(calls) == 1 and counters["extract_attempts"] == {"t0": 1}
    assert counters["readable_pages"] == {}
    assert first["evidence"][0]["id"] == second["evidence"][0]["id"]
    assert second["evidence"][0]["text"] == TEXT
    ctx.end_stage()


def test_real_runtime_assesses_unfamiliar_long_access_gate_before_counting_authentication_docs(tmp_path, monkeypatch):
    import socket
    from interview_prep.config import Settings
    from interview_prep.context import RunContext
    from interview_prep.retrieval import TavilyRetrieval
    from interview_prep.store import Store

    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))])
    state = state_for()
    store = Store(tmp_path / "access-gate.sqlite")
    store.create(state["run_id"])
    login = "# Workspace verification required\n" + "Your organization must authorize your identity before this page can be displayed. " * 40
    documentation = "# Sign in\nThis authentication guide describes password verification, session creation and HTTP-only cookies. Tests should cover successful login and rejected credentials."
    pages = [login, documentation]
    calls = []
    decisions = [tool_message("extract_page", "https://example.com/login", call_id="login"),
                 tool_message("extract_page", "https://example.com/auth-guide", call_id="guide",
                              source_assessment=assessment("S1", accepted=False, readable=False)),
                 AIMessage(content=json.dumps({"action": "draft", "reason": "The authentication guide provides substantive support.",
                                               "assessment": assessment("S2", excerpts=[documentation])}))]

    def transport(endpoint, body, headers, timeout):
        calls.append(body["urls"][0])
        return {"results": [{"url": body["urls"][0], "raw_content": pages.pop(0)}]}

    ctx = RunContext(store, state["run_id"], Settings(runtime_dir=tmp_path),
                     model=lambda messages, **kwargs: decisions.pop(0),
                     retrieval=TavilyRetrieval("controlled-key", transport=transport))
    ctx.bind_state(state)
    ctx.begin_stage()
    state.update(research_decide(state, ctx))
    state.update(research_execute(state, ctx))
    counters = store.counters(state["run_id"])
    assert counters["extract_attempts"] == {"t0": 1} and counters["readable_pages"] == {}
    assert len(login) > 2000 and state["evidence"][0]["status"] == "pending"
    assert state["evidence"][0]["readable"] is None
    state.update(research_decide(state, ctx))
    state.update(research_execute(state, ctx))
    counters = store.counters(state["run_id"])
    assert counters["extract_attempts"] == {"t0": 2} and counters["readable_pages"] == {}
    assert counters["transient_retries"] == 0 and len(calls) == 2
    assert state["evidence"][0]["status"] == "rejected" and state["evidence"][0]["readable"] is False
    assert state["evidence"][1]["status"] == "pending" and state["evidence"][1]["text"] == documentation
    assert "source_id" in state["research_messages"][-1].content
    state.update(research_decide(state, ctx))
    assert store.counters(state["run_id"])["readable_pages"] == {"t0": 1}
    assert state["evidence"][1]["status"] == "accepted" and state["evidence"][1]["readable"] is True
    ctx.end_stage()


def test_real_runtime_blocks_a_new_extraction_while_one_page_awaits_assessment(tmp_path):
    from interview_prep.config import Settings
    from interview_prep.context import RunContext
    from interview_prep.store import Store

    state = state_for()
    state["evidence"] = [evidence_for(source_id="S0", status="pending")]
    state["research_step"] = 1
    state["pending_tool"] = tool_message("extract_page", "https://example.com/next").tool_calls[0]
    store = Store(tmp_path / "pending-reservation.sqlite")
    store.create(state["run_id"])
    calls = []
    retrieval = FakeRetrieval()
    retrieval.extract_page = lambda *args, **kwargs: calls.append(args)
    ctx = RunContext(store, state["run_id"], Settings(runtime_dir=tmp_path), retrieval=retrieval)
    ctx.bind_state(state)
    output = research_execute(state, ctx)
    assert not calls and store.counters(state["run_id"])["extract_attempts"] == {}
    assert output["evidence"][0]["status"] == "pending"
    assert "pending page" in output["research_messages"][-1].content


def test_fourth_readable_assessment_blocks_fifth_extraction_before_provider_call(tmp_path):
    from interview_prep.config import Settings
    from interview_prep.context import RunContext
    from interview_prep.store import Store

    state = state_for()
    state["evidence"] = [evidence_for(source_id="S4", status="pending")]
    state["research_step"] = 4
    store = Store(tmp_path / "fourth-page.sqlite")
    store.create(state["run_id"])
    for index in range(3):
        store.readable(state["run_id"], f"earlier-{index}", "t0")
    for _ in range(4):
        store.charge(state["run_id"], "extract", "t0")
    calls = []
    retrieval = FakeRetrieval()
    retrieval.extract_page = lambda *args, **kwargs: calls.append(args)
    response = tool_message("extract_page", "https://example.com/fifth", source_assessment=assessment("S4"))
    ctx = RunContext(store, state["run_id"], Settings(runtime_dir=tmp_path),
                     model=lambda messages, **kwargs: response, retrieval=retrieval)
    ctx.bind_state(state)
    ctx.begin_stage()
    state.update(research_decide(state, ctx))
    assert store.counters(state["run_id"])["readable_pages"] == {"t0": 4}
    output = research_execute(state, ctx)
    assert not calls and store.counters(state["run_id"])["extract_attempts"] == {"t0": 4}
    assert "limit" in output["research_messages"][-1].content
    ctx.end_stage()


def test_completed_fourth_assessment_replays_after_model_and_time_limits_without_double_charge(tmp_path):
    from interview_prep.config import Settings
    from interview_prep.context import RunContext
    from interview_prep.store import Store

    state = state_for()
    state["evidence"] = [evidence_for(source_id="S4", status="pending")]
    store = Store(tmp_path / "assessment-replay.sqlite")
    store.create(state["run_id"])
    for index in range(3):
        store.readable(state["run_id"], f"earlier-{index}", "t0")
    for _ in range(29):
        store.charge(state["run_id"], "model")
    calls = []

    def model(messages, **kwargs):
        calls.append(messages)
        return AIMessage(content=json.dumps({"action": "draft", "reason": "The fourth source supplies sufficient support.",
                                            "assessment": assessment("S4")}))

    ctx = RunContext(store, state["run_id"], Settings(runtime_dir=tmp_path), model=model)
    ctx.bind_state(state)
    ctx.begin_stage()
    first = research_decide(state, ctx)
    assert store.counters(state["run_id"])["readable_pages"] == {"t0": 4}
    store.charge(state["run_id"], "model")
    store.charge(state["run_id"], "model")
    store.add_elapsed(state["run_id"], 481)
    replay = research_decide(state, ctx)  # Crash after count, before graph checkpoint.
    counters = store.counters(state["run_id"])
    assert counters["model_attempts"] == 32 and counters["readable_pages"] == {"t0": 4}
    assert len(calls) == 1 and replay["evidence"] == first["evidence"]
    assert replay["research_done"] and replay["error"] is None
    ctx.end_stage()


def test_real_runtime_replays_native_repair_after_crash_without_extra_model_attempts(tmp_path):
    from interview_prep.config import Settings
    from interview_prep.context import RunContext
    from interview_prep.store import Store

    state = state_for()
    store = Store(tmp_path / "repair.sqlite")
    store.create(state["run_id"])
    responses = [AIMessage(content="malformed decision"), tool_message("search_sources", "public documentation")]
    calls = []

    def model(messages, **kwargs):
        calls.append(messages)
        return responses.pop(0)

    ctx = RunContext(store, state["run_id"], Settings(runtime_dir=tmp_path), model=model, retrieval=FakeRetrieval())
    ctx.bind_state(state)
    ctx.begin_stage()
    first = research_decide(state, ctx)
    store.add_elapsed(state["run_id"], 481)
    second = research_decide(state, ctx)  # Replay before the decision checkpoint committed.
    counters = store.counters(state["run_id"])
    assert len(calls) == 2 and counters["model_attempts"] == 2 and counters["repairs"] == 1
    assert first["pending_tool"] == second["pending_tool"]
    ctx.end_stage()
