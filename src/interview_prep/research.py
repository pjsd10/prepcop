"""Research Coach graph steps: decide, execute one native tool, then draft.

The runtime context owns operation identity, deadlines, counters and the shared
retry/repair allowance. This module never calls a provider outside that context.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import Field, StrictBool, ValidationError

from .errors import BudgetExceeded, ProviderFailure, UnknownOperation
from .models import Brief, Evidence, PrepReport, StrictModel
from .retrieval import redact_contacts, validate_public_url


PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts"


class SourceAssessment(StrictModel):
    source_id: str
    readable: StrictBool
    accepted: bool
    reason: str = Field(min_length=12, max_length=1800)
    excerpts: list[str] = Field(default_factory=list, max_length=5)


class SearchDecision(StrictModel):
    topic_id: str
    query: str = Field(min_length=3, max_length=500)
    reason: str = Field(min_length=5, max_length=900)
    assessment: SourceAssessment | None = None


class ExtractDecision(StrictModel):
    topic_id: str
    url: str = Field(min_length=8, max_length=2048)
    reason: str = Field(min_length=5, max_length=900)
    assessment: SourceAssessment | None = None


class FinishDecision(StrictModel):
    action: Literal["draft"]
    reason: str = Field(min_length=5, max_length=900)
    assessment: SourceAssessment | None = None


class DraftResult(StrictModel):
    assessments: list[SourceAssessment] = Field(default_factory=list)
    report: PrepReport


RESEARCH_TOOLS = [
    {"type": "function", "function": {
        "name": "search_sources", "description": "Find public sources for one approved topic. Search snippets are leads, not evidence. Assess the pending extracted source in assessment before another retrieval.",
        "parameters": SearchDecision.model_json_schema(),
    }},
    {"type": "function", "function": {
        "name": "extract_page", "description": "Retrieve one public page for support assessment. Pick a suitable URL from search results or a known primary source. Assess the pending source first.",
        "parameters": ExtractDecision.model_json_schema(),
    }},
]


def _prompt(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8")


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _clean(value: str) -> str:
    return redact_contacts(value)


def _approved_brief(state: dict) -> Brief:
    brief = Brief.model_validate(state.get("brief"))
    if state.get("approved_brief_version") != brief.version:
        raise ValueError("Research requires approval of the current brief version.")
    selected = brief.selected_topic_ids
    skill_ids = {skill.id for skill in brief.skills}
    if len(selected) != len(set(selected)) or not set(selected) <= skill_ids:
        raise ValueError("Approved topics must be distinct skills in the current brief.")
    if "selected_topic_ids" in state and state["selected_topic_ids"] != selected:
        raise ValueError("Topic selection changed after brief approval.")
    return brief


def _brief_context(brief: Brief) -> dict:
    """Only role, selected requirements and relevant self-report go to research."""
    return {
        "version": brief.version, "role": _clean(brief.role), "seniority": _clean(brief.seniority),
        "summary": _clean(brief.summary),
        "interview_stage": _clean(brief.interview_stage),
        "preparation_minutes": brief.preparation_minutes,
        "selected_topic_ids": brief.selected_topic_ids,
        "question_targets": {topic: 6 // len(brief.selected_topic_ids) for topic in brief.selected_topic_ids},
        "skills": [{"id": skill.id, "name": _clean(skill.name), "requirement": _clean(skill.requirement),
                    "exposure": skill.exposure, "desired_depth": skill.desired_depth,
                    "notes": _clean(skill.notes)}
                   for skill in brief.skills if skill.id in brief.selected_topic_ids],
        "uncertainties": [_clean(item) for item in brief.uncertainties],
    }


def _evidence(state: dict) -> list[Evidence]:
    return [Evidence.model_validate(item) for item in state.get("evidence", [])]


def _source_context(evidence: list[Evidence], *, include_accepted_text: bool = False) -> list[dict]:
    result = []
    for item in evidence:
        record = item.model_dump()
        if item.status != "pending" and not (include_accepted_text and item.status == "accepted"):
            record.pop("text", None)
        result.append(record)
    return result


def _normal(value: str) -> str:
    return " ".join(value.split())


def _visible_markdown_text(value: str) -> str:
    """Remove presentation markup while retaining the extracted page's words."""
    value = re.sub(r"\[([^\]\n]+)\]\([^\n]*?\)", r"\1", value)
    return _normal(value.replace("`", "").replace("**", ""))


def _source_url_identity(url: str) -> tuple[str, str, int | None, str, str] | None:
    """Compare saved pages without changing citation URLs or meaningful queries."""
    try:
        parsed = urlsplit(validate_public_url(url))
    except (ValueError, TypeError):
        return None
    port = parsed.port
    if port == {"http": 80, "https": 443}.get(parsed.scheme):
        port = None
    # Fragments are removed by validation. Treat a trailing slash as the same
    # saved page, while preserving path case, protocol and query distinctions.
    return (parsed.scheme, (parsed.hostname or "").rstrip(".").lower(), port,
            parsed.path.rstrip("/") or "/", parsed.query)


def _matched_excerpts(source: Evidence, assessment: SourceAssessment) -> tuple[list[str], int]:
    proposed = [part.strip() for part in assessment.excerpts if part.strip()]
    page_text = _visible_markdown_text(source.text)
    excerpts = [part for part in proposed if len(part) <= 5000
                and (not assessment.accepted or len(part) >= 20)
                and _visible_markdown_text(part) in page_text]
    return excerpts, len(proposed) - len(excerpts)


def _apply_assessment(evidence: list[Evidence], assessment: SourceAssessment | None, *, require_complete: bool = True) -> None:
    pending = [item for item in evidence if item.status == "pending"]
    if assessment is None:
        if pending:
            raise ValueError("Assess the pending extracted page before continuing.")
        return
    source = next((item for item in evidence if item.id == assessment.source_id), None)
    if source is None or source.status != "pending":
        raise ValueError("Assessment must identify a pending extracted source.")
    if assessment.accepted and not assessment.readable:
        raise ValueError("Accepted evidence must be assessed as readable substantive page content.")
    excerpts, discarded = _matched_excerpts(source, assessment)
    if assessment.accepted and not excerpts:
        raise ValueError("Accepted evidence needs at least one substantive verbatim excerpt from the extracted page.")
    source.status = "accepted" if assessment.accepted else "rejected"
    source.readable = assessment.readable
    source.reason = (assessment.reason + (f" {discarded} proposed excerpt(s) omitted because they were not usable verbatim support."
                                          if discarded else ""))[:1800]
    source.excerpt = "\n\n[…]\n\n".join(excerpts)
    if require_complete and any(item.status == "pending" for item in evidence):
        raise ValueError("All pending pages need support assessments before continuing.")


def _assess_draft(draft: DraftResult, original: list[Evidence]) -> list[Evidence]:
    assessed = [item.model_copy(deep=True) for item in original]
    for assessment in draft.assessments:
        source = next((item for item in assessed if item.id == assessment.source_id), None)
        if source is not None and source.status == "rejected":
            if assessment.accepted or assessment.readable != source.readable or assessment.excerpts:
                raise ValueError("A rejected source cannot be revived or given supporting excerpts.")
            continue
        if source is not None and source.status == "accepted":
            if not assessment.accepted or assessment.readable is not True:
                raise ValueError("An accepted source can only gain exact supporting excerpts.")
            excerpts, discarded = _matched_excerpts(source, assessment)
            if not excerpts:
                raise ValueError("Accepted-source expansion needs a substantive verbatim excerpt.")
            retained = [part for part in source.excerpt.split("\n\n[…]\n\n") if part]
            source.excerpt = "\n\n[…]\n\n".join(dict.fromkeys([*retained, *excerpts]))
            if discarded:
                source.reason = (source.reason + f" {discarded} additional proposed excerpt(s) omitted as unmatched.")[:1800]
        else:
            _apply_assessment(assessed, assessment, require_complete=False)
    if any(item.status == "pending" for item in assessed):
        raise ValueError("Draft omitted support assessment of a pending page.")
    return assessed


def _record_assessed_pages(original: list[Evidence], assessed: list[Evidence], ctx) -> None:
    pending_ids = {item.id for item in original if item.status == "pending"}
    for source in assessed:
        if source.id in pending_ids and source.readable is True and source.status in ("accepted", "rejected"):
            # Stable source identity makes replay after a pre-checkpoint crash safe.
            ctx.record_readable(source.topic_id, source.id)


def _parse_content(message: AIMessage) -> dict:
    text = message.content
    if isinstance(text, list):
        text = "".join(item.get("text", "") for item in text if isinstance(item, dict))
    if not isinstance(text, str):
        raise ValueError("Research decision must be JSON or a native tool call.")
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Research decision must be a JSON object.")
    return value


def _decision(message: AIMessage, brief: Brief, original: list[Evidence]) -> tuple[dict | None, list[Evidence], str]:
    evidence = [item.model_copy(deep=True) for item in original]
    calls = message.tool_calls
    if getattr(message, "invalid_tool_calls", None):
        raise ValueError("The native tool-call arguments could not be parsed.")
    if len(calls) > 1:
        raise ValueError("Choose exactly one retrieval tool at a time.")
    if not calls:
        finish = FinishDecision.model_validate(_parse_content(message))
        _apply_assessment(evidence, finish.assessment)
        return None, evidence, finish.reason
    call = calls[0]
    schema = {"search_sources": SearchDecision, "extract_page": ExtractDecision}.get(call["name"])
    if schema is None:
        raise ValueError("Only search_sources and extract_page are permitted.")
    decision = schema.model_validate(call["args"])
    if decision.topic_id not in brief.selected_topic_ids:
        raise ValueError("Research tools must target an approved selected topic.")
    _apply_assessment(evidence, decision.assessment)
    if not call.get("id"):
        raise ValueError("Native tool call is missing its tool-call ID.")
    pending = {"name": call["name"], "id": call["id"], "args": decision.model_dump()}
    return pending, evidence, decision.reason


def _stop(state: dict, reason: str) -> dict:
    return {"pending_tool": None, "research_done": True, "stage": "drafting",
            "error": reason, "research_step": state.get("research_step", 0) + 1}


def research_decide(state: dict, ctx) -> dict:
    """One model decision; model tool requests are executed by the next node."""
    brief = _approved_brief(state)
    if state.get("research_done"):
        return {"pending_tool": None, "stage": "drafting"}
    step = state.get("research_step", 0)
    key = f"research:{brief.version}:decision:{step}"
    replaying = ctx.completed_operation(key)
    if not replaying and (not ctx.can_model(reserve=2) or ctx.remaining_stage_seconds() <= 0):
        return _stop(state, "Research stopped to preserve finalization resources.")
    allowed = {topic: {kind: ctx.can_retrieve(kind, topic) for kind in ("search", "extract")}
               for topic in brief.selected_topic_ids}
    if not replaying and not any(kinds["extract"] for kinds in allowed.values()):
        return _stop(state, "No extraction slots remain; drafting uses the evidence already saved.")
    original = _evidence(state)
    messages = [SystemMessage(content=_prompt("research_coach.txt")),
                HumanMessage(content=_json({"approved_brief": _brief_context(brief),
                                           "allowed_retrieval": allowed,
                                           "evidence": _source_context(original),
                                           "candidate_sources": state.get("candidates", [])})),
                *state.get("research_messages", [])]
    try:
        message = ctx.model_call(key, messages, tools=RESEARCH_TOOLS, max_tokens=1800)
        try:
            pending, evidence, reason = _decision(message, brief, original)
        except (ValueError, ValidationError, KeyError, TypeError) as exc:
            if not ctx.consume_repair(key + ":repair"):
                return _stop(state, "Research decision was malformed; the shared repair allowance is unavailable.")
            repair = messages + [HumanMessage(content="Repair your previous decision. Validation error: " + str(exc)[:900]
                                              + ". Return exactly one valid native tool call or the required finish JSON. Assess the pending page.")]
            message = ctx.model_call(key + ":repair", repair, tools=RESEARCH_TOOLS, max_tokens=1800)
            pending, evidence, reason = _decision(message, brief, original)
        _record_assessed_pages(original, evidence, ctx)
    except UnknownOperation:
        raise
    except (BudgetExceeded, ProviderFailure, ValueError, ValidationError, KeyError, TypeError):
        return _stop(state, "Research decision could not finish; saved evidence remains available.")
    ctx.event("research_decision", {"step": step, "action": pending["name"] if pending else "draft", "reason": reason})
    for before, after in zip(original, evidence):
        if before.status != after.status:
            ctx.event("source_assessment", {"source_id": after.id, "status": after.status,
                                            "readable": after.readable, "reason": after.reason})
    return {"research_messages": [message], "evidence": [item.model_dump() for item in evidence],
            "pending_tool": pending, "research_done": pending is None,
            "research_step": step + 1, "stage": "researching" if pending else "drafting", "error": None}


def research_execute(state: dict, ctx) -> dict:
    """Execute only the saved native tool call and return its result to the Coach."""
    brief = _approved_brief(state)
    pending = state.get("pending_tool")
    if not pending:
        return {"pending_tool": None}
    name, args = pending["name"], pending["args"]
    if name not in ("search_sources", "extract_page") or args.get("topic_id") not in brief.selected_topic_ids:
        raise ValueError("The pending research tool is not permitted by the approved brief.")
    topic = args["topic_id"]
    kind = "search" if name == "search_sources" else "extract"
    step = state.get("research_step", 0)
    key = f"research:{brief.version}:tool:{step}:{pending['id']}"
    evidence = _evidence(state)
    candidates = list(state.get("candidates", []))
    requested_url = _source_url_identity(args["url"]) if name == "extract_page" else None
    duplicate_saved_page = (name == "extract_page" and not ctx.completed_operation(key)
                            and requested_url is not None
                            and any(item.topic_id == topic and _source_url_identity(item.url) == requested_url and item.text.strip()
                                    for item in evidence))
    if duplicate_saved_page:
        result = {"ok": False, "status": "skipped", "content_available": False,
                  "error": "This page was already extracted and saved. Choose a different URL or draft from the saved source."}
        ctx.event("tool_skipped_duplicate", {"operation_id": key, "topic_id": topic})
    else:
        try:
            if (name == "extract_page" and not ctx.completed_operation(key)
                    and any(item.status == "pending" for item in evidence)):
                # One pending candidate reserves the next readable slot. Its assessment
                # must settle before another extraction can start, even across topics.
                raise BudgetExceeded("Assess the pending page before extracting another page.")
            if name == "search_sources":
                execute = lambda timeout: ctx.retrieval.search_sources(args["query"], timeout=timeout)
            else:
                execute = lambda timeout: ctx.retrieval.extract_page(args["url"], timeout=timeout)
            result = ctx.execute_tool(key, kind, topic, args, execute)
        except UnknownOperation:
            raise
        except BudgetExceeded as exc:
            result = {"ok": False, "status": "failed", "error": str(exc)}
        except ProviderFailure:
            result = {"ok": False, "status": "failed", "error": "Retrieval stopped by provider failure."}
    if not isinstance(result, dict):
        result = {"ok": False, "status": "failed", "error": "Retrieval returned an invalid result."}
    tool_result = dict(result)
    if result.get("ok") is True and name == "search_sources":
        for item in result.get("results", []):
            candidate = {**item, "topic_id": topic}
            if not any(old.get("url") == item.get("url") and old.get("topic_id") == topic for old in candidates):
                candidates.append(candidate)
    elif name == "extract_page" and not duplicate_saved_page:
        source_id = f"S{step}"
        # Stable across checkpoint recovery: the same operation yields the same ID.
        existing = next((item for item in evidence if item.id == source_id), None)
        if existing is None:
            extracted = bool(result.get("ok") and result.get("results")
                             and isinstance(result["results"][0].get("raw_content"), str)
                             and result["results"][0]["raw_content"].strip())
            item = result["results"][0] if extracted else {}
            url = item.get("url", args["url"])
            candidate = next((entry for entry in candidates if entry.get("url") == url), {})
            source = Evidence(id=source_id, topic_id=topic,
                              title=item.get("title") or candidate.get("title") or "Retrieved page",
                              url=url, text=item.get("raw_content", ""),
                              retrieved_at=datetime.now(timezone.utc).isoformat(),
                              status="pending" if extracted else "rejected", readable=None if extracted else False,
                              reason="Awaiting Coach readability and support assessment." if extracted else result.get("error", "Page unavailable."))
            evidence.append(source)
        tool_result["source_id"] = source_id
    history = list(state.get("research_messages", []))
    history.append(ToolMessage(content=_json(tool_result), tool_call_id=pending["id"], name=name))
    return {"research_messages": history, "pending_tool": None, "candidates": candidates,
            "evidence": [item.model_dump() for item in evidence], "stage": "researching"}


def _validate_report(report: PrepReport, brief: Brief, evidence: list[Evidence]) -> tuple[PrepReport, list[str]]:
    """Validate identity/coverage, never imply that reference IDs prove truth."""
    report = report.model_copy(deep=True)
    problems = []
    accepted = {item.id: item for item in evidence if item.status == "accepted" and item.excerpt.strip()}
    selected = brief.selected_topic_ids
    quota = 6 // len(selected)
    counts, ids, texts, questions = Counter(), set(), set(), []
    depths = {item.id: item.desired_depth for item in brief.skills}
    if report.brief_version != brief.version:
        problems.append("Draft referenced a different brief version.")
    report.brief_version = brief.version

    def supported(ids_to_check, topic):
        return bool(ids_to_check) and all(source_id in accepted and accepted[source_id].topic_id == topic for source_id in ids_to_check)

    for question in report.questions:
        text_key = re.sub(r"\W+", " ", question.question.casefold()).strip()
        valid = (question.topic_id in selected and question.id not in ids and text_key and text_key not in texts
                 and counts[question.topic_id] < quota and question.explanation.strip()
                 and question.depth == depths.get(question.topic_id) and question.key_points
                 and supported(question.source_ids, question.topic_id)
                 and all(point.text.strip() and supported(point.source_ids, question.topic_id) for point in question.key_points))
        if not valid:
            problems.append(f"Omitted question {question.id}: unsupported, duplicate, wrong depth, or outside topic allocation.")
            continue
        questions.append(question)
        counts[question.topic_id] += 1
        ids.add(question.id)
        texts.add(text_key)
    report.questions = questions
    plan = []
    for activity in report.study_plan:
        if activity.topic_id in selected and activity.activity.strip() and supported(activity.source_ids, activity.topic_id):
            plan.append(activity)
        else:
            problems.append("Omitted a study activity without accepted evidence for its topic.")
    report.study_plan = plan
    total = sum(item.minutes for item in plan)
    if total > brief.preparation_minutes:
        problems.append("Study-plan minutes exceeded the approved time budget; invalid plan omitted.")
        report.study_plan = []
        total = 0
    if total != brief.preparation_minutes:
        problems.append(f"Study plan allocates {total} of {brief.preparation_minutes} approved minutes.")
    if {item.topic_id for item in report.study_plan} != set(selected):
        problems.append("Study plan does not cover every selected topic.")
    for topic in selected:
        if counts[topic] != quota:
            problems.append(f"Topic {topic} has {counts[topic]} of {quota} required supported questions.")
    report.status = "complete" if len(questions) == 6 and not problems and report.status == "complete" else ("partial" if questions or accepted else "failed")
    if report.status != "complete":
        report.gaps = list(dict.fromkeys([*report.gaps, *problems]))
    return report, problems


def _has_usable_evidence(evidence: list[Evidence]) -> bool:
    return any(item.status in ("accepted", "pending", "uncertain") and item.text.strip() for item in evidence)


def _fallback_report(state: dict, brief: Brief, evidence: list[Evidence], reason: str) -> PrepReport:
    # Preserve a previously valid draft if final repair fails.
    saved = state.get("report")
    if saved:
        try:
            report, _ = _validate_report(PrepReport.model_validate(saved), brief, evidence)
            report.gaps = list(dict.fromkeys([*report.gaps, reason]))
            report.status = "partial" if report.questions or _has_usable_evidence(evidence) else "failed"
            return report
        except (ValueError, ValidationError):
            pass
    return PrepReport(version=1, brief_version=brief.version, status="partial" if _has_usable_evidence(evidence) else "failed",
                      title="PrepCop — incomplete", summary="Research could not produce a validated preparation pack.",
                      gaps=[reason, "Six supported questions and the study plan were not completed."])


def draft_pack(state: dict, ctx) -> dict:
    """Draft with no retrieval tools; retain supported work on failure or limits."""
    brief = _approved_brief(state)
    evidence = _evidence(state)
    unassessed = [item.model_copy(deep=True) for item in evidence]
    version = int((state.get("report") or {}).get("version", 0)) + 1
    messages = [SystemMessage(content=_prompt("research_draft.txt")),
                HumanMessage(content=_json({"approved_brief": _brief_context(brief), "report_version": version,
                                           "evidence": _source_context(evidence, include_accepted_text=bool(state.get("source_expansion_requested"))),
                                           "stopping_reason": state.get("error")}))]
    key = f"research:{brief.version}:draft:{version}"
    report = None
    failure = None
    try:
        if not ctx.completed_operation(key) and (not ctx.can_model() or ctx.remaining_stage_seconds() <= 0):
            raise BudgetExceeded("Drafting budget exhausted.")
        draft = ctx.structured_call(key, messages, DraftResult)
        draft = DraftResult.model_validate(draft)
        original = [item.model_copy(deep=True) for item in evidence]
        try:
            assessed = _assess_draft(draft, original)
            report, problems = _validate_report(draft.report, brief, assessed)
            evidence = assessed
            # Missing evidence in an honest partial pack is not malformed output.
            needs_repair = bool(problems) and (draft.report.status == "complete" or any(
                problem.startswith(("Omitted", "Draft referenced", "Study-plan minutes exceeded")) for problem in problems))
        except ValueError as exc:
            problems, needs_repair = [str(exc)], True
        repair_key = key + ":semantic-repair"
        if needs_repair and ctx.consume_repair(repair_key):
            repair_messages = messages + [HumanMessage(content="Repair this draft using the same saved evidence. Validation issues: "
                                                       + _json(problems) + ". Previous draft: " + draft.model_dump_json())]
            repaired = ctx.structured_call(repair_key, repair_messages, DraftResult)
            repaired = DraftResult.model_validate(repaired)
            repaired_evidence = _assess_draft(repaired, original)
            repaired_report, _ = _validate_report(repaired.report, brief, repaired_evidence)
            if report is None or len(repaired_report.questions) >= len(report.questions):
                report = repaired_report
                evidence = repaired_evidence
        elif report is None:
            failure = "Support assessment was invalid and the shared repair allowance is unavailable."
    except UnknownOperation:
        raise
    except (BudgetExceeded, ProviderFailure, ValueError, ValidationError, TypeError):
        failure = "Final drafting or validation could not finish within the shared resource/repair allowance. Saved evidence is retained."
    if report is None:
        report = _fallback_report(state, brief, evidence, failure or "No validated report was produced.")
    elif failure:
        report.status = "partial" if report.questions or _has_usable_evidence(evidence) else "failed"
        report.gaps = list(dict.fromkeys([*report.gaps, failure]))
    # A semantic repair can revise the provisional assessment. Charge only the
    # selected final evidence, after repair, with no retrieval tools available.
    _record_assessed_pages(unassessed, evidence, ctx)
    for source in evidence:
        if source.status == "pending":
            source.status = "uncertain"
            source.reason = "Extracted page retained; readability and support assessment could not be completed."
    report.version = version
    if state.get("error") and report.status != "complete":
        report.gaps = list(dict.fromkeys([*report.gaps, state["error"]]))
    ctx.event("report_drafted", {"status": report.status, "question_count": len(report.questions), "version": version})
    return {"report": report.model_dump(), "evidence": [item.model_dump() for item in evidence],
            "research_done": True, "pending_tool": None, "stage": "review_report", "error": None,
            "report_reviewed_version": None}
