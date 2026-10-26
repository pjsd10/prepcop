"""Native Streamlit interaction layer; saved workflow state owns approvals."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import re
from typing import Any, Callable
from uuid import uuid4

import streamlit as st

from .html_input import (
    InputValidationError, MAX_JD_CHARS, MAX_RESUME_CHARS,
    parse_resume_html, validate_input_text,
)
from .report import safe_source_url


EXPOSURES = {
    "not_used": "Not used",
    "studied_tutorial": "Studied / tutorial",
    "personal_course_project": "Personal / course project",
    "used_at_work": "Used at work",
    "not_sure": "Not sure",
}
DEPTHS = {
    "beginner": "Beginner", "intermediate": "Intermediate",
    "advanced": "Advanced", "not_sure": "Not sure",
}
STAGES = ("Your role", "Your experience", "Review and research", "Prep pack")


def escape_markdown(text: Any) -> str:
    """Display untrusted prose as prose, without image/link instructions."""
    value = str(text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return re.sub(r"([\\`*_{}\[\]()#+.!|~-])", r"\\\1", value)


def _write(text: Any) -> None:
    if text:
        st.markdown(escape_markdown(text))


def _write_saved_summary(text: Any) -> None:
    """Keep candidate-authored bullet lines readable without trusting Markdown."""
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if lines and all(line.startswith("- ") for line in lines):
        st.markdown("\n".join("- " + escape_markdown(line[2:]) for line in lines))
    else:
        _write(text)


def _caption(text: Any) -> None:
    st.caption(escape_markdown(text))


def _heading(text: Any, level: int = 3) -> None:
    st.markdown("#" * level + " " + escape_markdown(text))


def _attempt(label: str, callback: Callable[[], Any]) -> Any | None:
    try:
        with st.spinner(label):
            return callback()
    except (InputValidationError, ValueError) as exc:
        st.error(escape_markdown(str(exc)))
    except Exception as exc:
        st.error("This step could not finish. Your saved progress is available through your resume code. "
                 "Check setup or use Resume saved run before trying again.")
        with st.expander("Troubleshooting details"):
            st.text(f"Error category: {type(exc).__name__}")
    return None


def progress_message(kind: str, details: dict) -> str | None:
    """Allowlisted factual UI updates, never arbitrary provider text."""
    if kind == "research_decision":
        return {
            "search_sources": "Searching for relevant public sources…",
            "extract_page": "Reading a selected source…",
            "draft": "Preparing the report from saved evidence…",
        }.get(details.get("action"))
    if kind == "tool_completed":
        if details.get("status") != "success":
            return "A source request did not return usable results. Its outcome was saved."
        return "Source search saved." if details.get("kind") == "search" else "Retrieved page saved for support assessment."
    if kind == "source_assessment":
        return {
            "accepted": "A source was accepted as supporting evidence.",
            "rejected": "A source was unsuitable for this preparation pack.",
            "uncertain": "A source was retained with uncertain support.",
        }.get(details.get("status"))
    if kind == "report_drafted":
        status = details.get("status")
        return {"complete": "The complete pack is saved for your review.",
                "partial": "The partial pack and its gaps are saved for your review.",
                "failed": "Available evidence and the incomplete outcome are saved."}.get(status)
    if kind == "model_completed":
        return "A model response was saved."
    if kind in {"model_failed", "tool_failed"}:
        return "A request could not finish. Its outcome was saved."
    if kind == "operation_unknown":
        return "A request outcome is uncertain. Your saved progress needs a decision."
    return None


def _engine_step(engine: Any, label: str, callback: Callable[[], Any]) -> Any | None:
    with st.status(label, expanded=True) as progress:
        previous_observer = getattr(engine, "on_event", None)

        def show_event(kind: str, details: dict) -> None:
            message = progress_message(kind, details)
            if message:
                progress.write(message)
                progress.update(label=message)

        engine.on_event = show_event
        try:
            result = _attempt(label, callback)
        finally:
            engine.on_event = previous_observer
        needs_decision = isinstance(result, dict) and (result.get("pending") or {}).get("kind") == "operation_unknown"
        if result is None:
            progress.update(label="This step needs attention. Saved progress is retained.", state="error")
        elif needs_decision:
            progress.update(label="Saved progress needs your decision.", state="error")
        else:
            progress.update(label="Saved progress updated.", state="complete", expanded=False)
        return result


def _submit(engine: Any, state: dict, event: dict, label: str) -> None:
    # Event IDs survive a failed/unknown submission. Retrying the same payload
    # reuses its identifier; accepting a different action creates another ID.
    payload = json.dumps(event, sort_keys=True, ensure_ascii=False)
    signature = sha256((state["run_id"] + payload).encode()).hexdigest()
    pending = st.session_state.get("submission")
    if not pending or pending["signature"] != signature:
        pending = {"signature": signature, "id": str(uuid4())}
        st.session_state["submission"] = pending
    result = _engine_step(engine, label, lambda: engine.submit(state["run_id"], event, pending["id"]))
    if result is not None:
        st.session_state.pop("submission", None)
        st.rerun()


def _stage_index(state: dict | None) -> int:
    if not state:
        return 0
    kind = (state.get("pending") or {}).get("kind", "")
    if state.get("report") or kind == "report_review":
        return 3
    if state.get("brief") or kind in {"brief_review", "operation_unknown"}:
        return 2
    return 1


def _stage_strip(state: dict | None) -> None:
    active = _stage_index(state)
    for index, column in enumerate(st.columns(4)):
        with column:
            if index == active:
                st.markdown(f"**{index + 1}. {STAGES[index]}**")
            else:
                st.caption(f"{index + 1}. {STAGES[index]}")
    st.divider()


def _sidebar(engine: Any, readiness: list[str]) -> None:
    with st.sidebar:
        st.subheader("Your saved run")
        run_id = st.session_state.get("run_id")
        if run_id:
            st.caption("Keep this private code to resume after a refresh or restart.")
            st.code(run_id, language=None)
        with st.form("resume_saved_run"):
            token = st.text_input("Resume code", type="password", autocomplete="off")
            resumed = st.form_submit_button("Resume saved run")
        if resumed:
            result = _attempt("Opening saved progress…", lambda: engine.get_run(token.strip()))
            if result is not None:
                st.session_state["run_id"] = token.strip()
                st.session_state.pop("submission", None)
                st.rerun()
        if run_id and st.button("Start a new run", width="stretch"):
            for key in list(st.session_state):
                if key not in {"engine"}:
                    del st.session_state[key]
            st.rerun()
        st.caption("Progress is stored locally on the machine running this app. "
                   "A new run starts with new inputs.")
        if readiness:
            with st.expander("Setup needed", expanded=True):
                for message in readiness:
                    _write(message)
        else:
            st.caption("Provider settings are present. Access is checked when a request runs.")


def _inputs(engine: Any, readiness: list[str]) -> None:
    st.header("Tell us about the role")
    st.write("Upload a static HTML resume (.html or .htm). Word and PDF uploads are not supported "
             "in this pilot. Review the extracted text before continuing.")
    if readiness:
        st.info("Setup is needed before preparing a profile. Open Setup needed in the sidebar for the required settings.")
    uploaded = st.file_uploader("Your resume", type=["html", "htm"], key="resume_upload",
                                help="Static HTML up to 1 MB. Text is extracted locally for your review.")
    extraction_ok = False
    if uploaded is None and st.session_state.get("upload_digest") is not None:
        st.session_state.pop("upload_digest", None)
        st.session_state["resume_preview"] = ""
        st.session_state["input_confirmed"] = False
    if uploaded is not None:
        raw = uploaded.getvalue()
        digest = sha256(raw).hexdigest()
        if st.session_state.get("upload_digest") != digest:
            try:
                preview = parse_resume_html(raw, uploaded.name)
            except InputValidationError as exc:
                st.error(escape_markdown(str(exc)))
                st.session_state.pop("upload_digest", None)
                return
            st.session_state["upload_digest"] = digest
            st.session_state["resume_preview"] = preview
            st.session_state["input_confirmed"] = False
        extraction_ok = True
    resume = st.text_area("Review and correct the extracted resume", key="resume_preview", height=260,
                          max_chars=MAX_RESUME_CHARS, disabled=not extraction_ok)
    jd = st.text_area("Paste the job description", height=230, max_chars=MAX_JD_CHARS)
    left, right = st.columns(2)
    minutes = left.number_input("Study time available (minutes)", min_value=10, max_value=600, value=60, step=10)
    interview_stage = right.text_input("Interview stage", value="Not specified",
                                        placeholder="For example, technical screen or panel")
    confirmed = st.checkbox("I reviewed the resume text and job description.", key="input_confirmed")
    consent = st.checkbox("Use these confirmed inputs with the configured AI service to prepare my profile. "
                          "Research will use public web sources after I approve my brief.")
    started = st.button("Review my profile", type="primary",
                        disabled=not extraction_ok or bool(readiness) or not confirmed or not consent)
    if started:
        if not confirmed or not consent:
            st.warning("Confirm the text and service use before continuing.")
            return
        try:
            resume = validate_input_text(resume, "Resume", max_chars=MAX_RESUME_CHARS)
            jd = validate_input_text(jd, "Job description", max_chars=MAX_JD_CHARS)
        except InputValidationError as exc:
            st.error(escape_markdown(str(exc)))
            return
        run_id = _attempt("Saving your confirmed inputs…", lambda: engine.create_run(
            resume, jd, preparation_minutes=int(minutes), interview_stage=interview_stage.strip() or "Not specified"
        ))
        if run_id is not None:
            st.session_state["run_id"] = run_id
            _engine_step(engine, "Mapping the role and your experience…", lambda: engine.advance(run_id))
            st.rerun()


def _skill_editor(skills: list[dict], *, key: str) -> list[dict]:
    rows = [{
        "id": skill["id"], "name": skill["name"],
        "exposure": EXPOSURES[skill.get("exposure", "not_sure")],
        "desired_depth": DEPTHS[skill.get("desired_depth", "not_sure")],
        "notes": skill.get("notes", ""),
    } for skill in skills]
    edited = st.data_editor(
        rows, key=key, hide_index=True, width="stretch", num_rows="fixed",
        column_config={
            "id": None,
            "name": st.column_config.TextColumn("Skill / topic", required=True),
            "exposure": st.column_config.SelectboxColumn("Actual exposure", options=list(EXPOSURES.values()), required=True),
            "desired_depth": st.column_config.SelectboxColumn("Practice depth", options=list(DEPTHS.values()), required=True),
            "notes": st.column_config.TextColumn("Corrections or notes", width="large"),
        },
    )
    original = {skill["id"]: skill for skill in skills}
    exposure_keys = {label: value for value, label in EXPOSURES.items()}
    depth_keys = {label: value for value, label in DEPTHS.items()}
    return [{**original[row["id"]], "name": str(row["name"] or "").strip(),
             "exposure": exposure_keys[row["exposure"]], "desired_depth": depth_keys[row["desired_depth"]],
             "notes": str(row["notes"] or "")} for row in edited]


def _selection(skills: list[dict], selected: list[str], *, key: str) -> list[str]:
    names = {skill["id"]: skill["name"] for skill in skills}
    st.markdown("**Beta · Choose up to three skills you most want to prepare.**")
    st.caption("Select one, two or three priorities. Other skills stay deferred. "
               "Start a new run later to prepare another set.")
    selection = st.multiselect("Research priorities", options=list(names), default=selected,
                               format_func=lambda identity: names[identity], max_selections=3, key=key)
    st.caption("Selected chips above show your current choices; submit this form to save them.")
    deferred = [skill["name"] for skill in skills if skill["id"] not in selection]
    if deferred:
        _caption("Deferred this run: " + ", ".join(deferred))
    return selection


def _profile_review(engine: Any, state: dict) -> None:
    profile = state["profile"]
    version = state.get("profile_version", 1)
    st.header("Your experience and practice priorities")
    st.caption("Actual exposure describes what you have done. Practice depth describes what you want to practise. "
               "You can request advanced practice with tutorial-only experience.")
    with st.expander("Where the proposed skills came from"):
        for skill in profile["skills"]:
            _heading(skill["name"])
            _write("Job requirement: " + (skill.get("requirement") or "Not recorded"))
            _write("Resume evidence: " + (skill.get("resume_evidence") or "No explicit evidence found"))
    with st.form(f"profile_{version}"):
        role = st.text_input("Role", value=profile["role"])
        seniority = st.text_input("Seniority / level", value=profile.get("seniority", ""))
        skills = _skill_editor(profile["skills"], key=f"profile_skills_{version}")
        selected = _selection(profile["skills"], state.get("selected_topic_ids", []), key=f"profile_topics_{version}")
        minutes = st.number_input("Study time available (minutes)", min_value=10, max_value=600,
                                  value=state.get("preparation_minutes", 60), step=10)
        interview_stage = st.text_input("Interview stage", value=state.get("interview_stage", "Not specified"))
        confirmed = st.checkbox("These edits reflect my experience and requested practice depth.")
        saved = st.form_submit_button("Save and clarify my experience", type="primary")
    if saved:
        if not confirmed or not selected or any(not skill["name"] for skill in skills):
            st.warning("Confirm your edits, give every skill a name, and select one to three priorities.")
            return
        updated = {**profile, "role": role.strip(), "seniority": seniority.strip(), "skills": skills}
        _submit(engine, state, {"action": "save_profile", "profile": updated, "selected_topic_ids": selected,
                               "preparation_minutes": int(minutes), "interview_stage": interview_stage.strip()},
                "Saving your choices and preparing a clarification…")


def _clarification(engine: Any, state: dict) -> None:
    st.header("A little more about your experience")
    st.caption("Answer in your own words. You can skip a question or finish clarification at any time.")
    names = {skill["id"]: skill["name"] for skill in state.get("profile", {}).get("skills", [])}
    for record in state.get("answers", []):
        with st.chat_message("assistant"):
            _write(record["probe"]["question"])
        with st.chat_message("user"):
            _write("Skipped — experience remains uncertain." if record.get("skipped") else record.get("answer"))
    probe = state.get("pending_probe") or (state.get("pending") or {}).get("probe")
    if not probe:
        st.info("Your saved clarification is ready to continue.")
        if st.button("Continue clarification", type="primary"):
            if _engine_step(engine, "Preparing the next step…", lambda: engine.advance(state["run_id"])) is not None:
                st.rerun()
        return
    _caption(f"Clarification {state.get('probe_count', len(state.get('answers', [])) + 1)} of up to 5 · "
             f"{names.get(probe.get('skill_id'), 'Selected topic')}")
    with st.chat_message("assistant"):
        _write(probe["question"])
        _caption(probe.get("selection_reason", ""))
    answer = st.chat_input("Your experience or example", max_chars=6000, key=f"answer_{state.get('probe_count', 0)}")
    if answer:
        _submit(engine, state, {"action": "answer", "answer": answer}, "Saving your answer and preparing the next step…")
    left, right = st.columns(2)
    if left.button("Skip this question", width="stretch"):
        _submit(engine, state, {"action": "skip"}, "Keeping the uncertainty and continuing…")
    if right.button("Finish clarification", width="stretch"):
        _submit(engine, state, {"action": "finish"}, "Preparing your brief for review…")


def _brief_review(engine: Any, state: dict, readiness: list[str]) -> None:
    brief = state["brief"]
    version = brief["version"]
    st.header("Review your preparation brief")
    st.caption(f"Brief version {version}. Research starts only after you approve this saved version.")
    updated = deepcopy(brief)
    updated["role"] = st.text_input("Role", value=brief["role"])
    updated["seniority"] = st.text_input("Seniority / level", value=brief.get("seniority", ""))
    updated["summary"] = st.text_area("Experience summary and corrections", value=brief.get("summary", ""), height=260)
    updated["skills"] = _skill_editor(brief["skills"], key=f"brief_skills_{version}")
    updated["selected_topic_ids"] = _selection(brief["skills"], brief["selected_topic_ids"], key=f"brief_topics_{version}")
    updated["preparation_minutes"] = int(st.number_input("Study time available (minutes)", min_value=10, max_value=600,
                                                          value=brief["preparation_minutes"], step=10))
    updated["interview_stage"] = st.text_input("Interview stage", value=brief.get("interview_stage", "Not specified"))
    unknowns = st.text_area("What we still don't know — one item per line", value="\n".join(brief.get("uncertainties", [])))
    updated["uncertainties"] = [line.strip() for line in unknowns.splitlines() if line.strip()]
    updated["deferred_skills"] = [skill["name"] for skill in updated["skills"] if skill["id"] not in updated["selected_topic_ids"]]
    accepted = st.checkbox(f"I approve brief version {version} and its selected research topics.")
    left, right = st.columns(2)
    corrected = left.button("Save corrections", width="stretch")
    approved = right.button("Approve brief and research", type="primary", width="stretch",
                            disabled=bool(readiness) or not accepted)
    if corrected or approved:
        if not updated["selected_topic_ids"] or any(not skill["name"] for skill in updated["skills"]):
            st.warning("Name each skill and select one to three research priorities.")
            return
        if updated != brief:
            st.session_state["brief_notice"] = "Your corrections were saved in a new version. Review it and approve when ready."
            _submit(engine, state, {"action": "revise_brief", "brief": updated}, "Saving a new brief version…")
        elif corrected:
            st.info("There are no changed corrections to save.")
        elif not accepted:
            st.warning(f"Confirm that you approve brief version {version} before starting research.")
        else:
            _submit(engine, state, {"action": "approve_brief", "version": version},
                    "Researching your selected topics and preparing the pack. Saved work will be retained if a limit is reached…")


def _citations(identities: list[str], evidence: dict) -> None:
    for identity in dict.fromkeys(identities):
        source = evidence.get(identity)
        if not source or source.get("status") != "accepted":
            _caption(f"{identity}: supporting source unavailable")
            continue
        url = safe_source_url(source.get("url"))
        if url:
            st.link_button(escape_markdown(f"[{identity}] {source.get('title') or 'Supporting source'}"), url)
        else:
            _caption(f"{identity}: a safe source link is unavailable")


def _report(engine: Any, state: dict) -> None:
    report = state["report"]
    brief = state.get("brief", {})
    evidence = {source["id"]: source for source in state.get("evidence", [])}
    names = {skill["id"]: skill["name"] for skill in brief.get("skills", [])}
    st.header("Your preparation pack")
    status = report["status"]
    questions = report.get("questions", [])
    st.caption(f"{status.capitalize()} · Report version {report['version']} · {len(questions)} of 6 practice questions")
    if status == "complete":
        st.success("All six practice questions are available for review.")
    elif status == "partial":
        st.warning("This pack is partial. Review the available work and the gaps below.")
    else:
        st.error("A supported practice pack could not be completed. Saved evidence and gaps are retained below.")
    _heading(report["title"], 2)
    _write(report.get("summary"))
    with st.expander("Your approved experience and preparation brief"):
        _write_saved_summary(brief.get("summary"))
        for skill in brief.get("skills", []):
            if skill["id"] in brief.get("selected_topic_ids", []):
                _heading(skill["name"], 4)
                _write(f"Self-reported exposure: {EXPOSURES.get(skill.get('exposure'), 'Not sure')} · "
                       f"Requested practice depth: {DEPTHS.get(skill.get('desired_depth'), 'Not sure')}")
                _write(skill.get("notes"))
        if brief.get("deferred_skills"):
            _write("Deferred this run: " + ", ".join(brief["deferred_skills"]))
        for uncertainty in brief.get("uncertainties", []):
            _write("Still unclear: " + uncertainty)
    for topic_id in brief.get("selected_topic_ids", []):
        _heading(names.get(topic_id, topic_id), 2)
        topic_questions = [(number, question) for number, question in enumerate(questions, 1) if question["topic_id"] == topic_id]
        if not topic_questions:
            st.info("No supported questions were available for this topic.")
        for number, question in topic_questions:
            with st.container(border=True):
                _caption(f"Question {number} · {DEPTHS.get(question.get('depth'), 'Not sure')}")
                _heading(question["question"])
                _write(question.get("explanation"))
                _citations(question.get("source_ids", []), evidence)
                with st.expander("Key points your answer should cover"):
                    for point in question.get("key_points", []):
                        st.markdown("- " + escape_markdown(point["text"]))
                        _citations(point.get("source_ids", []), evidence)
                if question.get("candidate_example_prompt"):
                    _write(question["candidate_example_prompt"])
    st.subheader("Your study plan")
    plan = report.get("study_plan", [])
    if plan:
        st.caption(f"{sum(activity['minutes'] for activity in plan)} minutes planned · {brief.get('preparation_minutes', 0)} minutes available")
        show_topic = len({activity["topic_id"] for activity in plan}) > 1
        for step, activity in enumerate(plan, 1):
            _heading(f"Step {step} · {activity['minutes']} min", 4)
            if show_topic:
                _caption(names.get(activity["topic_id"], activity["topic_id"]))
            _write(activity["activity"])
            _citations(activity.get("source_ids", []), evidence)
    else:
        st.info("No study activities were available when this report was saved.")
    st.subheader("Gaps and limitations")
    for gap in report.get("gaps", []):
        st.markdown("- " + escape_markdown(gap))
    if not report.get("gaps"):
        st.caption("No additional gaps were recorded for this pack.")
    st.caption("Each source's extraction and assessment outcome is shown below. "
               "This pack has not had an independent evidence review.")
    with st.expander("Sources and retained evidence"):
        if not evidence:
            st.write("No source evidence was saved.")
        for identity, source in evidence.items():
            _heading(f"[{identity}] {source.get('title') or 'Source'}", 4)
            extracted_text = source.get("text") or ""
            has_content = bool(extracted_text.strip() or (source.get("excerpt") or "").strip())
            assessment_status = source.get("status", "uncertain")
            if not has_content:
                _caption("Extraction unsuccessful · No text available for assessment")
            elif assessment_status == "accepted":
                _caption("Extraction succeeded · Accepted as supporting evidence")
            elif assessment_status == "rejected":
                _caption("Extraction succeeded · Assessed as unsuitable for support")
            elif assessment_status == "pending":
                _caption("Extraction succeeded · Assessment pending")
            else:
                _caption("Extraction succeeded · Assessment incomplete or uncertain")
            _caption(f"Attempt recorded: {source.get('retrieved_at', 'not recorded')}")
            _write(source.get("reason"))
            _write(source.get("excerpt"))
            if extracted_text.strip() and assessment_status not in {"accepted", "rejected"}:
                preview = extracted_text[:500]
                st.caption(f"Unassessed extracted text — showing {len(preview):,} of "
                           f"{len(extracted_text):,} saved characters. This preview is not accepted supporting evidence.")
                st.text(preview)
            url = safe_source_url(source.get("url"))
            if url:
                st.link_button("Open source", url)
    reviewed = state.get("report_reviewed_version") == report["version"]
    if not reviewed:
        with st.expander("Continue research from this approved brief"):
            st.caption("Uses the saved brief and evidence, without repeating your earlier answers. "
                       "Any further provider calls count against this run's remaining limits.")
            feedback = st.text_area("What needs better source support?", value=
                                    "Check each cited answer point against its saved excerpt. Find primary evidence "
                                    "for concepts that the current sources do not support.", max_chars=1200,
                                    key=f"research_feedback_{report['version']}")
            if st.button("Continue research using remaining allowance", key=f"retry_research_{report['version']}"):
                _submit(engine, state, {"action": "retry_research", "feedback": feedback},
                        "Continuing research from saved evidence…")
        accepted = st.checkbox(f"I reviewed report version {report['version']}, including its sources and gaps.")
        if st.button("Mark reviewed and prepare download", type="primary", disabled=not accepted,
                     key=f"review_report_{report['version']}"):
            _submit(engine, state, {"action": "review_report", "version": report["version"]},
                    "Saving your report review…")
    else:
        export_fingerprint = sha256(json.dumps({key: state.get(key) for key in (
            "run_id", "report", "brief", "evidence", "approved_brief_version", "report_reviewed_version"
        )}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        cached = st.session_state.get("report_download")
        if not cached or cached["fingerprint"] != export_fingerprint:
            content = _attempt("Preparing your reviewed download…", lambda: engine.export_report(state["run_id"]))
            if content is None:
                return
            cached = {"fingerprint": export_fingerprint, "data": content}
            st.session_state["report_download"] = cached
        st.download_button("Download reviewed preparation pack", data=cached["data"],
                           file_name=f"prepcop-v{report['version']}.html", mime="text/html",
                           type="primary", on_click="ignore")
        st.caption(f"Reviewed version {report['version']} · Opens in a browser and can be printed. "
                   "Downloading reuses the saved report.")


def _progress(state: dict) -> None:
    counters = state.get("counters") or {}
    columns = st.columns(3)
    columns[0].metric("Clarifications generated", f"{state.get('probe_count', 0)} / 5")
    columns[1].metric("Search attempts", f"{counters.get('search_attempts', 0)} / 4")
    columns[2].metric("Model attempts", f"{counters.get('model_attempts', 0)} / 32")
    elapsed = counters.get("research_elapsed_seconds")
    if elapsed is not None:
        st.caption(f"Research and report machine time recorded: {float(elapsed):.0f} seconds. "
                   "Limits are stopping policies, not completion estimates.")
    with st.expander("Run details and recorded activity"):
        names = {skill["id"]: skill["name"] for skill in state.get("profile", {}).get("skills", [])}
        for topic_id, count in counters.get("extract_attempts", {}).items():
            _write(f"{names.get(topic_id, topic_id)}: {count} of 5 extraction attempts; "
                   f"{counters.get('readable_pages', {}).get(topic_id, 0)} of 4 pages confirmed readable by the Coach.")
        if counters.get("extract_attempts"):
            st.caption("Extraction can succeed before readability is confirmed. "
                       "A count of zero means no page has yet been confirmed readable.")
        st.caption(f"Shared transient retries: {counters.get('transient_retries', 0)} / 1 · "
                   f"Shared output repairs: {counters.get('repairs', 0)} / 1")
        for event in state.get("events", []):
            _caption(f"{event.get('at', '')} · {str(event.get('kind', 'Saved activity')).replace('_', ' ')}")
            # Show factual, bounded fields only. Provider payloads, prompts and
            # arbitrary error bodies never belong in this candidate-facing log.
            details = event.get("details") or {}
            allowed = {key: details[key] for key in ("message", "topic_id", "operation_id", "status", "attempt", "version") if key in details}
            for key, value in allowed.items():
                _caption(f"{key.replace('_', ' ')}: {value}")


def _unknown_operation(engine: Any, state: dict) -> None:
    st.warning("An earlier external request has no saved result. It may have completed before the interruption. "
               "Retrying can repeat paid work.")
    accepted = st.checkbox("I understand that retrying this request may repeat a charge.")
    left, right = st.columns(2)
    if left.button("Retry interrupted request", disabled=not accepted):
        _submit(engine, state, {"action": "retry_unknown"}, "Retrying the interrupted request…")
    if right.button("Stop and keep saved work"):
        _submit(engine, state, {"action": "stop"}, "Keeping the available work…")


def main() -> None:
    st.set_page_config(page_title="PrepCop", page_icon="📖", layout="wide", initial_sidebar_state="expanded")
    # Imports are delayed so render helpers remain testable independently of
    # environment setup and provider packages.
    from .config import Settings
    from .engine import Engine

    st.title("PrepCop")
    st.caption("Interrogate. Educate. Elevate.")
    st.caption("Your experience. Focused practice. Sources you can review.")
    try:
        readiness = Settings.from_env().readiness()
    except ValueError as exc:
        st.error(escape_markdown(str(exc)))
        return
    if "engine" not in st.session_state:
        try:
            st.session_state["engine"] = Engine()
        except Exception as exc:
            st.error("The local progress store could not be opened. Check that the app can write to its data directory.")
            with st.expander("Troubleshooting details"):
                st.text(f"Error category: {type(exc).__name__}")
            return
    engine = st.session_state["engine"]
    _sidebar(engine, readiness)
    run_id = st.session_state.get("run_id")
    state = _attempt("Loading saved progress…", lambda: engine.get_run(run_id)) if run_id else None
    _stage_strip(state)
    notice = st.session_state.pop("brief_notice", None)
    if notice:
        st.info(notice)
    if not state:
        _inputs(engine, readiness)
        return
    with st.expander("Your confirmed resume and job description"):
        st.caption("These are the reviewed inputs saved for this run.")
        _heading("Resume", 4)
        _write(state.get("resume_text"))
        _heading("Job description", 4)
        _write(state.get("jd_text"))
    _progress(state)
    pending = state.get("pending") or {}
    kind = pending.get("kind")
    if state.get("error"):
        st.warning(escape_markdown(state["error"]))
    if kind == "operation_unknown":
        _unknown_operation(engine, state)
    elif state.get("stage") == "stopped" and not state.get("report"):
        st.info("This run was stopped. Its saved inputs and progress are retained.")
        st.write("Use Start a new run in the sidebar when you want to begin again.")
    elif state.get("report"):
        _report(engine, state)
    elif kind == "brief_review":
        _brief_review(engine, state, readiness)
    elif kind == "clarification":
        _clarification(engine, state)
    elif kind == "profile_review":
        _profile_review(engine, state)
    else:
        st.info("Saved progress is ready to continue.")
        if st.button("Continue saved run", type="primary", disabled=bool(readiness)):
            if _engine_step(engine, "Continuing from saved progress…", lambda: engine.advance(run_id)) is not None:
                st.rerun()
