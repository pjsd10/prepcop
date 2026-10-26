"""Persisted LangGraph workflow with explicit human approval boundaries."""
from contextlib import contextmanager
from pathlib import Path
import json
import os
import re
import sqlite3
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.sqlite import SqliteSaver

from .config import Settings
from .context import RunContext
from .errors import BudgetExceeded, ProviderFailure, UnknownOperation
from .html_input import validate_input_text, MAX_JD_CHARS
from .models import Brief, Evidence, Profile, ProbeAnswer, PrepReport, RunState
from .profile import map_profile, next_probe, make_brief, validate_selected_topics
from .research import DraftResult, _assess_draft, research_decide, research_execute, draft_pack
from .report import build_reviewed_export
from .store import Store


@contextmanager
def run_lock(folder, run_id):
    """One writer per run, including separate Streamlit sessions/processes."""
    path = Path(folder) / (run_id + ".lock")
    with path.open("a+b") as handle:
        try:
            handle.seek(0, 2)
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise ValueError("This run is already working in another session. Resume after that request finishes.") from None
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _human_update(state, event):
    action = event.get("type")
    if action == "save_profile":
        profile = Profile.model_validate(event["profile"])
        topics = validate_selected_topics(profile, event.get("selected_topic_ids", []))
        for skill in profile.skills:
            skill.confirmed = True
        settings = Brief(version=state["profile_version"], role=profile.role, skills=profile.skills,
                         selected_topic_ids=topics, preparation_minutes=event.get("preparation_minutes", 60),
                         interview_stage=event.get("interview_stage", "Not specified"))
        return dict(profile=profile.model_dump(), selected_topic_ids=topics,
                    preparation_minutes=settings.preparation_minutes, interview_stage=settings.interview_stage,
                    stage="clarification", approved_brief_version=None, error=None)
    if action in ("answer", "skip", "finish"):
        answers = list(state.get("answers", []))
        if state.get("pending_probe"):
            answer = event.get("answer", "") if action == "answer" else ""
            if action == "answer":
                if not isinstance(answer, str) or not answer.strip() or len(answer) > 6000:
                    raise ValueError("Enter an answer of 1–6,000 characters, or choose Skip.")
                answer = answer.strip()
            answers.append(ProbeAnswer(probe=state["pending_probe"], answer=answer,
                                       skipped=action != "answer").model_dump())
        return dict(answers=answers, pending_probe=None, finish_probing=action == "finish", error=None)
    if action == "revise_brief":
        brief = Brief.model_validate(event["brief"])
        old = Brief.model_validate(state["brief"])
        if brief.version != old.version:
            raise ValueError("This brief changed. Reload the saved run before editing it.")
        profile = Profile(role=brief.role, seniority=brief.seniority, skills=brief.skills,
                          uncertainties=brief.uncertainties)
        validate_selected_topics(profile, brief.selected_topic_ids)
        brief.version = old.version + 1
        brief.deferred_skills = [s.name for s in brief.skills if s.id not in brief.selected_topic_ids]
        for skill in brief.skills:
            skill.confirmed = True
        return dict(brief=brief.model_dump(), profile=profile.model_dump(), profile_version=brief.version,
                    selected_topic_ids=brief.selected_topic_ids, preparation_minutes=brief.preparation_minutes,
                    interview_stage=brief.interview_stage, approved_brief_version=None,
                    research_messages=[], pending_tool=None, research_done=False, research_step=0,
                    evidence=[], candidates=[], report=None, report_reviewed_version=None, error=None)
    if action == "approve_brief":
        brief = Brief.model_validate(state["brief"])
        if event.get("version") != brief.version:
            raise ValueError("Approve the current brief version; this approval is stale.")
        validate_selected_topics(Profile(role=brief.role, skills=brief.skills), brief.selected_topic_ids)
        for skill in brief.skills:
            skill.confirmed = True
        return dict(brief=brief.model_dump(), approved_brief_version=brief.version, stage="researching", error=None)
    if action == "review_report":
        report = PrepReport.model_validate(state["report"])
        if (event.get("version") != report.version or report.brief_version != state["brief"]["version"]
                or state.get("approved_brief_version") != report.brief_version):
            raise ValueError("Review the current report and approved brief before exporting.")
        return dict(report_reviewed_version=report.version, stage="complete", error=None)
    if action == "retry_research":
        brief = Brief.model_validate(state["brief"])
        if state.get("approved_brief_version") != brief.version:
            raise ValueError("Research retry requires the current approved brief.")
        feedback = event.get("feedback", "")
        if not isinstance(feedback, str) or len(feedback) > 1200:
            raise ValueError("Research feedback must be at most 1,200 characters.")
        return {}
    raise ValueError("This action is not available at the current step.")


class Engine:
    def __init__(self, settings=None, *, model=None, retrieval=None, on_event=None):
        self._fixed_settings = settings
        self.settings = settings or Settings.from_env()
        self.model, self.retrieval = model, retrieval
        self.on_event = on_event
        self.folder = Path(self.settings.runtime_dir)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.store = Store(self.folder / "operations.sqlite")

    @staticmethod
    def _validate_id(run_id):
        if not isinstance(run_id, str) or not re.fullmatch(r"[0-9a-f]{32}", run_id):
            raise ValueError("Use the complete private resume code for this run.")

    @contextmanager
    def _graph(self, run_id):
        settings = self._fixed_settings or Settings.from_env()
        ctx = RunContext(self.store, run_id, settings, model=self.model, retrieval=self.retrieval, on_event=self.on_event)
        con = sqlite3.connect(self.folder / "checkpoints.sqlite", check_same_thread=False, timeout=10)
        graph = StateGraph(RunState)
        def node(fn, research=False):
            def call(state):
                ctx.bind_state(state)
                if research:
                    ctx.begin_stage()
                return fn(state, ctx)
            return call
        def human(kind, permitted):
            def call(state):
                ctx.end_stage()
                event = interrupt(dict(kind=kind, probe=state.get("pending_probe")))
                if event.get("type") not in permitted:
                    raise ValueError("That action does not belong to this review step.")
                return {**_human_update(state, event), "last_event_id": event.get("_event_id")}
            return call
        graph.add_node("map_profile", node(map_profile))
        graph.add_node("await_profile", human("profile_review", {"save_profile"}))
        graph.add_node("next_probe", node(next_probe))
        graph.add_node("await_answer", human("clarification", {"answer", "skip", "finish"}))
        graph.add_node("make_brief", node(make_brief))
        graph.add_node("await_brief", human("brief_review", {"approve_brief", "revise_brief"}))
        graph.add_node("research_decide", node(research_decide, True))
        graph.add_node("research_execute", node(research_execute, True))
        graph.add_node("draft_pack", node(draft_pack, True))
        graph.add_node("await_report", human("report_review", {"review_report"}))
        graph.add_edge(START, "map_profile")
        graph.add_edge("map_profile", "await_profile")
        graph.add_edge("await_profile", "next_probe")
        graph.add_conditional_edges("next_probe", lambda s: "make_brief" if s.get("finish_probing") else "await_answer")
        graph.add_conditional_edges("await_answer", lambda s: "make_brief" if s.get("finish_probing") else "next_probe")
        graph.add_edge("make_brief", "await_brief")
        graph.add_conditional_edges("await_brief", lambda s: "research_decide" if s.get("approved_brief_version") == s["brief"]["version"] else "await_brief")
        graph.add_conditional_edges("research_decide", lambda s: "research_execute" if s.get("pending_tool") else "draft_pack")
        graph.add_edge("research_execute", "research_decide")
        graph.add_edge("draft_pack", "await_report")
        graph.add_edge("await_report", END)
        try:
            yield graph.compile(checkpointer=SqliteSaver(con)), ctx
        finally:
            ctx.end_stage()
            con.close()

    @staticmethod
    def _config(run_id):
        return {"configurable": {"thread_id": run_id}, "recursion_limit": 100}

    def create_run(self, resume_text, jd_text, preparation_minutes=60, interview_stage="Not specified"):
        resume = validate_input_text(resume_text, "Resume", max_chars=self.settings.input_max_chars)
        jd = validate_input_text(jd_text, "Job description", max_chars=MAX_JD_CHARS)
        if not isinstance(preparation_minutes, int) or isinstance(preparation_minutes, bool) or not 10 <= preparation_minutes <= 600:
            raise ValueError("Preparation time must be 10–600 whole minutes.")
        if not isinstance(interview_stage, str) or not interview_stage.strip() or len(interview_stage) > 500:
            raise ValueError("Interview stage must contain 1–500 characters.")
        stage = interview_stage.strip()
        run_id = uuid4().hex
        self.store.create(run_id)
        initial = dict(run_id=run_id, stage="profile_mapping", resume_text=resume, jd_text=jd,
                       preparation_minutes=preparation_minutes, interview_stage=stage,
                       input_version=1, profile_version=1, selected_topic_ids=[], answers=[], probe_count=0,
                       finish_probing=False, research_messages=[], evidence=[], candidates=[], research_step=0,
                       research_done=False, approved_brief_version=None, report_reviewed_version=None,
                       last_event_id=None, error=None)
        with self._graph(run_id) as (graph, _):
            graph.update_state(self._config(run_id), initial, as_node=START)
        self.store.event(run_id, "run_created", {"provider": self.settings.provider, "model": self.settings.model})
        return run_id

    def get_run(self, run_id):
        self._validate_id(run_id)
        counters = self.store.counters(run_id)
        with self._graph(run_id) as (graph, _):
            snapshot = graph.get_state(self._config(run_id))
        state = dict(snapshot.values)
        if not state:
            raise ValueError("Saved run has no usable checkpoint.")
        pauses = [i for task in snapshot.tasks for i in task.interrupts]
        state["pending"] = pauses[0].value if pauses else None
        unknown = self.store.unresolved(run_id)
        if unknown:
            state["pending"] = dict(kind="operation_unknown", operation_id=unknown[0]["id"],
                                    message="A request outcome is unknown. Retry deliberately or stop with saved progress; another request may incur another charge.")
        state["counters"], state["events"] = counters, self.store.events(run_id)
        if not state.get("pending") and state["events"] and state["events"][-1]["kind"] == "run_paused":
            state["error"] = state["events"][-1]["details"]["reason"]
        return state

    def _invoke(self, graph, run_id, value):
        try:
            graph.invoke(value, self._config(run_id))
        except (UnknownOperation, ProviderFailure, BudgetExceeded) as exc:
            self.store.event(run_id, "run_paused", {"reason": str(exc), "category": type(exc).__name__})
        return self.get_run(run_id)

    def advance(self, run_id):
        self._validate_id(run_id)
        with run_lock(self.folder, run_id):
            state = self.get_run(run_id)
            if state.get("pending"):
                return state
            # Repeating Continue after a known failed request is a deliberate action.
            for op in self.store.failed(run_id):
                self.store.allow_retry(run_id, op["id"])
            with self._graph(run_id) as (graph, _):
                return self._invoke(graph, run_id, None)

    def _retry_research_update(self, run_id, state, feedback):
        """Reuse a validated saved assessment within the original run limits."""
        brief = Brief.model_validate(state["brief"])
        report = PrepReport.model_validate(state["report"])
        key = f"research:{brief.version}:draft:{report.version}"
        operation = self.store.operation(run_id, key)
        if not operation or operation["status"] != "completed" or not operation["result"]:
            raise ValueError("No completed saved draft is available to continue from.")
        saved = json.loads(operation["result"])
        content = saved.get("data", {}).get("content")
        if not isinstance(content, str):
            raise ValueError("The saved draft has no structured content to recheck.")
        draft = DraftResult.model_validate_json(content)
        original = [Evidence.model_validate(item) for item in state.get("evidence", [])]
        for item in original:
            if item.status == "uncertain":
                item.status = "pending"
        assessed = _assess_draft(draft, original)
        guidance = ("Continue the same approved brief. Check whether each answer point is directly "
                    "supported by the retained source excerpts; seek authoritative public evidence "
                    "for missing concepts before drafting. " + feedback.strip())
        return assessed, dict(stage="researching", evidence=[item.model_dump() for item in assessed],
                              research_messages=[HumanMessage(content=guidance)], candidates=[],
                              pending_tool=None, research_done=False, report_reviewed_version=None,
                              source_expansion_requested=True, error=None)

    def submit(self, run_id, event, event_id):
        self._validate_id(run_id)
        event = {**event, "type": event.get("type", event.get("action"))}
        event.pop("action", None)
        if not isinstance(event_id, str) or not 1 <= len(event_id) <= 100:
            raise ValueError("A stable submission identifier is required.")
        with run_lock(self.folder, run_id):
            state = self.get_run(run_id)
            if self.store.has_submission(run_id, event_id) and state.get("last_event_id") == event_id:
                return state
            kind = (state.get("pending") or {}).get("kind")
            permitted = {"profile_review": {"save_profile"}, "clarification": {"answer", "skip", "finish"},
                         "brief_review": {"approve_brief", "revise_brief"}, "report_review": {"review_report", "retry_research"},
                         "operation_unknown": {"retry_unknown", "stop"}}
            if event.get("type") not in permitted.get(kind, set()):
                raise ValueError("This action is stale or unavailable. Reload the current saved step.")
            if kind != "operation_unknown":
                _human_update(state, event)  # Validate before consuming a submission or leaving the pause.
            recovery = self._retry_research_update(run_id, state, event.get("feedback", "")) if event["type"] == "retry_research" else None
            self.store.claim_submission(run_id, event_id)
            self.store.event(run_id, "human_action", {"event_id": event_id, "action": event["type"], "version": event.get("version")})
            with self._graph(run_id) as (graph, _):
                if recovery is not None:
                    assessed, update = recovery
                    for before, after in zip(state.get("evidence", []), assessed):
                        if before.get("status") == "uncertain" and after.readable is True:
                            self.store.readable(run_id, after.id, after.topic_id)
                    graph.update_state(self._config(run_id), {**update, "last_event_id": event_id}, as_node="await_brief")
                    return self._invoke(graph, run_id, None)
                if kind == "operation_unknown":
                    for op in self.store.unresolved(run_id):
                        self.store.resolve_unknown_time(run_id, op["id"])
                        if event["type"] == "retry_unknown":
                            self.store.allow_retry(run_id, op["id"])
                        else:
                            self.store.finish_operation(run_id, op["id"], {"error": "user_stopped"}, "stopped")
                    if event["type"] == "stop":
                        if state.get("brief") and state.get("approved_brief_version") == state["brief"]["version"]:
                            useful = any(item.get("status") in ("accepted", "pending", "uncertain")
                                         and item.get("text", "").strip() for item in state.get("evidence", []))
                            report = PrepReport(version=1, brief_version=state["brief"]["version"],
                                                status="partial" if useful else "failed",
                                                title="Saved research — stopped", summary="You stopped after an uncertain request outcome.",
                                                gaps=["The preparation pack was not completed. Saved evidence remains available."])
                            graph.update_state(self._config(run_id), dict(report=report.model_dump(), stage="report_review", last_event_id=event_id), as_node="draft_pack")
                            return self._invoke(graph, run_id, None)
                        graph.update_state(self._config(run_id), dict(stage="stopped", last_event_id=event_id), as_node="await_report")
                        return self.get_run(run_id)
                    return self._invoke(graph, run_id, None)
                return self._invoke(graph, run_id, Command(resume={**event, "_event_id": event_id}))

    def export_report(self, run_id):
        return build_reviewed_export(self.get_run(run_id))
