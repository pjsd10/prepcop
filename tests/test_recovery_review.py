"""Recovery acceptance checks with actual SQLite/Engine and controlled providers.

No live HTTP is used. Crash windows are represented by the exact durable ledger
records that exist immediately before an interrupted step.
"""
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, message_to_dict
from pydantic import BaseModel

from interview_prep.config import Settings
from interview_prep.context import RunContext
from interview_prep.engine import Engine
from interview_prep.errors import BudgetExceeded, ProviderFailure
from interview_prep.models import Brief, Skill
from interview_prep.research import research_decide
from interview_prep.store import Store
from test_engine import JD, RESUME, Retrieval, ScriptedModel, mapping


class RateLimit(Exception):
    status_code = 429


class Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value


def make_context(tmp_path, responses=()):
    store = Store(tmp_path / "review-ledger.sqlite")
    run = uuid4().hex
    store.create(run)
    model = ScriptedModel(responses)
    clock = Clock()
    ctx = RunContext(store, run, Settings(runtime_dir=tmp_path, model_timeout=30),
                     model=model, retrieval=Retrieval(), clock=clock)
    brief = Brief(version=1, role="Software Engineer", selected_topic_ids=["python"], preparation_minutes=60,
                  skills=[Skill(id="python", name="Python", desired_depth="beginner", exposure="personal_course_project")])
    state = {"run_id": run, "brief": brief.model_dump(), "approved_brief_version": 1,
             "selected_topic_ids": ["python"], "research_step": 0, "evidence": [], "candidates": []}
    ctx.bind_state(state)
    ctx.begin_stage()
    return ctx, state, model, clock


def record_operation(ctx, key, kind, result, status="completed"):
    ctx.store.begin_operation(ctx.run_id, key, kind, {}, 20)
    ctx.store.finish_operation(ctx.run_id, key, result, status)


def test_engine_unknown_transient_model_child_can_be_deliberately_retried_once(tmp_path):
    model = ScriptedModel([RateLimit(), TimeoutError("controlled uncertain child"), mapping()])
    settings = Settings(runtime_dir=tmp_path, model_timeout=30)
    engine = Engine(settings, model=model, retrieval=Retrieval())
    run = engine.create_run(RESUME, JD)
    paused = engine.advance(run)
    assert paused["pending"]["kind"] == "operation_unknown"
    assert paused["pending"]["operation_id"].endswith(":transient-retry")
    assert paused["counters"]["model_attempts"] == 2
    assert len(model.calls) == 2

    restarted = Engine(settings, model=model, retrieval=Retrieval())
    resumed = restarted.submit(run, {"action": "retry_unknown"}, "retry-child-once")
    assert resumed["pending"]["kind"] == "profile_review"
    assert resumed["profile"]["role"] == "Software engineer"
    assert len(model.calls) == 3
    assert resumed["counters"]["model_attempts"] == 3
    assert resumed["counters"]["transient_retries"] == 1
    assert not restarted.store.unresolved(run)
    restarted.get_run(run)
    restarted.advance(run)
    assert len(model.calls) == 3


def test_engine_unknown_schema_repair_manual_retry_keeps_original_result_and_one_allowance(tmp_path):
    model = ScriptedModel([AIMessage(content="malformed JSON"), TimeoutError("controlled uncertain repair"), mapping()])
    settings = Settings(runtime_dir=tmp_path, model_timeout=30)
    engine = Engine(settings, model=model, retrieval=Retrieval())
    run = engine.create_run(RESUME, JD)
    paused = engine.advance(run)
    assert paused["pending"]["kind"] == "operation_unknown"
    assert paused["pending"]["operation_id"].endswith(":repair")
    assert len(model.calls) == 2 and paused["counters"]["repairs"] == 1
    repair_key = paused["pending"]["operation_id"]
    original_key = repair_key.removesuffix(":repair")
    before = engine.store.operation(run, original_key)["result"]

    restarted = Engine(settings, model=model, retrieval=Retrieval())
    resumed = restarted.submit(run, {"action": "retry_unknown"}, "retry-schema-once")
    assert resumed["pending"]["kind"] == "profile_review"
    assert len(model.calls) == 3
    assert resumed["counters"]["model_attempts"] == 3 and resumed["counters"]["repairs"] == 1
    assert restarted.store.operation(run, original_key)["result"] == before
    assert not restarted.store.unresolved(run)


def test_completed_transient_child_passes_research_gate_at_model_and_time_limits(tmp_path):
    ctx, state, model, _ = make_context(tmp_path)
    key = "research:1:decision:0"
    saved = AIMessage(content='{"action":"draft","reason":"Saved source selection is complete.","assessment":null}')
    record_operation(ctx, key, "model", {"error": "rate_limit"}, "failed")
    record_operation(ctx, key + ":transient-retry", "model", message_to_dict(saved))
    for _ in range(32):
        ctx.store.charge(ctx.run_id, "model")
    ctx.store.add_elapsed(ctx.run_id, 480)

    assert ctx.completed_operation(key)
    update = research_decide(state, ctx)
    assert update["research_done"] and update["error"] is None
    assert update["research_messages"][0].content == saved.content
    assert not model.calls and ctx.store.counters(ctx.run_id)["model_attempts"] == 32
    assert ctx.store.operation(ctx.run_id, key)["status"] == "completed"


def test_reserved_model_retry_without_child_record_resumes_after_crash(tmp_path):
    ctx, _, model, _ = make_context(tmp_path, [AIMessage(content="Recovered reserved retry")])
    key = "research:1:decision:0"
    ctx.store.charge(ctx.run_id, "model")
    record_operation(ctx, key, "model", {"error": "provider_failure", "status_code": 429}, "failed")
    assert ctx.store.allowance(ctx.run_id, "transient_retries", key + ":transient-retry")
    assert ctx.store.operation(ctx.run_id, key + ":transient-retry") is None

    assert ctx.model_call(key, []).content == "Recovered reserved retry"
    assert ctx.model_call(key, []).content == "Recovered reserved retry"
    assert len(model.calls) == 1
    counts = ctx.store.counters(ctx.run_id)
    assert counts["model_attempts"] == 2 and counts["transient_retries"] == 1


def test_reserved_tool_retry_without_child_record_resumes_after_crash(tmp_path):
    ctx, _, _, _ = make_context(tmp_path)
    key = "research:1:tool:1:search"
    ctx.store.charge(ctx.run_id, "search")
    record_operation(ctx, key, "search", {"ok": False, "status": "failed", "retryable": True})
    assert ctx.store.allowance(ctx.run_id, "transient_retries", key + ":transient-retry")
    calls = []

    def retrieval(timeout):
        calls.append(timeout)
        return {"ok": True, "status": "success", "results": []}

    assert ctx.execute_tool(key, "search", "python", {}, retrieval)["ok"] is True
    assert ctx.execute_tool(key, "search", "python", {}, retrieval)["ok"] is True
    assert len(calls) == 1
    counts = ctx.store.counters(ctx.run_id)
    assert counts["search_attempts"] == 2 and counts["transient_retries"] == 1


def test_decision_transient_retry_and_repair_cannot_spend_the_final_two_model_attempts(tmp_path):
    ctx, state, _, _ = make_context(tmp_path, [RateLimit(), AIMessage(content="malformed decision"),
                                           AIMessage(content='{"action":"draft","reason":"No further sources are necessary."}')])
    for _ in range(29):
        ctx.store.charge(ctx.run_id, "model")
    update = research_decide(state, ctx)
    assert update["research_done"]
    assert ctx.store.counters(ctx.run_id)["model_attempts"] <= 30
    assert ctx.can_model()


def test_blocked_native_repair_does_not_consume_the_finalization_repair_allowance(tmp_path):
    ctx, state, model, _ = make_context(tmp_path, [AIMessage(content="malformed decision")])
    for _ in range(29):
        ctx.store.charge(ctx.run_id, "model")
    assert research_decide(state, ctx)["research_done"]
    counters = ctx.store.counters(ctx.run_id)
    assert len(model.calls) == 1 and counters["model_attempts"] == 30
    assert counters["repairs"] == 0


def test_final_draft_can_use_both_reserved_model_attempts_for_schema_repair(tmp_path):
    class Result(BaseModel):
        value: int

    ctx, _, model, _ = make_context(tmp_path, [AIMessage(content="malformed draft"), {"value": 1}])
    for _ in range(30):
        ctx.store.charge(ctx.run_id, "model")
    assert ctx.structured_call("research:1:draft:1", [], Result).value == 1
    counters = ctx.store.counters(ctx.run_id)
    assert len(model.calls) == 2 and counters["model_attempts"] == 32
    assert counters["repairs"] == 1


def test_thirty_second_attempt_cannot_automatically_start_a_thirty_third(tmp_path):
    ctx, _, model, _ = make_context(tmp_path, [RateLimit(), AIMessage(content="must not be called")])
    for _ in range(31):
        ctx.store.charge(ctx.run_id, "model")
    with pytest.raises(ProviderFailure):
        ctx.model_call("research:1:draft:1", [])
    assert len(model.calls) == 1
    assert ctx.store.counters(ctx.run_id)["model_attempts"] == 32
    assert ctx.store.counters(ctx.run_id)["transient_retries"] == 0
    with pytest.raises(BudgetExceeded):
        ctx.model_call("research:1:draft:2", [])


def test_elapsed_deadline_blocks_transient_retry_before_it_is_charged(tmp_path):
    ctx, _, _, clock = make_context(tmp_path)
    ctx.store.add_elapsed(ctx.run_id, 478)
    calls = []

    def fail_at_deadline(messages, **kwargs):
        calls.append(kwargs["timeout"])
        clock.value += 2
        raise RateLimit()

    ctx.model = fail_at_deadline
    with pytest.raises(ProviderFailure):
        ctx.model_call("research:1:draft:1", [])
    assert calls == [2]
    assert ctx.remaining_stage_seconds() == 0 and not ctx.can_model()
    counters = ctx.store.counters(ctx.run_id)
    assert counters["research_elapsed_seconds"] == 480
    assert counters["model_attempts"] == 1 and counters["transient_retries"] == 0


def test_retrieval_wait_is_bounded_by_five_minute_cutoff(tmp_path):
    ctx, _, _, clock = make_context(tmp_path)
    ctx.store.add_elapsed(ctx.run_id, 298.5)
    waits = []

    def retrieve(timeout):
        waits.append(timeout)
        clock.value += 1.5
        return {"ok": True, "status": "success", "results": []}

    ctx.execute_tool("research:1:tool:1:search", "search", "python", {}, retrieve)
    assert waits == [1.5]
    assert not ctx.can_retrieve("search", "python")
    assert ctx.remaining_stage_seconds() == 180 and ctx.can_model()


def test_human_wait_and_restart_do_not_reset_or_consume_saved_stage_time(tmp_path):
    ctx, state, _, clock = make_context(tmp_path)
    clock.value = 20
    ctx.end_stage()
    clock.value = 10_000  # Human waiting while no research stopwatch is running.
    restarted = RunContext(ctx.store, ctx.run_id, ctx.settings, model=ctx.model, clock=clock)
    restarted.bind_state(state)
    restarted.begin_stage()
    assert restarted.remaining_stage_seconds() == 460
    clock.value += 10
    restarted.end_stage()
    assert ctx.store.counters(ctx.run_id)["research_elapsed_seconds"] == 30
