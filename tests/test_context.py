"""Budget, timeout, completed-result replay and unknown-outcome boundaries."""
import json
import time
from uuid import uuid4
import pytest
from pydantic import BaseModel
from langchain_core.messages import AIMessage, message_to_dict
from interview_prep.config import Settings
from interview_prep.context import RunContext
from interview_prep.errors import BudgetExceeded, ProviderFailure, UnknownOperation
from interview_prep.store import Store

@pytest.fixture
def context(tmp_path):
    store=Store(tmp_path/'ledger.sqlite')
    run=uuid4().hex; store.create(run)
    ctx=RunContext(store,run,Settings(runtime_dir=tmp_path,model_timeout=.05),
                   model=lambda *a,**k: AIMessage(content='{"value":1}'))
    ctx.bind_state(dict(brief=dict(version=1,selected_topic_ids=["python"]),approved_brief_version=1))
    return ctx

def test_completed_model_replay_bypasses_exhausted_budget(context):
    ctx=context
    result=ctx.model_call("map",[])
    for _ in range(31): ctx.store.charge(ctx.run_id,"model")
    assert ctx.model_call("map",[]).content==result.content
    assert ctx.store.counters(ctx.run_id)["model_attempts"]==32
    with pytest.raises(BudgetExceeded): ctx.model_call("another",[])

def test_timeout_is_unknown_and_never_automatically_repeated(context):
    ctx=context; calls=[]
    def slow(*a,**k): calls.append(1); time.sleep(.2); return AIMessage(content="late")
    ctx.model=slow
    with pytest.raises(UnknownOperation): ctx.model_call("slow",[])
    with pytest.raises(UnknownOperation): ctx.model_call("slow",[])
    assert len(calls)==1
    assert ctx.store.counters(ctx.run_id)["model_attempts"]==1
    assert ctx.store.counters(ctx.run_id)["transient_retries"]==0
    assert ctx.store.operation(ctx.run_id,"slow")["status"]=="unknown"

def test_provider_timeout_exception_is_also_unknown(context):
    def timeout(*a,**k): raise TimeoutError("private request details")
    context.model=timeout
    with pytest.raises(UnknownOperation): context.model_call("socket",[])
    assert context.store.operation(context.run_id,"socket")["status"]=="unknown"

def test_tool_unknown_flag_never_uses_transient_retry(context):
    ctx=context; ctx.begin_stage()
    calls=[]
    def timeout(wait): calls.append(wait); return dict(ok=False,unknown_outcome=True,retryable=False)
    with pytest.raises(UnknownOperation): ctx.execute_tool("research:tool","search","python",{},timeout)
    assert len(calls)==1
    assert ctx.store.counters(ctx.run_id)["search_attempts"]==1
    assert ctx.store.counters(ctx.run_id)["transient_retries"]==0

def test_shared_retry_counts_attempts_and_replay_does_not_repeat(context):
    ctx=context; calls=[]
    class RateLimit(Exception): status_code=429
    def provider(*a,**k):
        calls.append(1)
        if len(calls)==1: raise RateLimit()
        return AIMessage(content="success")
    ctx.model=provider
    assert ctx.model_call("retry",[]).content=="success"
    assert ctx.model_call("retry",[]).content=="success"
    assert len(calls)==2
    c=ctx.store.counters(ctx.run_id)
    assert c["model_attempts"]==2 and c["transient_retries"]==1

def test_completed_child_reconciles_parent_after_crash(context):
    ctx=context
    ctx.store.begin_operation(ctx.run_id,"research:tool","search",{},20)
    ctx.store.finish_operation(ctx.run_id,"research:tool",dict(ok=False,retryable=True))
    ctx.store.begin_operation(ctx.run_id,"research:tool:transient-retry","search",{},20)
    ctx.store.finish_operation(ctx.run_id,"research:tool:transient-retry",dict(ok=True,results=["saved"]))
    result=ctx.execute_tool("research:tool","search","python",{},lambda t:pytest.fail("Repeated a saved result"))
    assert result==dict(ok=True,results=["saved"])

def test_pending_child_prevents_parent_failure_from_hiding_unknown(context):
    ctx=context
    ctx.store.begin_operation(ctx.run_id,"research:tool","search",{},20)
    ctx.store.finish_operation(ctx.run_id,"research:tool",dict(ok=False,retryable=True))
    ctx.store.begin_operation(ctx.run_id,"research:tool:transient-retry","search",{},20)
    with pytest.raises(UnknownOperation):
        ctx.execute_tool("research:tool","search","python",{},lambda t:pytest.fail("Repeated unknown operation"))

def test_schema_repair_is_shared_counted_and_replayable(context):
    class Result(BaseModel): value:int
    ctx=context; values=iter(["bad",'{"value":1}'])
    ctx.model=lambda *a,**k:AIMessage(content=next(values))
    assert ctx.structured_call("schema",[],Result).value==1
    assert ctx.structured_call("schema",[],Result).value==1
    c=ctx.store.counters(ctx.run_id)
    assert c["model_attempts"]==2 and c["repairs"]==1
    assert not ctx.consume_repair("different-repair")

def test_retrieval_deadline_reserves_three_minutes_for_drafting(context):
    ctx=context; ctx.begin_stage()
    ctx.store.add_elapsed(ctx.run_id,300)
    assert not ctx.can_retrieve("search","python")
    assert ctx.can_model()
    with pytest.raises(BudgetExceeded):
        ctx.execute_tool("late","search","python",{},lambda t:pytest.fail("Called after cutoff"))

def test_malformed_single_line_fence_uses_one_repair_instead_of_crashing(context):
    class Result(BaseModel): value:int
    ctx=context; values=iter(["```json incomplete", '```json{"value":2}```'])
    ctx.model=lambda *a,**k: AIMessage(content=next(values))
    assert ctx.structured_call("broken-fence",[],Result).value==2
    assert ctx.store.counters(ctx.run_id)["model_attempts"]==2
    assert ctx.store.counters(ctx.run_id)["repairs"]==1

def test_failed_attempts_and_readable_pages_have_separate_limits(context):
    ctx=context; ctx.begin_stage()
    for index in range(5):
        ctx.execute_tool(f"extract-{index}","extract","python",{},lambda t:dict(ok=False,status="failed"))
    assert not ctx.can_retrieve("extract","python")
    assert ctx.store.counters(ctx.run_id)["readable_pages"]=={}
    for index in range(4): ctx.record_readable("python",f"source-{index}")
    ctx.record_readable("python","source-0")
    with pytest.raises(BudgetExceeded):ctx.record_readable("python","source-5")
    assert ctx.store.counters(ctx.run_id)["readable_pages"]=={"python":4}

def test_stale_approval_blocks_external_tool_before_charge(context):
    ctx=context; ctx.state["approved_brief_version"]=0
    with pytest.raises(ValueError): ctx.execute_tool("unapproved","search","python",{},lambda t:pytest.fail("Unapproved research"))
    assert ctx.store.counters(ctx.run_id)["search_attempts"]==0

def test_crashed_pending_operation_reserves_time_once(context):
    ctx=context
    ctx.store.begin_operation(ctx.run_id,"research:crash","model",{},75)
    ctx.store.resolve_unknown_time(ctx.run_id,"research:crash")
    ctx.store.resolve_unknown_time(ctx.run_id,"research:crash")
    assert ctx.store.counters(ctx.run_id)["research_elapsed_seconds"]==75

def test_human_pause_does_not_consume_research_clock(context):
    ctx=context; now=[10.0]; ctx.clock=lambda:now[0]
    ctx.begin_stage(); now[0]=20; ctx.end_stage()
    now[0]=4000
    assert ctx.remaining_stage_seconds()==470

def test_observer_error_does_not_lose_durable_event(context):
    context.on_event=lambda *args: (_ for _ in ()).throw(RuntimeError("display broke"))
    context.event("test",{"status":"saved"})
    assert context.store.events(context.run_id)[-1]["kind"]=="test"
