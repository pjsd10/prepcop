"""Real SQLite/LangGraph integration with deterministic provider responses."""
import json
from dataclasses import replace
from uuid import uuid4
import pytest
from langchain_core.messages import AIMessage
from interview_prep.config import Settings
from interview_prep.engine import Engine, run_lock

RESUME = "Candidate has used Python for a personal course project and practiced writing automated unit tests."
JD = "Software engineer position requiring Python functions and automated testing of production services."

def mapping():
    return dict(role="Software engineer", seniority="Junior", skills=[dict(
        id="python", name="Python", requirement="Python functions and automated testing",
        resume_evidence="Python for a personal course project", exposure="personal_course_project",
        desired_depth="beginner", notes="", confirmed=False)], uncertainties=["Work ownership is unknown."])

def probe():
    return dict(probe=dict(skill_id="python", question="What Python work did you personally own?",
                          selection_reason="Clarify selected topic ownership.", addresses_unknown="Work ownership"))

def brief():
    return dict(summary="Candidate requests beginner Python practice based on course work.",
                clarifications=[], uncertainties=[], resolved_uncertainties=[])

def complete_draft():
    return dict(assessments=[], report=dict(version=1, brief_version=1, status="complete", title="Python practice",
        summary="Practice grounded in the retrieved documentation.", questions=[dict(
            id=f"Q{i}", topic_id="python", question=f"How would you test function case {i}?",
            depth="beginner", explanation="A function can return a computed result. Test its expected result.",
            key_points=[dict(text=t, source_ids=["S2"]) for t in
                        ["Define a function with def.", "Return a computed result.", "Test the expected result."]],
            source_ids=["S2"], candidate_example_prompt="Discuss your course example, if applicable.") for i in range(6)],
        study_plan=[dict(topic_id="python", activity="Read the accepted excerpt and practice the six questions.",
                         minutes=60, source_ids=["S2"])], gaps=[]))

class ScriptedModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
    def __call__(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if not self.responses:
            raise AssertionError("Unexpected model call")
        response = self.responses.pop(0)
        if isinstance(response, Exception): raise response
        return response if isinstance(response, AIMessage) else AIMessage(content=json.dumps(response))

class Retrieval:
    def __init__(self): self.calls=[]
    def search_sources(self, query, timeout=20):
        self.calls.append(("search",query,timeout))
        return dict(ok=True,status="success",results=[dict(url="https://docs.python.org/3/tutorial/controlflow.html",title="Python functions",content="Search lead")])
    def extract_page(self, url, timeout=20):
        self.calls.append(("extract",url,timeout))
        return dict(ok=True,status="success",content_available=True,results=[dict(url=url,title="Python functions",
              raw_content="Define a function with def. Return a computed result. Test the expected result.")])

def call(name, args, ident):
    return AIMessage(content="",tool_calls=[dict(name=name,args=args,id=ident,type="tool_call")])

def scenario():
    return [mapping(),probe(),brief(),
        call("search_sources",dict(topic_id="python",query="Python functions official documentation",reason="Find primary Python source"),"search-1"),
        call("extract_page",dict(topic_id="python",url="https://docs.python.org/3/tutorial/controlflow.html",reason="Read primary Python function details"),"extract-1"),
        dict(action="draft",reason="Enough support for this focused beginner practice",assessment=dict(source_id="S2",readable=True,accepted=True,
            reason="The excerpt explains basic function definition and return behavior.", excerpts=["Define a function with def. Return a computed result. Test the expected result."])),
        complete_draft()]

@pytest.fixture
def setup(tmp_path):
    settings=Settings(runtime_dir=tmp_path)
    model=ScriptedModel(scenario())
    retrieval=Retrieval()
    return Engine(settings,model=model,retrieval=retrieval),model,retrieval,settings

def submit(engine,run,event):
    return engine.submit(run,event,uuid4().hex)

def to_brief(engine):
    run=engine.create_run(RESUME,JD)
    state=engine.advance(run)
    assert state["pending"]["kind"]=="profile_review"
    state=submit(engine,run,dict(action="save_profile",profile=state["profile"],selected_topic_ids=["python"]))
    assert state["pending"]["kind"]=="clarification"
    state=submit(engine,run,dict(action="finish"))
    assert state["pending"]["kind"]=="brief_review"
    return run,state

def test_full_graph_approvals_restart_native_tools_and_reviewed_export(setup):
    engine,model,retrieval,settings=setup
    run,state=to_brief(engine)
    assert len(model.calls)==3 and retrieval.calls==[]
    assert state["approved_brief_version"] is None
    with pytest.raises(ValueError): engine.export_report(run)
    # New engine uses durable SQLite state, not the old Python instance.
    restarted=Engine(settings,model=model,retrieval=retrieval)
    assert restarted.get_run(run)["brief"]==state["brief"]
    state=submit(restarted,run,dict(action="approve_brief",version=1))
    assert state["pending"]["kind"]=="report_review"
    assert state["report"]["status"]=="complete"
    assert len(state["report"]["questions"])==6
    assert [r[0] for r in retrieval.calls]==["search","extract"]
    assert state["counters"]["model_attempts"]==7
    assert state["counters"]["search_attempts"]==1
    assert state["counters"]["extract_attempts"]=={"python":1}
    assert state["counters"]["readable_pages"]=={"python":1}
    assert all(not args["tools"] for _,args in model.calls[:3])
    with pytest.raises(ValueError): restarted.export_report(run)
    state=submit(restarted,run,dict(action="review_report",version=1))
    assert state["stage"]=="complete"
    export=restarted.export_report(run)
    assert b"Python practice" in export and b"docs.python.org" in export
    restarted.get_run(run); restarted.advance(run); restarted.export_report(run)
    assert len(model.calls)==7 and len(retrieval.calls)==2

def test_duplicate_submission_and_idle_reads_do_not_call_providers(setup):
    engine,model,retrieval,_=setup
    run=engine.create_run(RESUME,JD)
    engine.get_run(run); engine.get_run(run)
    assert model.calls==[]
    state=engine.advance(run)
    event=dict(action="save_profile",profile=state["profile"],selected_topic_ids=["python"])
    first=engine.submit(run,event,"stable-click")
    second=engine.submit(run,event,"stable-click")
    assert first["pending"]==second["pending"] and len(model.calls)==2


def test_research_retry_keeps_approval_and_limits_and_requires_new_report_review(setup):
    engine,model,retrieval,_=setup
    run,_=to_brief(engine)
    state=submit(engine,run,dict(action="approve_brief",version=1))
    assert state["report"]["status"]=="complete"
    retry_draft=complete_draft()
    retry_draft["report"]["version"]=2
    model.responses.extend([dict(action="draft",reason="Saved evidence is sufficient after review",assessment=None),retry_draft])
    state=submit(engine,run,dict(action="retry_research",feedback="Recheck support for each key point."))
    assert state["pending"]["kind"]=="report_review"
    assert state["approved_brief_version"]==1
    assert state["report"]["version"]==2 and state["report_reviewed_version"] is None
    assert state["counters"]["model_attempts"]==9
    assert state["counters"]["search_attempts"]==1 and len(retrieval.calls)==2

def test_brief_revision_invalidates_approval_stale_approval_is_rejected(setup):
    engine,model,retrieval,_=setup
    run,state=to_brief(engine)
    revision=state["brief"].copy(); revision["summary"]="Human correction."
    state=submit(engine,run,dict(action="revise_brief",brief=revision))
    assert state["brief"]["version"]==2 and state["approved_brief_version"] is None
    with pytest.raises(ValueError,match="stale"):
        submit(engine,run,dict(action="approve_brief",version=1))
    assert retrieval.calls==[] and len(model.calls)==3

def test_short_answer_is_accepted_and_bound_to_saved_probe(setup):
    engine,model,_,_=setup
    run=engine.create_run(RESUME,JD,interview_stage="Technical screen")
    state=engine.advance(run)
    state=submit(engine,run,dict(action="save_profile",profile=state["profile"],selected_topic_ids=["python"]))
    # Let the next adaptive decision stop after the first selected skill.
    model.responses.insert(0,dict(probe=None,stop_reason="The initial answer resolves useful ambiguity."))
    state=submit(engine,run,dict(action="answer",answer="Tutorial only"))
    assert state["answers"][0]["answer"]=="Tutorial only"
    assert state["pending"]["kind"]=="brief_review"

@pytest.mark.parametrize("topics",[[],["python","python"],["absent"]])
def test_invalid_selection_cannot_leave_human_pause(setup,topics):
    engine,model,_,_=setup
    run=engine.create_run(RESUME,JD); state=engine.advance(run)
    with pytest.raises(ValueError): submit(engine,run,dict(action="save_profile",profile=state["profile"],selected_topic_ids=topics))
    assert engine.get_run(run)["pending"]["kind"]=="profile_review" and len(model.calls)==1

def test_run_lock_blocks_concurrent_writer(setup):
    engine,_,_,_=setup
    run=engine.create_run(RESUME,JD)
    with run_lock(engine.folder,run):
        with pytest.raises(ValueError,match="another session"): engine.advance(run)

def test_stop_after_unknown_request_with_only_rejected_sources_is_failed(setup):
    engine,model,retrieval,_=setup
    run,_=to_brief(engine)
    retrieval.extract_page=lambda *a,**k:dict(ok=False,status="failed",error="Page unavailable")
    model.responses[2]=TimeoutError("Unknown model outcome")
    state=submit(engine,run,dict(action="approve_brief",version=1))
    assert state["pending"]["kind"]=="operation_unknown"
    assert len(state["evidence"])==1 and state["evidence"][0]["status"]=="rejected"
    state=submit(engine,run,dict(action="stop"))
    assert state["report"]["status"]=="failed"
    assert state["pending"]["kind"]=="report_review"
    assert len(state["evidence"])==1
