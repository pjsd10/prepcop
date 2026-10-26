"""Real UI + SQLite/LangGraph checks with explicitly controlled providers."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from interview_prep.config import Settings
from interview_prep.engine import Engine
from test_engine import JD, RESUME, Retrieval, ScriptedModel, scenario, to_brief


APP = Path(__file__).resolve().parents[1] / "app.py"


def configured_engine(tmp_path, monkeypatch):
    settings = Settings(runtime_dir=tmp_path, api_key="controlled-test", model="controlled-test", tavily_api_key="controlled-test")
    monkeypatch.setattr(Settings, "from_env", classmethod(lambda cls: settings))
    model = ScriptedModel(scenario())
    retrieval = Retrieval()
    return Engine(settings, model=model, retrieval=retrieval), model, retrieval, settings


def button(app, label):
    return next(element for element in app.button if element.label == label)


def checkbox(app, prefix):
    return next(element for element in app.checkbox if element.label.startswith(prefix))


def assert_rendered(app):
    assert len(app.exception) == 0, [element.message for element in app.exception]


def test_profile_button_waits_for_both_confirmations(tmp_path, monkeypatch):
    engine, model, retrieval, _ = configured_engine(tmp_path, monkeypatch)
    resume_path = APP.parent / "tests" / "fixtures" / "fictional-resumes" / "02-qa-engineer" / "resume.html"

    class UploadedResume:
        name = "resume.html"

        def getvalue(self):
            return resume_path.read_bytes()

    monkeypatch.setattr("streamlit.file_uploader", lambda *args, **kwargs: UploadedResume())
    app = AppTest.from_file(str(APP), default_timeout=20)
    app.session_state["engine"] = engine
    app.run()
    assert_rendered(app)
    assert button(app, "Review my profile").disabled

    checkbox(app, "I reviewed the resume").check().run()
    assert_rendered(app)
    assert button(app, "Review my profile").disabled

    checkbox(app, "Use these confirmed inputs").check().run()
    assert_rendered(app)
    assert not button(app, "Review my profile").disabled
    assert model.calls == [] and retrieval.calls == []


def test_removing_resume_clears_stale_preview_and_input_review(tmp_path, monkeypatch):
    engine, model, retrieval, _ = configured_engine(tmp_path, monkeypatch)
    resume_path = APP.parent / "tests" / "fixtures" / "fictional-resumes" / "02-qa-engineer" / "resume.html"

    class UploadedResume:
        name = "resume.html"

        def getvalue(self):
            return resume_path.read_bytes()

    current_upload = [UploadedResume()]
    monkeypatch.setattr("streamlit.file_uploader", lambda *args, **kwargs: current_upload[0])
    app = AppTest.from_file(str(APP), default_timeout=20)
    app.session_state["engine"] = engine
    app.run()
    assert_rendered(app)
    preview = next(element for element in app.text_area if element.label == "Review and correct the extracted resume")
    assert "FICTIONAL TEST PROFILE" in preview.value
    checkbox(app, "I reviewed the resume").check().run()
    assert checkbox(app, "I reviewed the resume").value

    current_upload[0] = None
    app.run()
    assert_rendered(app)
    preview = next(element for element in app.text_area if element.label == "Review and correct the extracted resume")
    assert preview.value == ""
    assert preview.disabled
    assert not checkbox(app, "I reviewed the resume").value
    assert button(app, "Review my profile").disabled
    assert model.calls == [] and retrieval.calls == []


def test_real_ui_profile_clarification_revision_research_review_and_cached_export(tmp_path, monkeypatch):
    engine, model, retrieval, _ = configured_engine(tmp_path, monkeypatch)
    run = engine.create_run(RESUME, JD)
    engine.advance(run)
    app = AppTest.from_file(str(APP), default_timeout=20)
    app.session_state["engine"] = engine
    app.session_state["run_id"] = run
    app.run()
    assert_rendered(app)
    assert len(model.calls) == 1
    app.multiselect[0].set_value(["python"])
    checkbox(app, "These edits reflect").check()
    button(app, "Save and clarify my experience").click().run()
    assert_rendered(app)
    assert engine.get_run(run)["pending"]["kind"] == "clarification"
    assert len(model.calls) == 2
    app.run()
    assert len(model.calls) == 2
    button(app, "Finish clarification").click().run()
    assert_rendered(app)
    assert engine.get_run(run)["pending"]["kind"] == "brief_review"
    assert len(model.calls) == 3
    summary = next(element for element in app.text_area if element.label == "Experience summary and corrections")
    summary.set_value("I want beginner practice based only on my course project.")
    assert button(app, "Approve brief and research").disabled
    checkbox(app, "I approve brief version").check().run()
    assert not button(app, "Approve brief and research").disabled
    button(app, "Approve brief and research").click().run()
    assert_rendered(app)
    revised = engine.get_run(run)
    assert revised["brief"]["version"] == 2
    assert revised["approved_brief_version"] is None
    assert retrieval.calls == [] and len(model.calls) == 3
    # The controlled provider responds against the now-reviewed brief version.
    model.responses[-1]["report"]["brief_version"] = 2
    assert button(app, "Approve brief and research").disabled
    checkbox(app, "I approve brief version 2").check().run()
    assert not button(app, "Approve brief and research").disabled
    button(app, "Approve brief and research").click().run()
    assert_rendered(app)
    ready = engine.get_run(run)
    assert ready["report"]["status"] == "complete"
    assert len(ready["report"]["questions"]) == 6
    assert len(model.calls) == 7 and len(retrieval.calls) == 2
    assert ready["report_reviewed_version"] is None
    assert button(app, "Mark reviewed and prepare download").disabled
    checkbox(app, "I reviewed report version 1").check().run()
    assert not button(app, "Mark reviewed and prepare download").disabled
    button(app, "Mark reviewed and prepare download").click().run()
    assert_rendered(app)
    assert engine.get_run(run)["report_reviewed_version"] == 1
    exported = app.session_state["report_download"]["data"]
    assert exported == engine.export_report(run)
    assert b"Brief version 2" in exported and b"6 of 6 practice questions" in exported
    app.run()
    assert_rendered(app)
    assert app.session_state["report_download"]["data"] == exported
    assert len(model.calls) == 7 and len(retrieval.calls) == 2


def test_resume_sidebar_reopens_same_saved_brief_without_model_calls(tmp_path, monkeypatch):
    engine, model, retrieval, settings = configured_engine(tmp_path, monkeypatch)
    run, original = to_brief(engine)
    restarted = Engine(settings, model=model, retrieval=retrieval)
    app = AppTest.from_file(str(APP), default_timeout=20)
    app.session_state["engine"] = restarted
    app.run()
    assert_rendered(app)
    next(element for element in app.text_input if element.label == "Resume code").set_value(run)
    button(app, "Resume saved run").click().run()
    assert_rendered(app)
    assert app.session_state["run_id"] == run
    assert restarted.get_run(run)["brief"] == original["brief"]
    assert len(model.calls) == 3 and retrieval.calls == []


def test_bad_resume_code_has_friendly_error_and_does_not_reveal_saved_runs(tmp_path, monkeypatch):
    engine, model, _, _ = configured_engine(tmp_path, monkeypatch)
    app = AppTest.from_file(str(APP), default_timeout=20)
    app.session_state["engine"] = engine
    app.run()
    next(element for element in app.text_input if element.label == "Resume code").set_value("not-a-private-code")
    button(app, "Resume saved run").click().run()
    assert_rendered(app)
    assert any("complete private resume code" in element.value for element in app.error)
    assert model.calls == []


def test_stopped_run_without_report_has_no_continue_action(tmp_path, monkeypatch):
    engine, model, _, _ = configured_engine(tmp_path, monkeypatch)
    run = engine.create_run(RESUME, JD)
    with engine._graph(run) as (graph, _):
        graph.update_state(engine._config(run), {"stage": "stopped"}, as_node="await_report")
    app = AppTest.from_file(str(APP), default_timeout=20)
    app.session_state["engine"] = engine
    app.session_state["run_id"] = run
    app.run()
    assert_rendered(app)
    assert any("This run was stopped" in element.value for element in app.info)
    assert not any(element.label == "Continue saved run" for element in app.button)
    assert model.calls == []
