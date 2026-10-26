"""Shared contracts. Models propose facts; only controller events grant approval."""
from typing import Any, Literal, TypedDict
from pydantic import BaseModel, ConfigDict, Field, model_validator

Exposure = Literal["not_used", "studied_tutorial", "personal_course_project", "used_at_work", "not_sure"]
Depth = Literal["beginner", "intermediate", "advanced", "not_sure"]

class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

class Skill(StrictModel):
    id: str
    name: str
    requirement: str = ""
    resume_evidence: str = ""
    exposure: Exposure = "not_sure"
    desired_depth: Depth = "not_sure"
    notes: str = ""
    confirmed: bool = False

class Profile(StrictModel):
    role: str
    seniority: str = ""
    skills: list[Skill] = Field(min_length=1, max_length=30)
    uncertainties: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_skills(self):
        if len({s.id for s in self.skills}) != len(self.skills):
            raise ValueError("Skill IDs must be unique")
        return self

class Probe(StrictModel):
    skill_id: str
    question: str = Field(min_length=5, max_length=900)
    selection_reason: str
    addresses_unknown: str

class ProbeAnswer(StrictModel):
    probe: Probe
    answer: str = ""
    skipped: bool = False

class Brief(StrictModel):
    version: int = Field(ge=1)
    role: str
    seniority: str = ""
    skills: list[Skill]
    selected_topic_ids: list[str] = Field(min_length=1, max_length=3)
    preparation_minutes: int = Field(ge=10, le=600)
    interview_stage: str = "Not specified"
    summary: str = ""
    uncertainties: list[str] = Field(default_factory=list)
    deferred_skills: list[str] = Field(default_factory=list)

class Evidence(StrictModel):
    id: str
    topic_id: str
    title: str
    url: str
    text: str = ""
    excerpt: str = ""
    retrieved_at: str
    status: Literal["pending", "accepted", "rejected", "uncertain"] = "pending"
    readable: bool | None = None
    reason: str = ""

class AnswerPoint(StrictModel):
    text: str
    source_ids: list[str] = Field(default_factory=list)

class PracticeQuestion(StrictModel):
    id: str
    topic_id: str
    question: str
    depth: Depth
    explanation: str
    key_points: list[AnswerPoint]
    source_ids: list[str] = Field(default_factory=list)
    candidate_example_prompt: str = ""

class StudyActivity(StrictModel):
    topic_id: str
    activity: str
    minutes: int = Field(gt=0, le=600)
    source_ids: list[str] = Field(default_factory=list)

class PrepReport(StrictModel):
    version: int = Field(ge=1)
    brief_version: int = Field(ge=1)
    status: Literal["complete", "partial", "failed"]
    title: str
    summary: str = ""
    questions: list[PracticeQuestion] = Field(default_factory=list, max_length=6)
    study_plan: list[StudyActivity] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)

class RunState(TypedDict, total=False):
    last_event_id: str | None
    run_id: str
    stage: str
    resume_text: str
    jd_text: str
    preparation_minutes: int
    interview_stage: str
    input_version: int
    profile_version: int
    profile: dict
    selected_topic_ids: list[str]
    answers: list[dict]
    probe_count: int
    pending_probe: dict | None
    finish_probing: bool
    brief: dict
    approved_brief_version: int | None
    research_messages: list[Any]
    pending_tool: dict | None
    candidates: list[dict]
    evidence: list[dict]
    research_step: int
    research_done: bool
    source_expansion_requested: bool
    report: dict
    report_reviewed_version: int | None
    error: str | None
    events: list[dict]
    counters: dict
    pending: dict | None
