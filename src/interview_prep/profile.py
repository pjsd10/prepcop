"""Profile Interviewer nodes and deterministic candidate-control boundaries.

Provider accounting, operation reuse and repair are owned by ``ctx``.  Human
pause/approval nodes live in the controller, separate from these model calls.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator

from .html_input import MAX_JD_CHARS, validate_input_text
from .models import Brief, Depth, Exposure, Probe, ProbeAnswer, Profile, StrictModel


MAX_PROBES = 5
_PROMPTS = Path(__file__).resolve().parents[2] / "prompts"


def _prompt(name: str) -> str:
    return (_PROMPTS / name).read_text(encoding="utf-8")


def _plain(value: Any) -> Any:
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else value


def _messages(prompt_name: str, data: dict) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _prompt(prompt_name)},
        {"role": "user", "content": "UNTRUSTED CANDIDATE DATA (JSON):\n" + json.dumps(data, ensure_ascii=False)},
    ]


def _key(operation: str, state: dict, data: dict) -> str:
    digest = hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
    return (
        f"profile:{operation}:input-{state.get('input_version', 1)}:"
        f"v-{state.get('profile_version', 1)}:{digest}"
    )


def _normalized(text: str) -> str:
    return " ".join(text.split()).casefold()


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item.strip() for item in items if item.strip()))


def validate_selected_topics(profile: Profile | dict, selected_topic_ids: list[str]) -> list[str]:
    """Use a single explicit selection; reject invalid choices, never trim them."""
    profile = Profile.model_validate(_plain(profile))
    if not isinstance(selected_topic_ids, list) or not 1 <= len(selected_topic_ids) <= 3:
        raise ValueError("Choose one, two or three research topics.")
    if any(not isinstance(topic_id, str) or not topic_id.strip() for topic_id in selected_topic_ids):
        raise ValueError("Research topic IDs must be nonempty strings.")
    if len(set(selected_topic_ids)) != len(selected_topic_ids):
        raise ValueError("Research topics must be unique.")
    if not set(selected_topic_ids) <= {skill.id for skill in profile.skills}:
        raise ValueError("Every selected research topic must belong to the current profile.")
    return list(selected_topic_ids)


def _answers(state: dict) -> list[ProbeAnswer]:
    answers = [ProbeAnswer.model_validate(_plain(item)) for item in state.get("answers", [])]
    if len(answers) > MAX_PROBES:
        raise ValueError("A run cannot contain more than five clarification replies.")
    for answer in answers:
        if len(answer.answer) > 6_000:
            raise ValueError("A clarification reply may contain at most 6,000 characters.")
        if not answer.skipped and not answer.answer.strip():
            raise ValueError("An unanswered clarification must be marked skipped.")
    return answers


def allowed_probe_skill_ids(state: dict) -> list[str]:
    """Cover every selected skill once before spending spare follow-up slots."""
    selected = validate_selected_topics(state["profile"], state.get("selected_topic_ids", []))
    covered = {answer.probe.skill_id for answer in _answers(state)}
    uncovered = [topic_id for topic_id in selected if topic_id not in covered]
    return uncovered or selected


def map_profile(state: dict, ctx: Any) -> dict:
    resume = validate_input_text(state.get("resume_text", ""), "Resume")
    jd = validate_input_text(state.get("jd_text", ""), "Job description", max_chars=MAX_JD_CHARS)
    data = {"resume_text": resume, "jd_text": jd}

    class ProfileMapping(Profile):
        @model_validator(mode="after")
        def check_mapping(self):
            if not self.role.strip():
                raise ValueError("The proposed role cannot be blank.")
            names: set[str] = set()
            for skill in self.skills:
                if not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", skill.id):
                    raise ValueError("Use stable lowercase skill IDs with letters, numbers, hyphens or underscores.")
                if not skill.name.strip() or _normalized(skill.name) in names:
                    raise ValueError("Skill names must be nonempty and unique.")
                names.add(_normalized(skill.name))
                if not skill.requirement.strip():
                    raise ValueError("Every mapped skill needs its relevant JD requirement.")
                if skill.resume_evidence and _normalized(skill.resume_evidence) not in _normalized(resume):
                    raise ValueError("Resume evidence must be an exact excerpt; use an empty string if absent.")
                if skill.confirmed:
                    raise ValueError("Only the candidate can confirm a proposed skill.")
            return self

    mapped = ctx.structured_call(_key("map", state, data), _messages("profile_mapping.txt", data), ProfileMapping)
    profile = ProfileMapping.model_validate(_plain(mapped))
    return {
        "profile": profile.model_dump(mode="json"),
        "profile_version": max(1, state.get("profile_version", 1)),
        "stage": "profile_review",
        "approved_brief_version": None,
    }


class ProbeDecision(StrictModel):
    probe: Probe | None = None
    stop_reason: str = ""


def next_probe(state: dict, ctx: Any) -> dict:
    profile = Profile.model_validate(state["profile"])
    selected = validate_selected_topics(profile, state.get("selected_topic_ids", []))
    answers = _answers(state)
    count = state.get("probe_count", 0)
    if not isinstance(count, int) or isinstance(count, bool) or not 0 <= count <= MAX_PROBES:
        raise ValueError("The persisted clarification counter is invalid.")
    if count < len(answers):
        raise ValueError("The persisted clarification counter cannot be less than the reply count.")
    if state.get("finish_probing"):
        return {"pending_probe": None, "probe_count": count, "finish_probing": True, "stage": "clarification"}
    if state.get("pending_probe"):
        pending = Probe.model_validate(state["pending_probe"])
        if pending.skill_id not in selected:
            raise ValueError("The pending clarification belongs to a topic that is no longer selected.")
        return {"pending_probe": pending.model_dump(mode="json"), "probe_count": count, "stage": "clarification"}
    if count >= MAX_PROBES:
        return {"pending_probe": None, "probe_count": count, "finish_probing": True, "stage": "clarification"}
    covered = {answer.probe.skill_id for answer in answers}
    uncovered = [topic_id for topic_id in selected if topic_id not in covered]
    allowed_ids = uncovered or selected
    previous_questions = {_normalized(answer.probe.question).rstrip("?.!") for answer in answers}

    class BoundedProbeDecision(ProbeDecision):
        @model_validator(mode="after")
        def check_policy(self):
            if self.probe is None:
                if uncovered:
                    raise ValueError("Give every selected skill an initial probe before choosing to stop.")
                if not self.stop_reason.strip():
                    raise ValueError("Explain briefly why further clarification is unnecessary.")
            else:
                if self.probe.skill_id not in allowed_ids:
                    raise ValueError(f"Choose a clarification skill from {allowed_ids}; initial coverage comes first.")
                if _normalized(self.probe.question).rstrip("?.!") in previous_questions:
                    raise ValueError("Do not repeat a question already asked or skipped.")
                if not self.probe.selection_reason.strip() or not self.probe.addresses_unknown.strip():
                    raise ValueError("State the useful selection reason and uncertainty this question addresses.")
            return self

    data = {
        "profile": profile.model_dump(mode="json"),
        "selected_topic_ids": selected,
        "allowed_skill_ids": allowed_ids,
        "initial_coverage_remaining": uncovered,
        "answers": [answer.model_dump(mode="json") for answer in answers],
        "remaining_probes": MAX_PROBES - count,
        "probe_number": count + 1,
    }
    result = ctx.structured_call(_key(f"probe-{count + 1}", state, data), _messages("profile_probe.txt", data), BoundedProbeDecision)
    decision = BoundedProbeDecision.model_validate(_plain(result))
    if decision.probe is None:
        return {"pending_probe": None, "probe_count": count, "finish_probing": True, "stage": "clarification"}
    return {
        "pending_probe": decision.probe.model_dump(mode="json"),
        "probe_count": count + 1,
        "finish_probing": False,
        "stage": "clarification",
    }


class SkillClarification(StrictModel):
    skill_id: str
    answer_index: int = Field(ge=0, le=4)
    answer_excerpt: str = Field(min_length=1)
    exposure: Exposure | None = None
    desired_depth: Depth | None = None
    notes: str = ""


class ResolvedUncertainty(StrictModel):
    uncertainty: str
    answer_index: int = Field(ge=0, le=4)
    answer_excerpt: str = Field(min_length=1)


class BriefSynthesis(StrictModel):
    summary: str = Field(min_length=1, max_length=4_000)
    clarifications: list[SkillClarification] = Field(default_factory=list, max_length=5)
    uncertainties: list[str] = Field(default_factory=list, max_length=30)
    resolved_uncertainties: list[ResolvedUncertainty] = Field(default_factory=list, max_length=30)


def make_brief(state: dict, ctx: Any) -> dict:
    profile = Profile.model_validate(state["profile"])
    selected = validate_selected_topics(profile, state.get("selected_topic_ids", []))
    answers = _answers(state)
    version = state.get("profile_version", 1)
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise ValueError("The profile version must be a positive integer.")
    # Validate authoritative controller fields before spending a model attempt.
    authoritative = Brief(
        version=version,
        role=profile.role,
        seniority=profile.seniority,
        skills=profile.skills,
        selected_topic_ids=selected,
        preparation_minutes=state.get("preparation_minutes", 60),
        interview_stage=state.get("interview_stage", "Not specified"),
    )

    def check_answer_reference(index: int, excerpt: str) -> ProbeAnswer:
        if index >= len(answers):
            raise ValueError("A clarification must reference an existing answer.")
        answer = answers[index]
        if answer.skipped:
            raise ValueError("A skipped question cannot support a new claim or resolve an unknown.")
        if not excerpt.strip() or excerpt not in answer.answer:
            raise ValueError("Supporting excerpts must occur exactly in the referenced candidate answer.")
        return answer

    class GroundedBriefSynthesis(BriefSynthesis):
        @model_validator(mode="after")
        def check_answer_support(self):
            changed_ids: set[str] = set()
            for update in self.clarifications:
                if update.skill_id not in selected:
                    raise ValueError("Clarifications may update only selected skills.")
                if update.skill_id in changed_ids:
                    raise ValueError("Consolidate each skill's supported clarifications into one update.")
                changed_ids.add(update.skill_id)
                answer = check_answer_reference(update.answer_index, update.answer_excerpt)
                if answer.probe.skill_id != update.skill_id:
                    raise ValueError("A clarification must reference an answer about the same skill.")
            for resolved in self.resolved_uncertainties:
                if resolved.uncertainty not in profile.uncertainties:
                    raise ValueError("Resolve only an uncertainty present in the current profile.")
                check_answer_reference(resolved.answer_index, resolved.answer_excerpt)
            return self

    data = {
        "profile": profile.model_dump(mode="json"),
        "selected_topic_ids": selected,
        "answers": [answer.model_dump(mode="json") for answer in answers],
        "preparation_minutes": authoritative.preparation_minutes,
        "interview_stage": authoritative.interview_stage,
        "candidate_finished_early": bool(state.get("finish_probing")) and state.get("probe_count", 0) < MAX_PROBES,
    }
    result = ctx.structured_call(_key("brief", state, data), _messages("profile_brief.txt", data), GroundedBriefSynthesis)
    synthesis = GroundedBriefSynthesis.model_validate(_plain(result))
    updates = {update.skill_id: update for update in synthesis.clarifications}
    skills = []
    for skill in profile.skills:
        revised = skill.model_copy(deep=True)
        notes = [skill.notes]
        update = updates.get(skill.id)
        if update:
            if update.exposure is not None:
                revised.exposure = update.exposure
            if update.desired_depth is not None:
                revised.desired_depth = update.desired_depth
            if revised.exposure != skill.exposure or revised.desired_depth != skill.desired_depth:
                revised.confirmed = False
            notes.append(update.notes)
        for answer in answers:
            if answer.probe.skill_id == skill.id:
                if answer.skipped:
                    notes.append(f"Skipped clarification; still unknown: {answer.probe.addresses_unknown}")
                else:
                    # Preserve exact human corrections even if the synthesis omits one.
                    notes.append(f"Candidate clarification: {answer.answer}")
        revised.notes = "\n".join(_unique(notes))
        skills.append(revised)
    resolved = {item.uncertainty for item in synthesis.resolved_uncertainties}
    unknowns = [item for item in profile.uncertainties if item not in resolved]
    unknowns.extend(synthesis.uncertainties)
    for answer in answers:
        if answer.skipped:
            unknowns.append(f"{answer.probe.skill_id}: skipped clarification — {answer.probe.addresses_unknown}")
    covered = {answer.probe.skill_id for answer in answers}
    for skill in skills:
        if skill.id in selected:
            if skill.id not in covered:
                unknowns.append(f"{skill.name}: no clarification answer was collected before finishing.")
            if skill.exposure == "not_sure":
                unknowns.append(f"{skill.name}: actual exposure is not sure.")
            if skill.desired_depth == "not_sure":
                unknowns.append(f"{skill.name}: desired practice depth is not sure; confirm or disclose an assumed depth.")
    unknowns = _unique(unknowns)
    revised_profile = profile.model_copy(update={"skills": skills, "uncertainties": unknowns})
    brief = authoritative.model_copy(update={
        "skills": skills,
        "summary": synthesis.summary,
        "uncertainties": unknowns,
        "deferred_skills": [skill.name for skill in skills if skill.id not in selected],
    })
    return {
        "profile": revised_profile.model_dump(mode="json"),
        "brief": brief.model_dump(mode="json"),
        "pending_probe": None,
        "stage": "brief_review",
        "approved_brief_version": None,
    }
