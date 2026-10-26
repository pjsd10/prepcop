"""Deterministic, escaped HTML export; this module makes no external calls."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum
from html import escape
from typing import Any
from urllib.parse import urlsplit


def as_mapping(value: Any) -> dict[str, Any]:
    """Accept saved JSON mappings and Pydantic models without importing a model."""
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    raise TypeError("Expected a report model or mapping.")


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


def _html(value: Any) -> str:
    return escape(_text(value), quote=True)


def safe_source_url(value: Any) -> str | None:
    """Only permit plain HTTP(S) links, never uploaded or generated markup."""
    url = _text(value).strip()
    if not url or any(ord(char) < 32 or ord(char) == 127 for char in url):
        return None
    try:
        parsed = urlsplit(url)
        if parsed.scheme.lower() not in {"http", "https"}:
            return None
        if not parsed.hostname or parsed.username or parsed.password:
            return None
        # Accessing the port also rejects malformed/non-numeric ports.
        _ = parsed.port
    except ValueError:
        return None
    return url


def _paragraph(value: Any, class_name: str = "") -> str:
    content = _html(value).replace("\n", "<br>")
    return f'<p class="{class_name}">{content}</p>' if content else ""


def _list(items: Sequence[Any]) -> str:
    return "<ul>" + "".join(f"<li>{_html(item)}</li>" for item in items) + "</ul>" if items else ""


_STYLE = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body { margin: 0; background: #f6f4ef; color: #1d3045; font: 16px/1.6 system-ui, sans-serif; }
main { max-width: 900px; margin: 0 auto; padding: 48px 28px; }
h1, h2, h3, h4 { line-height: 1.25; }
h1 { font-size: 2.2rem; margin-bottom: 10px; }
h2 { margin-top: 36px; border-bottom: 1px solid #d7ddd9; padding-bottom: 10px; }
h3 { margin-top: 0; }
a { color: #006a6a; overflow-wrap: anywhere; }
.eyebrow, .muted { color: #536577; }
.eyebrow { text-transform: uppercase; letter-spacing: .08em; font-size: .8rem; }
.status { display: inline-block; padding: 4px 12px; border-radius: 20px; background: #e1eeea; font-weight: 650; }
.partial, .failed, .notice { background: #fff1d3; }
.notice { padding: 16px 20px; border-radius: 8px; }
.question { padding: 24px; margin: 18px 0; background: #fff; border: 1px solid #dce3df; border-radius: 10px; }
.question-number { color: #006a6a; font-size: .9rem; font-weight: 650; }
.source { margin: 18px 0; padding: 14px 18px; border-left: 3px solid #7da7a0; background: #fff; }
.source p { margin: 6px 0; }
blockquote { margin: 12px 0 0; padding-left: 16px; border-left: 2px solid #d7ddd9; }
table { width: 100%; border-collapse: collapse; margin: 12px 0; }
th, td { text-align: left; vertical-align: top; padding: 10px; border-bottom: 1px solid #d7ddd9; }
th { font-weight: 650; }
footer { margin-top: 40px; font-size: .88rem; color: #536577; }
@media (max-width: 600px) { main { padding: 24px 16px; } .question { padding: 18px; } table { font-size: .88rem; } }
@media print { body { background: white; font-size: 11pt; } main { max-width: none; padding: 0; } .question, .source { break-inside: avoid; } a { color: inherit; } }
"""


def _source_index(sources: Sequence[Any]) -> dict[str, tuple[dict[str, Any], str]]:
    result = {}
    for number, raw in enumerate(sources, 1):
        source = as_mapping(raw)
        identity = _text(source.get("id"))
        if identity and identity not in result:
            result[identity] = (source, f"source-{number}")
    return result


def _citations(source_ids: Sequence[str], sources: dict) -> str:
    citations = []
    for identity in dict.fromkeys(source_ids):
        stored = sources.get(identity)
        if stored and stored[0].get("status") == "accepted":
            citations.append(f'<a href="#{stored[1]}">[{_html(identity)}]</a>')
        else:
            citations.append(f'<span class="muted">[{_html(identity)}: support unavailable]</span>')
    return " ".join(citations)


def render_report_html(
    report: Any, *, brief: Any | None = None, sources: Sequence[Any] = ()
) -> str:
    """Render a preview or an already reviewed report without inventing support.

    Call ``build_reviewed_export`` for the download path. This function itself
    is deliberately usable before review so the candidate can inspect a report.
    """
    document = as_mapping(report)
    context = as_mapping(brief) if brief is not None else {}
    evidence = _source_index(sources)
    skills = {skill["id"]: skill for skill in context.get("skills", [])}
    status = _text(document.get("status", "partial"))
    if status not in {"complete", "partial", "failed"}:
        status = "partial"
    title = document.get("title") or "PrepCop preparation pack"
    questions = document.get("questions", [])
    sections = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'unsafe-inline\'; base-uri \'none\'; form-action \'none\'">',
        f"<title>{_html(title)}</title><style>{_STYLE}</style></head><body><main>",
        '<div class="eyebrow">PrepCop</div>',
        f"<h1>{_html(title)}</h1>",
        f'<span class="status {status}">{status.capitalize()}</span>',
        _paragraph(
            f"Report version {document.get('version', '—')} · "
            f"Brief version {document.get('brief_version', '—')} · "
            f"{len(questions)} of 6 practice questions",
            "muted",
        ),
        _paragraph(document.get("summary")),
    ]
    if status != "complete":
        sections.append(_paragraph(
            "This pack is incomplete. The available work and gaps are retained below; "
            "missing questions or support have not been filled in.", "notice"
        ))

    if context:
        sections.extend(["<h2>Your preparation brief</h2>", _paragraph(context.get("role"))])
        sections.append(_paragraph(context.get("summary")))
        sections.append(_paragraph(
            f"Interview stage: {context.get('interview_stage', 'Not specified')} · "
            f"Available study time: {context.get('preparation_minutes', '—')} minutes"
        ))
        selected = context.get("selected_topic_ids", [])
        rows = []
        for identity in selected:
            skill = skills.get(identity, {})
            rows.append("<tr>" + "".join(f"<td>{_html(value)}</td>" for value in (
                skill.get("name", identity),
                _text(skill.get("exposure", "not_sure")).replace("_", " "),
                _text(skill.get("desired_depth", "not_sure")).replace("_", " "),
                skill.get("notes", ""),
            )) + "</tr>")
        if rows:
            sections.append("<table><thead><tr><th>Selected topic</th><th>Self-reported exposure</th>"
                            "<th>Requested depth</th><th>Your notes</th></tr></thead><tbody>"
                            + "".join(rows) + "</tbody></table>")
        if context.get("deferred_skills"):
            sections.append(_paragraph("Deferred this run: " + ", ".join(context["deferred_skills"])))
        if context.get("uncertainties"):
            sections.append("<h3>Experience still unclear</h3>" + _list(context["uncertainties"]))

    sections.append("<h2>Practice questions</h2>")
    if not questions:
        sections.append(_paragraph("No supported practice questions are available in this report."))
    for number, raw in enumerate(questions, 1):
        question = as_mapping(raw)
        topic = skills.get(question.get("topic_id"), {}).get("name", question.get("topic_id", "Topic"))
        sections.extend([
            '<article class="question">',
            f'<p class="question-number">Question {number} · {_html(topic)} · '
            f'{_html(_text(question.get("depth", "not_sure")).replace("_", " "))}</p>',
            f'<h3>{_html(question.get("question"))}</h3>',
            _paragraph(question.get("explanation")),
        ])
        if question.get("source_ids"):
            sections.append('<p class="muted">Supporting sources: ' + _citations(question["source_ids"], evidence) + "</p>")
        sections.append("<h4>Key points your answer should cover</h4><ul>")
        for raw_point in question.get("key_points", []):
            point = as_mapping(raw_point)
            sections.append(f'<li>{_html(point.get("text"))} {_citations(point.get("source_ids", []), evidence)}</li>')
        sections.append("</ul>")
        if question.get("candidate_example_prompt"):
            sections.append(_paragraph(question["candidate_example_prompt"], "notice"))
        sections.append("</article>")

    sections.append("<h2>Study plan</h2>")
    plan = document.get("study_plan", [])
    if plan:
        total = sum(int(item.get("minutes", 0)) for item in plan)
        sections.append(_paragraph(f"Planned study time: {total} minutes"))
        sections.append("<table><thead><tr><th>Topic</th><th>Activity</th><th>Minutes</th></tr></thead><tbody>")
        for raw_activity in plan:
            activity = as_mapping(raw_activity)
            topic = skills.get(activity.get("topic_id"), {}).get("name", activity.get("topic_id", ""))
            sections.append(
                f'<tr><td>{_html(topic)}</td><td>{_html(activity.get("activity"))} '
                f'{_citations(activity.get("source_ids", []), evidence)}</td>'
                f'<td>{_html(activity.get("minutes"))}</td></tr>'
            )
        sections.append("</tbody></table>")
    else:
        sections.append(_paragraph("No study activities were available when this report was saved."))

    sections.append("<h2>Gaps and limitations</h2>")
    sections.append(_list(document.get("gaps", [])) or _paragraph("No additional gaps were recorded for this pack."))
    sections.append(_paragraph("Sources were assessed by the Research Coach. This pack has not had an independent evidence review."))
    sections.append("<h2>Sources and retained evidence</h2>")
    if not evidence:
        sections.append(_paragraph("No source evidence was saved."))
    for identity, (source, anchor) in evidence.items():
        title_text = _html(source.get("title") or source.get("url") or identity)
        url = safe_source_url(source.get("url"))
        linked_title = f'<a href="{_html(url)}" rel="noopener noreferrer">{title_text}</a>' if url else title_text
        sections.extend([
            f'<section class="source" id="{anchor}"><h3>[{_html(identity)}] {linked_title}</h3>',
            _paragraph(f"Status: {source.get('status', 'uncertain')} · Retrieved: {source.get('retrieved_at', 'Not recorded')}", "muted"),
            _paragraph(source.get("reason")),
        ])
        if source.get("excerpt"):
            sections.append(f'<blockquote>{_html(source["excerpt"])}</blockquote>')
        if not url:
            sections.append(_paragraph("A safe web link is unavailable for this source.", "muted"))
        sections.append("</section>")
    sections.append("<footer>Candidate experience is self-reported. Use your own real examples when practising. "
                    "This saved report does not grade interview readiness.</footer></main></body></html>")
    return "\n".join(sections)


def build_reviewed_export(state: Any) -> bytes:
    """Export exactly the reviewed current report and its approved brief."""
    saved = as_mapping(state)
    if not saved.get("report"):
        raise ValueError("There is no saved report to download yet.")
    report = as_mapping(saved["report"])
    brief = as_mapping(saved.get("brief", {}))
    if type(report.get("version")) is not int or report["version"] < 1:
        raise ValueError("The saved report does not have a valid review version.")
    if type(brief.get("version")) is not int or brief["version"] < 1:
        raise ValueError("The saved brief does not have a valid approval version.")
    if saved.get("report_reviewed_version") != report.get("version"):
        raise ValueError("Review the current report version before downloading it.")
    if not brief or report.get("brief_version") != brief.get("version"):
        raise ValueError("This report belongs to an earlier brief. Review the current preparation brief first.")
    if saved.get("approved_brief_version") != brief.get("version"):
        raise ValueError("The preparation brief must be approved before exporting its report.")
    return render_report_html(report, brief=brief, sources=saved.get("evidence", [])).encode("utf-8")
