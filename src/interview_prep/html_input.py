"""Bounded, inert text extraction for the static HTML resume pilot.

The returned string is candidate-editable text, never trusted markup.  This
module does not render HTML, execute code, open URLs, or load external assets.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import PurePath


MAX_UPLOAD_BYTES = 1_000_000
MAX_RESUME_CHARS = 40_000
MAX_JD_CHARS = 30_000
MIN_TEXT_CHARS = 40
MAX_HTML_ELEMENTS = 50_000
MAX_HTML_DEPTH = 256


class InputValidationError(ValueError):
    """A user-correctable input error that must stop provider work."""


_VOID_TAGS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link",
    "meta", "param", "source", "track", "wbr",
}
_EXCLUDED_TAGS = {
    "head", "title", "script", "style", "noscript", "template", "svg",
    "math", "iframe", "object", "embed", "canvas",
}
_BLOCK_TAGS = {
    "address", "article", "aside", "blockquote", "body", "caption", "dd",
    "details", "div", "dl", "dt", "fieldset", "figcaption", "figure",
    "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6", "header",
    "hr", "li", "main", "nav", "ol", "p", "pre", "section", "summary",
    "table", "tbody", "tfoot", "thead", "tr", "ul",
}
_CONTENT_TAGS = _BLOCK_TAGS | {"html", "span", "td", "th"}


def validate_input_text(
    text: str,
    label: str = "Input",
    *,
    max_chars: int = MAX_RESUME_CHARS,
) -> str:
    """Validate confirmed plain text without silently removing qualifiers."""
    if not isinstance(text, str):
        raise InputValidationError(f"{label} must be text.")
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(text) > max_chars:
        raise InputValidationError(
            f"{label} is too long ({len(text):,} characters). "
            f"The limit is {max_chars:,}; shorten it and review again."
        )
    if any(ord(char) < 32 and char not in "\n\t" for char in text):
        raise InputValidationError(f"{label} contains unsupported control characters.")
    if len(text) < MIN_TEXT_CHARS or sum(char.isalpha() for char in text) < 20:
        raise InputValidationError(
            f"{label} does not contain enough readable text. "
            "Provide meaningful experience or role details and review the text."
        )
    return text


class _ResumeTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.stack: list[tuple[str, bool]] = []
        self.has_content_tag = False
        self.element_count = 0

    @property
    def suppressed(self) -> bool:
        return bool(self.stack and self.stack[-1][1])

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.element_count += 1
        if self.element_count > MAX_HTML_ELEMENTS or len(self.stack) >= MAX_HTML_DEPTH:
            raise InputValidationError(
                "The HTML structure is too complex. Export a simple static HTML resume."
            )
        attrs_map = dict(attrs)
        inline_style = re.sub(r"\s+", "", attrs_map.get("style") or "").lower()
        hidden = (
            "hidden" in attrs_map
            or (attrs_map.get("aria-hidden") or "").strip().lower() == "true"
            or bool(re.search(r"(?:^|;)display:none(?:!important)?(?:;|$)", inline_style))
            or bool(re.search(r"(?:^|;)visibility:hidden(?:!important)?(?:;|$)", inline_style))
        )
        excluded = self.suppressed or tag in _EXCLUDED_TAGS or hidden
        if not excluded:
            self.has_content_tag |= tag in _CONTENT_TAGS
            if tag in _BLOCK_TAGS or tag == "br":
                self.parts.append("\n")
            elif tag in {"td", "th"}:
                # A row remains readable, with its cells explicitly separated.
                if self.parts and not self.parts[-1].endswith("\n"):
                    self.parts.append(" | ")
        if tag not in _VOID_TAGS:
            self.stack.append((tag, excluded))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag not in _VOID_TAGS:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        was_suppressed = self.suppressed
        for position in range(len(self.stack) - 1, -1, -1):
            if self.stack[position][0] == tag:
                del self.stack[position:]
                break
        if not was_suppressed and tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.suppressed:
            self.parts.append(data)

    def text(self) -> str:
        # Preserve paragraph/list/table boundaries; only collapse layout spacing.
        lines = (re.sub(r"\s+", " ", line).strip() for line in "".join(self.parts).splitlines())
        return "\n".join(line for line in lines if line)


def parse_resume_html(
    data: bytes,
    filename: str,
    *,
    max_bytes: int = MAX_UPLOAD_BYTES,
    max_chars: int = MAX_RESUME_CHARS,
) -> str:
    """Return plain resume text or a useful validation error.

    UTF-8 (with or without its BOM) is the supported pilot encoding.  Complex
    layout, external CSS visibility and JavaScript-generated content are outside
    this extractor's guarantees; the editable preview is always required.
    """
    if not isinstance(filename, str) or PurePath(filename).suffix.lower() not in {".html", ".htm"}:
        raise InputValidationError(
            "Unsupported file type. This pilot accepts .html or .htm. "
            "Export to a supported format; renaming the extension does not convert a document."
        )
    if not isinstance(data, bytes):
        raise InputValidationError("The uploaded resume must be file bytes.")
    if not data:
        raise InputValidationError("The uploaded resume is empty.")
    if len(data) > max_bytes:
        raise InputValidationError(
            f"The uploaded resume is too large. The limit is {max_bytes:,} bytes."
        )
    if data.startswith((b"%PDF", b"PK\x03\x04", b"\xd0\xcf\x11\xe0", b"\x89PNG", b"\xff\xd8")):
        raise InputValidationError(
            "This file is not static HTML. Export it as HTML; renaming the extension is insufficient."
        )
    try:
        markup = data.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as exc:
        raise InputValidationError(
            "The resume encoding is unsupported. Save or export the HTML as UTF-8 and try again."
        ) from exc
    if "\x00" in markup:
        raise InputValidationError("This file contains binary data and is not supported static HTML.")
    parser = _ResumeTextParser()
    try:
        parser.feed(markup)
        parser.close()
    except InputValidationError:
        raise
    except (ValueError, AssertionError) as exc:
        raise InputValidationError("The HTML could not be read. Export a simple static HTML resume.") from exc
    if not parser.has_content_tag:
        raise InputValidationError(
            "No supported HTML content was found. Export a simple static HTML resume."
        )
    try:
        return validate_input_text(parser.text(), "Extracted resume", max_chars=max_chars)
    except InputValidationError as exc:
        if "enough readable text" in str(exc):
            raise InputValidationError(
                "The HTML does not contain enough readable resume text. "
                "Images and JavaScript-generated content are unsupported; export a text-based static HTML resume."
            ) from exc
        raise
