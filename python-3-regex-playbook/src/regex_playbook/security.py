# src/regex_playbook/security.py
from __future__ import annotations
import re
import unicodedata
from dataclasses import dataclass
from typing import List, Tuple

# ---- Heuristic patterns that often indicate injection attempts ----
# (No regex can guarantee safety in all contexts; these are pragmatic tripwires.)
_DANGEROUS_PATTERNS: list[tuple[str, str]] = [
    # Shell metacharacters & chaining
    ("shell_metachar", r"[;&|`><]"),
    ("double_pipe_or_andand", r"\|\||&&"),
    ("subshell", r"\$\("),  # $(whoami)
    ("backticks", r"`[^`]*`"),
    # Python / system execution
    ("python_exec", r"\b(eval|exec|__import__|compile|globals|locals)\b"),
    ("python_os_subprocess", r"\b(os\.|subprocess\.|system\()"),
    # SQL-ish
    ("sql_keywords", r"(?i)\b(UNION|SELECT|INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|EXEC)\b"),
    ("sql_comment_or_quote", r"--|/\*|\*/|;"),
    # Path traversal
    ("path_traversal", r"\.\./|\.\.\\"),
]

_COMPILED = [(name, re.compile(pattern)) for name, pattern in _DANGEROUS_PATTERNS]


@dataclass(frozen=True)
class MatchHit:
    name: str
    text: str
    start: int
    end: int


def normalize_unicode(s: str) -> str:
    """Normalize to NFKC so weird-width Unicode becomes ASCII-like where possible."""
    return unicodedata.normalize("NFKC", s)


def strip_control_chars(s: str) -> str:
    """Remove non-printable control chars (but keep newlines/tabs if you prefer)."""
    # Remove all C0/C1 controls except \n and \t; tweak as needed.
    return re.sub(r"[^\x09\x0A\x20-\x7E]", "", s)


def collapse_whitespace(s: str) -> str:
    """Collapse runs of whitespace to a single space; trim ends."""
    return re.sub(r"\s+", " ", s).strip()


def whitelist_filter(s: str, allowed_char_class: str = r"A-Za-z0-9 _\.\-@:+,/#") -> str:
    """
    Keep only characters from the allowed class (inside a [...]).
    Default allows letters, digits, space, dot, dash, underscore, @:+, comma, slash, #.
    """
    return re.sub(fr"[^{allowed_char_class}]", "", s)


def sanitize_user_input(
    s: str,
    *,
    keep_newlines: bool = False,
    allowed_char_class: str = r"A-Za-z0-9 _\.\-@:+,/#"
) -> str:
    """A conservative sanitizer suitable for 'search term', 'username', etc."""
    s = normalize_unicode(s)
    s = strip_control_chars(s if keep_newlines else s.replace("\n", " "))
    s = collapse_whitespace(s)
    s = whitelist_filter(s, allowed_char_class=allowed_char_class)
    return s


def check_for_injection(s: str) -> List[MatchHit]:
    """Return a list of hits for dangerous patterns; empty list means 'no hits'."""
    hits: List[MatchHit] = []
    for name, rx in _COMPILED:
        for m in rx.finditer(s):
            hits.append(MatchHit(name=name, text=m.group(), start=m.start(), end=m.end()))
    return hits


def sanitize_and_check(raw: str) -> tuple[str, List[MatchHit], List[MatchHit]]:
    """
    Convenience: returns (sanitized, hits_in_raw, hits_in_sanitized).
    Useful to see if sanitation removed suspicious content.
    """
    raw_hits = check_for_injection(raw)
    sanitized = sanitize_user_input(raw)
    sanitized_hits = check_for_injection(sanitized)
    return sanitized, raw_hits, sanitized_hits
