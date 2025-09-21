# tests/test_security.py
import pytest
from regex_playbook.security import sanitize_user_input, check_for_injection, sanitize_and_check

SAFE_INPUTS = [
    "Jane_Doe-123",
    "budget 2025-09-21",
    "orders/report_01",
    "hello+world@example.com",
    "abc.DEF-12_34",
    "ＡＢＣ １２３",  # full-width; will normalize to ASCII-like
]

BAD_INPUTS = [
    "hello; rm -rf /",
    "foo && echo pwned",
    "$(whoami)",
    "__import__('os').system('calc')",
    "1; DROP TABLE users; --",
    "../etc/passwd",
    "`uname -a`",
    "subprocess.Popen('ls')",
]

@pytest.mark.security
@pytest.mark.parametrize("s", SAFE_INPUTS)
def test_sanitize_safe_inputs(s):
    sanitized, raw_hits, sanitized_hits = sanitize_and_check(s)
    print(f"\nSAFE: raw='{s}'  →  sanitized='{sanitized}'")
    print("raw hits:", [h.name for h in raw_hits])
    print("sanitized hits:", [h.name for h in sanitized_hits])
    assert not raw_hits, "Unexpected dangerous hits in safe input"
    assert sanitized  # shouldn't sanitize to empty


@pytest.mark.security
@pytest.mark.parametrize("s", BAD_INPUTS)
def test_detect_bad_inputs(s):
    sanitized, raw_hits, sanitized_hits = sanitize_and_check(s)
    print(f"\nBAD: raw='{s}'  →  sanitized='{sanitized}'")
    if raw_hits:
        for h in raw_hits:
            print(f"  ⚠ {h.name}: '{h.text}' at [{h.start}:{h.end}] (raw)")
    if sanitized_hits:
        for h in sanitized_hits:
            print(f"  ⚠ {h.name}: '{h.text}' at [{h.start}:{h.end}] (sanitized)")
    assert raw_hits, "Expected to detect dangerous content in raw input"
