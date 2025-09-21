import pytest
from regex_playbook.cleaning import clean_phone_number, clean_phone_number_lenient

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("1234567890", "123-456-7890"),
        ("123-456-7890", "123-456-7890"),
        ("(123) 456-7890", "123-456-7890"),
        ("123.456.7890", "123-456-7890"),
        ("  123   456   7890  ", "123-456-7890"),
        ("abc123-456-7890xyz", "123-456-7890"),
        ("+1 (123) 456-7890", None),  # strict version rejects leading country code
        ("12345678", None),
        ("", None),
    ],
)
def test_clean_phone_number_strict(raw, expected):
    assert clean_phone_number(raw) == expected


def test_clean_phone_number_lenient_accepts_country_code():
    assert clean_phone_number_lenient("+1 (123) 456-7890") == "123-456-7890"
    assert clean_phone_number_lenient("1-123-456-7890") == "123-456-7890"
    assert clean_phone_number_lenient("11234567890") == "123-456-7890"


def test_clean_phone_number_demo_no_capsys():
    raw = "Call me at (123) 456-7890!"
    cleaned = clean_phone_number(raw)
    print(f"raw: {raw} -> cleaned: {cleaned}")  # will print with -s
    assert cleaned == "123-456-7890"

