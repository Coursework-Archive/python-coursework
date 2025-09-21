import re

def clean_phone_number(phone_number: str) -> str | None:
    """
    Strip everything that isn't a digit. If we end up with exactly 10 digits,
    format as XXX-XXX-XXXX. Otherwise return None.
    """
    digits_only = re.sub(r"\D", "", phone_number)
    if len(digits_only) != 10:
        return None
    return f"{digits_only[:3]}-{digits_only[3:6]}-{digits_only[6:]}"


# Optional: a lenient variant that accepts a leading country code '1'
def clean_phone_number_lenient(phone_number: str) -> str | None:
    digits = re.sub(r"\D", "", phone_number)
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    if len(digits) != 10:
        return None
    return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
