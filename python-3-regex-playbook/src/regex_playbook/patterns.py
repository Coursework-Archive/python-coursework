# Common regex patterns you reuse
PHONE_DASHED = r'(\d{3})-(\d{3})-(\d{4})'
DATE_YMD = r'\d{4}-\d{2}-\d{2}'
PRICE = r'\$?\d+(?:\.\d{2})?'
HELLO_WORLD = r'hello.*world'  # pair with re.DOTALL
SAMPLE_WORD = r'\bsample\b'   # whole word
PHONE = r'\b\d{3}-\d{3}-\d{4}\b'
EMAIL = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
ZIP   = r'\b\d{5}(?:-\d{4})?\b'
ORDER = r'\b[A-Z]{2}-\d{4}-\d{4}\b'      # e.g., AB-1234-5678
HEX   = r'#[A-Fa-f0-9]{6}\b'