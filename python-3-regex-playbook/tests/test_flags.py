import re
from regex_playbook.search_util import run_flags
from regex_playbook import patterns as P

def test_flags():
    run_flags(
        patterns=[r"hello"],
        texts=["Hello, world"],
        flag=re.IGNORECASE,
    )
    run_flags(
        patterns=[P.HELLO_WORLD],
        texts=["hello\nworld"],
        flag=re.DOTALL,
    )
