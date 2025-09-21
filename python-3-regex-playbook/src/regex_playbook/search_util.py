import re

def compare(patterns, text):
    for p in patterns:
        m = re.search(p, text)
        if m:
            print(f"✅ Found: '{p}' in: {text}")
        else:
            print(f"❌ No match for: '{p}'")
        print("-" * 40)

def substitute(patterns, text, replacement):
    out = text
    for p in patterns:
        new_out = re.sub(p, replacement, out)
        if new_out != out:
            print(f"✂️  Pattern '{p}' replaced")
            print(f"   Original: {out}")
            print(f"   New:      {new_out}")
            print("-" * 40)
        out = new_out
    return out

def parts(patterns, text):
    for p in patterns:
        pieces = re.split(p, text)
        if len(pieces) > 1:
            print(f"✂️  Split on '{p}' → {pieces}")
            print("-" * 40)

def list_matches(patterns, text):
    for p in patterns:
        matches = re.findall(p, text)
        if matches:
            print(f"🔎 '{p}' → {matches}")
        else:
            print(f"🚫 '{p}' → no matches")
        print("-" * 40)

def boundary(patterns, text):
    for p in patterns:
        word_pat = r"\b" + re.escape(p) + r"\b"
        if re.search(word_pat, text):
            print(f"✅ Whole word match for '{p}' in: {text}")
        else:
            print(f"❌ No whole-word match for '{p}'")
        print("-" * 40)

def run_flags(patterns, texts, flag):
    for p in patterns:
        for t in texts:
            m = re.search(p, t, flag)
            if m:
                print("\n✅ Match found!")
                print(f"Pattern: {p}")
                print(f"Text: {t}")
                print(f"Matched: {m.group()}")
            else:
                print("\n❌ No match found")
                print(f"Pattern: {p}")
                print(f"Text: {t}")
            print("-" * 40)
