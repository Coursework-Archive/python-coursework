# src/regex_playbook/security_demo.py
from regex_playbook.security import sanitize_and_check

def main():
    try:
        raw = input("Enter value: ")
    except KeyboardInterrupt:
        return
    sanitized, raw_hits, sanitized_hits = sanitize_and_check(raw)

    print("\nSanitized:", sanitized or "(empty after sanitization)")
    if raw_hits:
        print("⚠ Detected potentially dangerous content in RAW input:")
        for h in raw_hits:
            print(f"  - {h.name}: '{h.text}' at [{h.start}:{h.end}]")
    else:
        print("✅ No dangerous patterns detected in raw input.")

    if sanitized_hits:
        print("⚠ Still dangerous after sanitization!")
    else:
        print("✅ Sanitized input appears clean (heuristically).")

if __name__ == "__main__":
    main()
