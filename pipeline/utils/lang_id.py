import re

ENGLISH_REGEX = re.compile(r"^[a-z]+$")

ROMAN_URDU_HINTS = {
    "kya", "kab", "kaun", "kis", "hai", "hain", "ki",
    "ka", "mein", "se", "par", "tha", "thi", "bane",
    "bani", "jeeta", "jeete", "kitni", "kitna"
}

def identify_language(token: str) -> str:
    token = token.lower()

    if token in ROMAN_URDU_HINTS:
        return "ur"

    if ENGLISH_REGEX.match(token):
        return "en"

    return "ur"


def tag_code_switches(language_tags):
    switches = []
    for i in range(1, len(language_tags)):
        if language_tags[i] != language_tags[i - 1]:
            switches.append(i)
    return switches
