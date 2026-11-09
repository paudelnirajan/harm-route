"""Deterministic text transforms that make one condition from another (e.g. a script change).

Registered by name in TRANSFORMS and referenced from configs/languages.yaml, so no language is hardcoded
in scripts. Each transform is a pure str -> str function.
"""

# Serbian Cyrillic -> Latin (Gaj's alphabet). One Cyrillic letter -> one Latin letter or digraph, so the
# direction Cyrillic -> Latin is deterministic and loses nothing. (Latin -> Cyrillic is NOT: "nj" can be
# н+ј in words like "injekcija".) Capital digraphs are title-cased (Љ -> Lj), fine for sentence text.
_SRP = dict(zip(
    "абвгдђежзијклмнопрстћуфхцчш",
    ["a", "b", "v", "g", "d", "đ", "e", "ž", "z", "i", "j", "k", "l", "m", "n", "o", "p", "r", "s", "t",
     "ć", "u", "f", "h", "c", "č", "š"],
))
_SRP |= {"љ": "lj", "њ": "nj", "џ": "dž"}
_SRP |= {c.upper(): l[0].upper() + l[1:] for c, l in _SRP.items()}


def srp_cyrl_to_latn(text: str) -> str:
    return "".join(_SRP.get(ch, ch) for ch in text)


TRANSFORMS = {"srp_cyrl_to_latn": srp_cyrl_to_latn}
