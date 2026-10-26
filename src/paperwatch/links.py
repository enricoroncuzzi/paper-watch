"""The bare domains in model-written text, shared by the draft check and the summary check."""

from __future__ import annotations

import re

# The top-level domains a research post links to. A curated set rather than "any two letters",
# because dotted code is everywhere in this corpus: torch.compile, nn.Module, np.array, train.py
# and model.to all look like hosts to a looser rule. A label that is also a common attribute or
# file extension (.py, .sh, .md, .rs, .cc, .pl, .to, .at, .id, .info, .page, .cloud) is left out.
_TOP_LEVEL = (
    "com",
    "org",
    "net",
    "edu",
    "gov",
    "mil",
    "int",
    "io",
    "ai",
    "dev",
    "co",
    "me",
    "ly",
    "gg",
    "tv",
    "fm",
    "so",
    "xyz",
    "tech",
    "blog",
    "news",
    "science",
    "research",
    "uk",
    "de",
    "fr",
    "ch",
    "cn",
    "jp",
    "kr",
    "ca",
    "us",
    "eu",
    "it",
    "nl",
    "se",
    "fi",
    "es",
    "be",
    "dk",
    "au",
    "nz",
    "il",
    "sg",
    "hk",
    "tw",
    "in",
    "br",
)
# A host: dotted labels ending in one of those domains, written in lower case so "ASP.NET" stays a
# product name. Not glued to a word, an email address or a longer dotted name on either side.
BARE_DOMAIN = re.compile(
    r"(?<![\w@.-])(?:[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?\.)+"
    rf"(?:{'|'.join(_TOP_LEVEL)})(?![\w-]|\.\w)"
)


def bare_domains(text: str) -> list[str]:
    """Every host-shaped token in the text, in order."""
    return [match.group(0) for match in BARE_DOMAIN.finditer(text)]
