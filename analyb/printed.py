"""Printed-value manifest: record each number exactly as it is typeset.

A generator that prints numbers into a report or plotbook records, for each
one, a ``key``, the numeric ``value``, the ``text`` a reader sees, and
optionally the ``source`` role it came from, a ``context`` label that shares
its table row or line, and the ``section``/``page`` it belongs to. A review
gate can then check that every recorded ``text`` really appears in the final
PDF. Recording is off unless a manifest is active, so callers may use
:func:`printed` unconditionally::

    with printed.value_manifest("values.json", producer="generate_report.py",
                                sources={"evidence": evidence_json}):
        cell = printed.printed("ratio/ptl1", ratio, ".6f", context="ptl1",
                               source="evidence")
        lines.append("ptl1 & " + cell + " \\\\")

Matching rule (``MATCH_RULE``, stored in the manifest), implemented by
:func:`find_printed`. Unary minus glyphs fold to ``-``; dash glyphs remain
separators. Unicode decimal digits fold to ASCII. One tokenizer,
:func:`numeric_tokens`, splits both the extracted PDF line and the record's
``text`` into complete numeric tokens (sign, comma groups, fraction, exponent
and an optional ``%`` or unit suffix, each with the typesetting whitespace
TeX may insert). A ``text`` that is exactly one such token matches only a
line containing a token with the same canonical form, so ``3.2`` never
matches ``-3.2`` or ``− 3.2``, ``12.5%`` never ``-12.5%``, ``234.5`` never
``1, 234.5``. Unsupported or ambiguous candidates are consumed whole and
rejected. Other text keeps a literal rule (characters in order with optional
whitespace), but every numeric character must belong to a supported token,
and each token must also occur on the line. ``context`` must occur on the
same extracted line as a whole token sequence; ``_`` or whitespace in it matches ``_`` or whitespace,
because pdflatex's default encoding extracts ``\\_`` as a space. Only text
presence and the row context are verified: which cell of the row holds the
number (value or error, which column) is not.
"""

import contextvars
import hashlib
import json
import math
import numbers
import os
import re
import unicodedata
from contextlib import contextmanager

SCHEMA = "analyb.printed_values"
SCHEMA_VERSION = 4
MATCH_RULE = (
    "v4; fold unary minus U+2212 U+FE63 U+FF0D to '-' and Unicode decimal digits "
    "to ASCII; preserve dash U+2012 U+2013 U+2010 U+2011 U+2014 as separators "
    "and treat ASCII '-' runs as placeholders; "
    "one maximal-candidate scanner for line and record text: optional unary '+' "
    "or '-' with zero or one non-tab, non-linebreak whitespace character after "
    "it; a longer sign gap is ambiguous and rejects the following candidate; "
    "a sign directly after a word, '.', ')' or another sign is invalid; a sign "
    "directly after a numeric candidate without whitespace on either side is "
    "a compact range separator; a spaced sign after a numeric candidate with "
    "at most one intervening whitespace character is ambiguous and rejected; "
    "mantissa digits with optional '.'+digits, or leading '.'+digits; grouping "
    "commas allow whitespace on both sides and require a first group of one to "
    "three digits and subsequent groups of exactly three digits, before the "
    "fraction; optional exponent e/E, optional sign and digits with whitespace "
    "around e/E and its sign; optional suffix with any preceding whitespace: "
    "% GeV MeV TeV keV eV fb-1 pb-1 fb pb nb mb mm cm rad mrad ns ps; "
    "malformed groups, fractions or exponents, nondecimal numerals, attached "
    "unsupported words and word-adjacent candidates are consumed and rejected "
    "whole, never rescanned inside; canonical form removes whitespace; record "
    "text containing any numeric character outside accepted spans fails; an "
    "exactly one-token text matches an equal canonical line token; other text "
    "matches its characters in order with optional whitespace (dash glyphs "
    "fold to '-' only for this literal comparison), Unicode decimal boundaries, "
    "and every text token also present as a complete line token; context on the "
    "same line matches whole Unicode word boundaries, with '_' and whitespace "
    "interchangeable; only row presence, not cells or arithmetic, is verified"
)

_MINUS = str.maketrans({c: "-" for c in "\u2212\ufe63\uff0d"})
_DASH = str.maketrans({c: "-" for c in "\u2012\u2013\u2010\u2011\u2014"})
_UNITS = ("fb-1", "pb-1", "mrad", "GeV", "MeV", "TeV", "keV", "eV", "rad",
          "fb", "pb", "nb", "mb", "mm", "cm", "ns", "ps", "%")
# Candidate consumption is deliberately broader than this supported grammar.
# Validate only after consumption, so a malformed token cannot expose a prefix.
_TOKEN = re.compile(
    r"[+-]?(?:(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?|\.[0-9]+)"
    r"(?:[eE][+-]?[0-9]+)?(?:" + "|".join(re.escape(u) for u in _UNITS) + r")?"
)
_active = contextvars.ContextVar("analyb_printed_manifest", default=None)


def normalize_text(text):
    """Fold unary minus glyphs and decimal digits, preserving dash identity."""
    return "".join(str(unicodedata.decimal(c)) if c.isdecimal() else c
                   for c in text.translate(_MINUS))


def _word(c):
    return c.isalnum() or c == "_"


def _skip_space(s, i):
    while i < len(s) and s[i].isspace():
        i += 1
    return i


def _digits(s, i):
    # Consume unsupported numerals too, so no interior fragment can reappear.
    while i < len(s) and s[i].isnumeric():
        i += 1
    return i


def _numeric_start(s, i):
    return i < len(s) and (s[i].isnumeric() or (
        s[i] == "." and i + 1 < len(s) and s[i + 1].isnumeric()))


def _numeric_end(s, i):
    if i < len(s) and s[i] == ".":
        i += 1
    return _digits(s, i)


def _sign_end(s, i):
    while i < len(s) and s[i] in "+-":
        i = _skip_space(s, i + 1)
    return i


def _number_at(s, i, previous_end):
    """Consume one maximal candidate; return its end and whether it is supported."""
    j, valid = i, True
    if s[j] in "+-":
        j = _sign_end(s, j)
        gap = s[i + 1:j]
        valid = (len(gap) <= 1 and not any(c in "\t\r\n" for c in gap)
                 and not (i > 0 and (_word(s[i - 1]) or s[i - 1] in ".)+-")))
        # PDF extraction cannot distinguish a binary minus from a spaced unary
        # sign in neighbouring cells. Reject the ambiguous operand as a whole.
        if (gap and previous_end is not None and i - previous_end <= 1
                and not s[previous_end:i].strip()):
            valid = False
    elif i > 0 and (_word(s[i - 1]) or s[i - 1] == "."):
        valid = False

    end = _numeric_end(s, j)
    suffix = False
    while end < len(s):
        k = _skip_space(s, end)
        if k < len(s) and s[k] == ",":
            # Include malformed comma runs and signed groups in the candidate.
            start = k
            while start < len(s) and s[start] == ",":
                start = _skip_space(s, start + 1)
            start = _sign_end(s, start)
            if _numeric_start(s, start):
                end = _numeric_end(s, start)
                continue
            break  # A prose comma, not a numeric continuation.
        if s[end] == ".":
            start = end
            while start < len(s) and s[start] == ".":
                start += 1
            if start < len(s) and s[start].isnumeric():
                end = _digits(s, start)
                continue
        if k < len(s) and s[k] in "eE":
            start = _skip_space(s, k + 1)
            signed = start < len(s) and s[start] in "+-"
            if signed:
                start = _sign_end(s, start)
            # eV is a suffix, while even an incomplete or repeated exponent
            # must be consumed before the supported grammar is checked.
            if (signed or _numeric_start(s, start) or start == len(s)
                    or not _word(s[start])):
                end = _numeric_end(s, start)
                continue
        if not suffix:
            for unit in _UNITS:
                stop = k + len(unit)
                # Consume even a suffix with an attached invalid continuation;
                # the full grammar must reject it without exposing a prefix.
                # An alphabetic prefix of a separate word is not a unit.
                if s.startswith(unit, k) and (k == end or not unit.isalpha()
                                             or stop == len(s) or not _word(s[stop])):
                    end, suffix = stop, True
                    break
            else:
                stop = None
            if stop is not None:
                continue
        if _word(s[end]):
            # Consume attached unknown words one character at a time so a
            # later e+digits continuation is swallowed too, not rescanned.
            end += 1
            continue
        break
    return end, valid and _TOKEN.fullmatch(_canonical(s[i:end])) is not None


def _scan(s):
    """Accepted (start, end) spans; rejected candidates are consumed whole."""
    spans, i, previous_end = [], 0, None
    while i < len(s):
        start = _sign_end(s, i)
        if s[i] == "-" and i + 1 < len(s) and s[i + 1] == "-":
            # A run of ASCII dashes is a placeholder, never a unary sign.
            while i < len(s) and s[i] == "-":
                i += 1
            continue
        if s[i] in "+-" and previous_end == i and start == i + 1:
            # Compact range (0-5), not a signed right-hand endpoint.
            i += 1
            continue
        if _numeric_start(s, start):
            end, valid = _number_at(s, i, previous_end)
            if valid:
                spans.append((i, end))
            previous_end = end if valid else None
            i = end
        else:
            i += 1
    return spans


def _canonical(token):
    return re.sub(r"\s+", "", token)


def numeric_tokens(line):
    """Canonical numeric tokens of ``line`` (see ``MATCH_RULE``)."""
    line = normalize_text(line)
    return [_canonical(line[a:b]) for a, b in _scan(line)]


def numeric_text(text):
    """Canonical form if ``text`` is exactly one numeric token, else None."""
    stripped = normalize_text(text).strip()
    spans = _scan(stripped)
    if len(spans) == 1 and spans[0] == (0, len(stripped)):
        return _canonical(stripped)
    return None


def text_pattern(text):
    """Pattern for non-numeric ``text``: characters in order, optional whitespace."""
    chars = [re.escape(c) for c in normalize_text(text).translate(_DASH) if not c.isspace()]
    if not chars:
        raise ValueError("printed text must contain a non-space character")
    return re.compile(r"(?<![\d.])" + r"\s*".join(chars) + r"(?!\d|\.\d)")


def context_pattern(context):
    """Pattern for a ``context`` label as a whole token sequence.

    Each whitespace run or ``_`` matches whitespace or ``_`` (pdflatex's
    default encoding extracts an escaped ``_`` as a space); the label may not
    continue into a neighbouring letter, digit or ``_``.
    """
    parts = re.split(r"(\s+|_)", normalize_text(context).strip())
    body = "".join(r"(?:\s+|_)" if part.isspace() or part == "_" else re.escape(part)
                   for part in parts if part)
    if not re.search(r"[^\W_]", context):
        raise ValueError("context must contain a letter or digit")
    return re.compile(r"(?<!\w)" + body + r"(?!\w)")


def find_printed(lines, text, context=None):
    """Return indices of ``lines`` where ``text`` (and ``context``, if given) match."""
    text = normalize_text(text).strip()
    spans = _scan(text)
    # Literal matching must never bypass a rejected number (e.g. 12kg).
    if any(c.isnumeric() and not any(a <= i < b for a, b in spans)
           for i, c in enumerate(text)):
        return []
    parts = [_canonical(text[a:b]) for a, b in spans]
    if len(spans) == 1 and spans[0] == (0, len(text)):
        matches = lambda line: parts[0] in numeric_tokens(line)  # noqa: E731
    else:
        pattern = text_pattern(text)

        def matches(line):
            tokens = numeric_tokens(line)
            return (bool(pattern.search(line.translate(_DASH)))
                    and all(part in tokens for part in parts))
    context_re = context_pattern(context) if context else None
    hits = []
    for index, line in enumerate(lines):
        line = normalize_text(line)
        if matches(line) and (context_re is None or context_re.search(line)):
            hits.append(index)
    return hits


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class ValueManifest:
    """Ordered printed-value records for one producer.

    ``sources`` maps role names to files; their sha256 is computed when the
    manifest is written. A record's ``source`` must be one of those roles.
    Re-recording a key with identical fields is a no-op; a conflicting record
    raises ``ValueError``.
    """

    def __init__(self, producer, *, sources=None):
        if not producer:
            raise ValueError("producer is required")
        self.producer = str(producer)
        self.sources = {str(role): os.fspath(path) for role, path in (sources or {}).items()}
        self._records = {}

    def record(self, key, value, text, *, source=None, context=None, section=None, page=None):
        if not isinstance(key, str) or not key:
            raise ValueError("key must be a non-empty string")
        if isinstance(value, bool):
            raise ValueError("value must be a number or None")
        if value is not None:
            value = int(value) if isinstance(value, numbers.Integral) else float(value)
            if isinstance(value, float) and not math.isfinite(value):
                value = None  # printed as a placeholder such as "---"
        text = str(text)
        text_pattern(text)  # rejects empty text
        if context is not None:
            context = str(context)
            context_pattern(context)
        if source is not None and source not in self.sources:
            raise ValueError(f"unknown source role {source!r}; register it in sources")
        if page is not None and (not isinstance(page, int) or page < 1):
            raise ValueError("page must be a positive integer")
        entry = {"key": key, "value": value, "text": text, "source": source,
                 "context": context, "section": section, "page": page}
        previous = self._records.get(key)
        if previous is not None and previous != entry:
            raise ValueError(f"printed value {key!r} recorded twice with different fields")
        self._records[key] = entry
        return text

    @property
    def records(self):
        return list(self._records.values())

    def to_dict(self):
        return {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "producer": self.producer,
            "match_rule": MATCH_RULE,
            "sources": {role: {"path": path, "sha256": _sha256(path)}
                        for role, path in self.sources.items()},
            "values": self.records,
        }

    def write(self, path):
        path = os.fspath(path)
        record = self.to_dict()
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(tmp, path)
        return path


def active_manifest():
    """The manifest recording in this context, or ``None``."""
    return _active.get()


@contextmanager
def value_manifest(path, producer, *, sources=None):
    """Activate a :class:`ValueManifest`; write it to ``path`` if the block succeeds."""
    manifest = ValueManifest(producer, sources=sources)
    token = _active.set(manifest)
    try:
        yield manifest
    finally:
        _active.reset(token)
    manifest.write(path)


def record_value(key, value, text, **fields):
    """Record already formatted ``text`` in the active manifest; return ``text``."""
    manifest = _active.get()
    if manifest is not None:
        manifest.record(key, value, text, **fields)
    return text


def printed(key, value, fmt, **fields):
    """Format ``value`` with ``format(value, fmt)``, record it if active, return the text."""
    return record_value(key, value, format(value, fmt), **fields)
