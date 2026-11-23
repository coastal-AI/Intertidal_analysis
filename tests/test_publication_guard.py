# -*- coding: utf-8 -*-
"""Publication guard: nothing private goes into this repository.

It checks every file a commit would take (the files git lists as tracked, or as untracked and not ignored; every file
under the root when this is not a git checkout):

1. no forbidden token: no whole token (5 characters or more) from a private list. Text files are read whole; binary
   files (netCDF, GeoTIFF, PNG, npz, fonts) through their printable strings, which is where their metadata lives;
2. no forbidden word part: the same idea on the letter runs of each word, split at digits, punctuation and camelCase,
   for a second private list of names and terms that must not appear in any form, in the text of every text file and
   in every file path. Long base64 runs (embedded images, package hashes) are left out of this check, and so are the
   minified bundles of the website (docs/assets/), whose generated names are random letters;
3. no personal folder, mounted disk, person-named temporary folder, credential or private key, in the text of every
   text file (website bundles included) and in the printable strings of every binary file;
4. no file over 50 MB;
5. the private lists themselves are not among the files a commit would take.

The private lists are not in this repository, not even as hashes: the hash of a short word is reversed by hashing
candidate words, so a published list of hashes would publish the words. They live in a local file,
tests/publication_guard_local.json, which .gitignore keeps out of every commit:

    {"tokens": ["<sha256>", ...], "word_parts": ["<sha256>", ...]}

Each entry is the SHA-256 (hex) of a lower-cased word, for example

    python -c "import hashlib; print(hashlib.sha256('someword'.encode()).hexdigest())"

Without that file, checks 1 and 2 are skipped, with a message that says why; checks 3 to 5 always run.

Run it from the repository root before every commit:

    python -X utf8 -m pytest -q -p no:cacheprovider tests/test_publication_guard.py
"""
import functools  # read each file once for all the checks
import hashlib  # SHA-256 of each token
import json  # the local file of private lists
import os  # walking the repository
import re  # tokens and generic patterns
import subprocess  # the list of files a commit would take

import pytest  # skipping checks 1 and 2 without the local file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the repository root
SELF = "tests/test_publication_guard.py"  # this file spells the generic patterns themselves
LOCAL = "tests/publication_guard_local.json"  # the private lists of checks 1 and 2: local only, ignored by git


def _local_lists():
    """The SHA-256 sets ``(tokens, word_parts)`` of the local file, or None when the file is not there."""
    path = os.path.join(ROOT, LOCAL)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        lists = json.load(f)
    return {h.lower() for h in lists["tokens"]}, {h.lower() for h in lists["word_parts"]}


LOCAL_LISTS = _local_lists()  # None without the local file
#: SHA-256 of the lower-cased forbidden tokens, and of the lower-cased forbidden word parts (letters only)
FORBIDDEN_SHA256, FORBIDDEN_PARTS_SHA256 = LOCAL_LISTS or (set(), set())
NO_LISTS = (f"{LOCAL} is not in this checkout: this check needs the maintainers' private list, which is kept out "
            f"of the repository (see the docstring of {SELF})")

#: Generic patterns: personal folders, mounted disks, person-named temporary folders, credentials and private keys.
#: The look-behind keeps a URL path such as https://example.org/home/ from counting as a folder.
GENERIC = [
    re.compile(r"(?i)\b[a-z]:[\\/]+users[\\/]+"),  # a Windows personal folder
    re.compile(r"(?<![\w.-])/home/[A-Za-z0-9_.-]+"),  # a Linux personal folder
    re.compile(r"(?<![\w.-])/Users/[A-Za-z0-9_.-]+"),  # a macOS personal folder
    re.compile(r"(?<![\w.-])/media/[A-Za-z0-9_.-]+"),  # a mounted disk
    re.compile(r"(?<![\w.-])/mnt/[A-Za-z0-9_.-]+"),  # another mount point
    re.compile(r"(?<![\w.-])/tmp/[A-Za-z0-9.-]+_[A-Za-z0-9_.-]+"),  # a temporary folder named after a person or a job
    re.compile(r"(?i)refresh[_-]?token['\"]?\s*[:=]\s*['\"][A-Za-z0-9._-]{20,}"),  # an openEO refresh token
    re.compile(r"(?i)password['\"]?\s*[:=]\s*['\"][^'\"\s]+['\"]"),  # a password written into a file
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),  # a private key
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}"),  # a GitHub token
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),  # an AWS access key
]

TEXT = {".py", ".pyi", ".ipynb", ".md", ".rst", ".txt", ".json", ".jsonl", ".yaml", ".yml", ".cff", ".toml",
        ".cfg", ".ini", ".html", ".htm", ".js", ".mjs", ".cjs", ".jsx", ".ts", ".tsx", ".mts", ".cts", ".map",
        ".css", ".csv", ".tsv", ".geojson", ".webmanifest", ".svg", ".xml", ".sh", ".ps1", ".bat", ".r", ".tex",
        ".bib", ""}  # the extensions read as text ("" = files without one, e.g. LICENSE or .gitignore)
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".ipynb_checkpoints", "runs", ".pytest_cache", "node_modules",
             ".vite"}  # never descended into (they are ignored by git as well)
BUNDLES = "docs/assets/"  # the website's minified bundles: exempt from the word-part check only
MAX_BYTES = 50 * 1024 * 1024  # GitHub warns above 50 MB and refuses files above 100 MB

TOKEN = re.compile(r"[A-Za-z0-9_.@-]+")  # what counts as one token
WORD = re.compile(r"[A-Za-z]+")  # a run of letters
CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+")  # the parts of a camelCase run
BASE64 = re.compile(r"[A-Za-z0-9+/]{60,}={0,2}")  # embedded images, fonts and package hashes
PRINTABLE = re.compile(rb"[\x20-\x7e]{5,}")  # the strings of a binary file


def _sha(word):
    return hashlib.sha256(word.lower().encode("utf-8")).hexdigest()


def _skipped(rel):
    return any(part in SKIP_DIRS for part in rel.split("/")[:-1])


@functools.lru_cache(maxsize=1)
def repo_files():
    """Relative paths (with /) of the files a commit would take, sorted."""
    try:  # a git checkout: exactly what `git add -A` would add (tracked, or untracked and not ignored)
        env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")  # a read: never refresh the index
        top = subprocess.run(["git", "-C", ROOT, "rev-parse", "--show-toplevel"], capture_output=True, text=True,
                             timeout=60, env=env)
        if top.returncode == 0 and os.path.samefile(top.stdout.strip(), ROOT):
            out = subprocess.run(["git", "-C", ROOT, "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                                 capture_output=True, timeout=120, env=env)
            if out.returncode == 0:
                names = [p for p in out.stdout.decode("utf-8", "replace").split("\0") if p]
                return tuple(sorted(p for p in names
                                    if not _skipped(p) and os.path.isfile(os.path.join(ROOT, p))))
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    found = []  # not a git checkout (an archive, a copy): every file under the root
    for root, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        found += [os.path.relpath(os.path.join(root, name), ROOT).replace("\\", "/") for name in files]
    return tuple(sorted(found))


def is_text(rel):
    return os.path.splitext(rel)[1].lower() in TEXT


@functools.lru_cache(maxsize=None)
def strings_of(rel):
    """The text of a text file, or the printable strings of a binary one (one per line)."""
    path = os.path.join(ROOT, rel)
    if os.path.getsize(path) > MAX_BYTES:
        return ""  # reported by test_no_oversized_files
    if is_text(rel):
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    with open(path, "rb") as f:
        return "\n".join(m.decode("ascii") for m in PRINTABLE.findall(f.read()))


def forbidden_tokens(text):
    """Tokens of ``text`` whose SHA-256 is on the list (also each token without its edge punctuation)."""
    hits = set()
    for tok in TOKEN.findall(text):
        if len(tok) < 5:
            continue  # every forbidden token is longer
        for cand in {tok, tok.strip("._-@")}:
            if _sha(cand) in FORBIDDEN_SHA256:
                hits.add(cand)
    return hits


def forbidden_parts(text):
    """Word parts of ``text`` whose SHA-256 is on the list (long base64 runs left out)."""
    hits = set()
    for run in WORD.findall(BASE64.sub(" ", text)):
        for part in CAMEL.findall(run):
            if _sha(part) in FORBIDDEN_PARTS_SHA256:
                hits.add(part)
    return hits


def test_the_file_list_is_not_empty():
    files = repo_files()
    assert SELF in files, f"the guard does not see itself among {len(files)} files"


def test_no_forbidden_tokens():
    if LOCAL_LISTS is None:
        pytest.skip(NO_LISTS)
    bad = {}
    for rel in repo_files():
        hits = forbidden_tokens(rel) | forbidden_tokens(strings_of(rel))
        if hits:
            bad[rel] = sorted(hits)
    assert not bad, f"private names in: {bad}"


def test_no_forbidden_word_parts():
    if LOCAL_LISTS is None:
        pytest.skip(NO_LISTS)
    bad = {}
    for rel in repo_files():
        if rel.startswith(BUNDLES):
            continue  # minified bundles with random generated names
        hits = forbidden_parts(rel)
        if is_text(rel):
            hits |= forbidden_parts(strings_of(rel))
        if hits:
            bad[rel] = sorted(hits)
    assert not bad, f"forbidden words in: {bad}"


def test_no_personal_paths_or_credentials():
    bad = {}
    for rel in repo_files():
        if rel == SELF:
            continue  # this file spells the generic patterns themselves
        text = strings_of(rel)
        hits = sorted({m.group(0) for rx in GENERIC for m in rx.finditer(text)})
        if hits:
            bad[rel] = hits[:5]
    assert not bad, f"personal paths or credentials in: {bad}"


def test_no_oversized_files():
    big = {rel: os.path.getsize(os.path.join(ROOT, rel)) for rel in repo_files()
           if os.path.getsize(os.path.join(ROOT, rel)) > MAX_BYTES}
    assert not big, f"files over {MAX_BYTES // 2**20} MB: {big}"


def test_the_private_lists_stay_local():
    assert LOCAL not in repo_files(), f"{LOCAL} would be committed: it must stay listed in .gitignore"
