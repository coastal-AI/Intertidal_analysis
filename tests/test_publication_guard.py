# -*- coding: utf-8 -*-
"""Publication guard: nothing private goes into this repository.

It checks every file a commit would take (the files git lists as tracked, or as untracked and not ignored; every file
under the root when this is not a git checkout):

1. no forbidden token: machine, account and private file names, compared by their SHA-256, so this file publishes
   none of the words it guards against. Text files are read whole; binary files (netCDF, GeoTIFF, PNG, npz, fonts)
   through their printable strings, which is where their metadata lives;
2. no forbidden word part: the same idea on the letter runs of each word, split at digits, punctuation and camelCase,
   for a few names and terms that must not appear in any form, in the text of every text file and in every file path.
   Long base64 runs (embedded images, package hashes) are left out of this check, and so are the minified bundles of
   the website (docs/assets/), whose generated names are random letters;
3. no personal folder, mounted disk, person-named temporary folder, credential or private key, in the text of every
   text file (website bundles included) and in the printable strings of every binary file;
4. no file over 50 MB.

Run it from the repository root before every commit:

    python -X utf8 -m pytest -q -p no:cacheprovider tests/test_publication_guard.py
"""
import functools  # read each file once for all the checks
import hashlib  # SHA-256 of each token
import os  # walking the repository
import re  # tokens and generic patterns
import subprocess  # the list of files a commit would take

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the repository root
SELF = "tests/test_publication_guard.py"  # this file spells the generic patterns themselves

#: SHA-256 of lower-cased forbidden tokens (machine and account names, a network address, private file names).
FORBIDDEN_SHA256 = {
    "a8c6131bf7a207338edb3ced65e8d7b204ab985358d07125f1c840fd61de540d",
    "df62a5cf66486cb8b7e374ee7e7531c800de2f73ad3b5506aa9aca6750801d5a",
    "ddb648b4d14d6973a521e022d7db119611f34e27a845faaa49821e58c255821c",
    "947defbdd230418a099a19dc4223f941d1b63a009bf1a3dc809a7e106cda528d",
    "2d0d75dcdd4549123defb69055609b6c16702acf3d902ac121b204ec452d0649",
    "e18ea11600b63f481c54f081284dd94ec4c39ea320a8750cb2c4fc906ea1c642",
    "0e7d538f9fea38f7fa1a0c69a3399f0dee31b095d81a767572d7b35fedacd178",
    "ceca322dce6a76937c5eaa4ab9dcdf1e98b4cc844a6c2c4f05a4cde1e5464db3",
    "aa7b8007d6e360cfc0e6e0eee0c4ddf05987fc608acacc61851a39f7b50e9db3",
    "d224661143df657521f8b422d0c3a0e8d37596c1bd68d4d1a67f16a2dd030bd1",
    "48cde36d7bb2c6a33e9ef7768bd361f892bc67012fe26aeb431dd76107b2a64b",
    "85cbbb8261617b4da05ef4b1030ea122cdfb13f20aba881129a4cb889d01c82a",
    "21bde7e7a9e7b8ae791915cea18415ed0578b722c6b04953faf1b23d3df4058f",
    "5d9af8f465c053146e589f7dfda73557c9565d76365a5985174b185cbead77cb",
}

#: SHA-256 of lower-cased forbidden word parts (letters only): two survey terms, a machine, an account and a local
#: folder name, the development branch's name, a ledger's name, and assistant attribution.
FORBIDDEN_PARTS_SHA256 = {
    "3dc0deea30053c57bcbb7c1ff189b8cd5668eea42faf9b1aff650d1a4a7eb0de",
    "c97147d5c731672d8a583714b1886d79e983b4d438e83b0fa737860afb6a48f1",
    "df62a5cf66486cb8b7e374ee7e7531c800de2f73ad3b5506aa9aca6750801d5a",
    "947defbdd230418a099a19dc4223f941d1b63a009bf1a3dc809a7e106cda528d",
    "d8621f989f75156e5337c458497eed81b8cbbed3a167ac8a4ea452b403d66437",
    "59767eb5ab466157bd11baa4214c32ada9981e7db58434ae95d92a8ce98dcd4a",
    "9509d868376ef50fc9989f7ace5ee7a2cd150eeeb88f391754b62f46263abb8e",
    "c857d09db23e6822e3600bc06ad8d58f92ed62bc8efd81c753f77048662cb97d",
    "c70eca6b0f88f44d81a41311647e50fda1ac454ec04ffd442b0eb4743a993131",
}

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
    bad = {}
    for rel in repo_files():
        hits = forbidden_tokens(rel) | forbidden_tokens(strings_of(rel))
        if hits:
            bad[rel] = sorted(hits)
    assert not bad, f"private names in: {bad}"


def test_no_forbidden_word_parts():
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
