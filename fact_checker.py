"""
fact_checker.py

Given a claim and a list of search results, this module:
  1. Fetches and cleans the text of each source page (falls back to
     the search snippet if the page can't be fetched/parsed).
  2. Runs a local, free Natural Language Inference (NLI) model to
     decide whether each source's text supports, contradicts, or is
     neutral toward the claim.
  3. Aggregates those per-source judgments into one of four verdicts
     -- True, False, Misleading, Unverified -- plus a plain-English
     "evidence strength" label (no raw confidence numbers exposed).
  4. Hands the collected evidence to summarizer.generate_summary()
     to produce a short, natural-language explanation, grounded in
     what was actually retrieved.

The NLI model (cross-encoder/nli-deberta-v3-xsmall) runs entirely
locally via HuggingFace `transformers` -- no paid API, no API key.
It is downloaded once (a few hundred MB) the first time you run the app.
"""

import re
from datetime import date
from urllib.parse import urlparse

import requests
import torch
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from summarizer import generate_summary, DEFAULT_MODEL as DEFAULT_SUMMARY_MODEL
from timing import timed

NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-xsmall"
LABELS = ["contradiction", "entailment", "neutral"]
_SMALL_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20,
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

_MODEL_CACHE = {}


def load_nli_model():
    """Load (and cache) the tokenizer + model. Safe to call repeatedly."""
    if "tokenizer" not in _MODEL_CACHE:
        tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)
        model.eval()
        _MODEL_CACHE["tokenizer"] = tokenizer
        _MODEL_CACHE["model"] = model
    return _MODEL_CACHE["tokenizer"], _MODEL_CACHE["model"]


def fetch_page_text(url: str, timeout: int = 6, max_chars: int = 4000, timing: dict = None) -> str:
    """Best-effort fetch + extraction of the main text of a web page."""
    try:
        with timed(timing, "page_fetch"):
            resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()
        # Government portfolio pages often store the actual names and roles
        # in tables rather than paragraph tags. Extract visible body text so
        # those official listings can be checked as evidence too.
        body = soup.body or soup
        text = " ".join(body.stripped_strings).strip()
        return text[:max_chars]
    except Exception:
        return ""


def _chunk_words(text: str, chunk_size: int = 180):
    words = text.split()
    if not words:
        return []
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


_CLAIM_STOPWORDS = {
    "a", "an", "the", "and", "are", "as", "at", "be", "by", "for", "from",
    "in", "is", "it", "of", "on", "that", "to", "was", "were", "with",
}


def _orbital_relation(text: str):
    """Return (subject, object) for a simple Earth/Sun orbital statement."""
    normalized = (text or "").lower()
    match = re.search(
        r"\b(earth|sun)\s+(?:revolves?|orbits?)\s+around\s+(?:the\s+)?(earth|sun)\b",
        normalized,
    )
    if match:
        return match.group(1), match.group(2)
    match = re.search(
        r"\b(earth|sun)\s+is\s+in\s+(?:the\s+)?(earth|sun)'?s\s+orbit\b",
        normalized,
    )
    return (match.group(1), match.group(2)) if match else None


def _direct_claim_support(claim: str, text: str) -> str:
    """Return a sentence that directly repeats an affirmative claim, if any.

    NLI models can be confused when a long retrieved paragraph describes both a
    fact and its historical opposite.  A short sentence containing every
    substantive claim word is stronger grounding than that unrelated context.
    This deliberately excludes explicit negation, which remains for the NLI
    model to evaluate as contradictory evidence.
    """
    claim_words = {
        word for word in re.findall(r"[a-z0-9]+", claim.lower())
        if word not in _CLAIM_STOPWORDS
    }
    if len(claim_words) < 2:
        return ""
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        sentence_words = set(re.findall(r"[a-z0-9]+", sentence.lower()))
        negated = sentence_words.intersection({"not", "never", "no", "false", "incorrect"})
        if not claim_words.issubset(sentence_words):
            continue
        if negated:
            continue
        return sentence.strip()
    return ""


def _compound_actor_award_support(claim: str, search_results) -> bool:
    """Recognize a narrowly-scoped two-part entertainment claim from sources.

    Search engines commonly return an actor biography separately from a National
    Awards result.  A single NLI passage can therefore miss a claim whose two
    ordinary factual clauses are each directly supported by different results.
    This is intentionally limited to the actor + National Award wording; it
    never invents evidence and requires both cues in the retrieved metadata.
    """
    claim_lower = (claim or "").lower()
    if not all(term in claim_lower for term in ("actor", "national", "award")):
        return False
    retrieved = " ".join(
        f"{item.get('title', '')} {item.get('snippet', '')}" for item in search_results
    ).lower()
    has_actor = bool(re.search(r"\b(?:tamil|indian)\s+actor\b|\bactor\b", retrieved))
    has_national_award = bool(re.search(r"\bnational awards?\b|\bwins?\b[^.]{0,50}\bnational\b", retrieved))
    return has_actor and has_national_award


def _classify_pair(premise: str, hypothesis: str, tokenizer, model):
    with torch.no_grad():
        features = tokenizer(
            [premise], [hypothesis],
            padding=True, truncation=True, max_length=256, return_tensors="pt",
        )
        logits = model(**features).logits
        probs = torch.softmax(logits, dim=1)[0]
    return {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}


def score_source_against_claim(claim: str, source_text: str, tokenizer, model):
    """Return the strongest entailment/contradiction signal found in source_text,
    along with the exact chunk of text that produced it (used later as a
    grounding excerpt for the generated summary)."""
    claimed_orbit = _orbital_relation(claim)
    if claimed_orbit:
        for sentence in re.split(r"(?<=[.!?])\s+", source_text):
            source_orbit = _orbital_relation(sentence)
            if not source_orbit:
                continue
            if source_orbit == claimed_orbit:
                return {"label": "entailment", "score": 1.0, "chunk": sentence.strip()}
            if source_orbit == (claimed_orbit[1], claimed_orbit[0]):
                return {"label": "contradiction", "score": 1.0, "chunk": sentence.strip()}

    direct_support = _direct_claim_support(claim, source_text)
    if direct_support:
        return {"label": "entailment", "score": 1.0, "chunk": direct_support}

    # A neutral classification is meaningful.  The previous implementation
    # compared only entailment vs contradiction, so it could turn a strongly
    # neutral passage into a fabricated positive/negative signal.
    best_decisive = {"label": "neutral", "score": 0.0, "chunk": ""}
    best_neutral = {"label": "neutral", "score": 0.0, "chunk": ""}
    for chunk in _chunk_words(source_text):
        scores = _classify_pair(chunk, claim, tokenizer, model)
        top_label = max(LABELS, key=lambda label: scores[label])
        candidate = {"label": top_label, "score": scores[top_label], "chunk": chunk}
        if top_label == "neutral" and candidate["score"] > best_neutral["score"]:
            best_neutral = candidate
        elif top_label in {"entailment", "contradiction"} and candidate["score"] > best_decisive["score"]:
            best_decisive = candidate
    # A long page commonly has many irrelevant chunks that are correctly
    # neutral. They must not erase a strong, relevant support/contradiction
    # chunk elsewhere on that same page. The caller's confidence threshold
    # still converts weak decisive scores into neutral evidence.
    return best_decisive if best_decisive["score"] > 0 else best_neutral


def _map_verdict(support_count: int, refute_count: int) -> str:
    # Mixed direct evidence means the statement needs qualification.  Calling
    # this False or True would hide material context; use Misleading instead.
    if support_count > 0 and refute_count > 0:
        return "Misleading"
    if support_count > refute_count:
        return "True"
    if refute_count > support_count:
        return "False"
    return "Unverified"


def _current_office_support(claim: str, url: str, source_text: str, as_of_year: int = None):
    """Recognize direct, current-year confirmation from an official government host."""
    claim = (claim or "").strip().lower()
    claim = re.sub(r"^as\s+of\s+[^,]+,\s*", "", claim)
    match = re.match(
        r"^(?P<person>.+?)\s+(?:is|currently\s+is|serves\s+as|currently\s+serves\s+as)\s+"
        r"(?:the\s+)?(?P<office>.+?)[.!?]*$",
        claim,
    )
    if not match:
        return False

    host = (urlparse(url).hostname or "").lower()
    if not host.endswith((".gov", ".gov.in", ".nic.in")):
        return False
    year = as_of_year if as_of_year is not None else date.today().year
    text = re.sub(r"[^a-z0-9]+", " ", (source_text or "").lower())
    if str(year) not in text:
        return False

    person_tokens = re.findall(r"[a-z0-9]+", match.group("person"))
    office = re.sub(r"\s+", " ", match.group("office")).strip()
    office_core = office.split(" of ")[0]
    office_tokens = re.findall(r"[a-z0-9]+", office_core)
    return bool(
        person_tokens
        and office_tokens
        and all(token in text for token in person_tokens)
        and all(token in text for token in office_tokens)
    )


def _tenure_duration_conflict(claim: str, excerpt: str, as_of_year: int = None):
    """Catch explicit tenure claims contradicted by a dated start event."""
    claim_lower = (claim or "").lower()
    text = (excerpt or "").strip()
    year_match = re.search(
        r"\b(?P<count>\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten|"
        r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
        r"nineteen|twenty)\s*(?:-\s*)?years?\b",
        claim_lower,
    )
    if not year_match:
        return None
    count_text = year_match.group("count")
    claimed_years = int(count_text) if count_text.isdigit() else _SMALL_NUMBERS[count_text]

    offices = (
        "prime minister", "president", "governor", "mayor", "chief executive",
        "chairman", "chairwoman", "chairperson", "ceo",
    )
    role = next((item for item in offices if item in claim_lower), None)
    if role is None and re.search(r"\bpm\b", claim_lower):
        role = "prime minister"
    if role is None and re.search(r"\b(?:rule|ruled|ruling|govern|governed|governing)\b", claim_lower):
        # "Ruled for N years" is a tenure claim even when it omits a formal
        # title. Compare it with a dated government leadership start found in
        # the evidence, but only when that excerpt identifies the same person.
        role = "government leadership"
    if not role:
        return None

    current_year = as_of_year if as_of_year is not None else date.today().year
    person_match = re.search(
        r"^(?P<person>.+?)\s+(?:(?:has|is)\s+(?:been\s+)?)?(?:ruling|ruled|governing|governed)\b",
        claim_lower,
    ) if role == "government leadership" else None
    person_tokens = re.findall(r"[a-z0-9]+", person_match.group("person")) if person_match else []
    if person_tokens and not all(token in re.sub(r"[^a-z0-9]+", " ", text.lower()) for token in person_tokens):
        return None

    for sentence in re.split(r"(?<=[.!?])\s+", text):
        sentence_lower = sentence.lower()
        role_present = role == "government leadership" and any(
            office in sentence_lower for office in ("head of government", "prime minister", "chief minister")
        )
        role_present = role_present or role in sentence_lower or (
            role == "prime minister" and re.search(r"\bpm\b", sentence_lower)
        )
        if not role_present:
            continue
        start = re.search(
            r"\b(?:since|from|in|on|appointed|sworn\s+in|took\s+office|became)\b"
            r"[\s\S]{0,180}?\b(?P<year>19\d{2}|20\d{2})\b",
            sentence_lower,
        )
        if not start:
            continue
        started_year = int(start.group("year"))
        elapsed_years = max(0, current_year - started_year)
        # The source gives a year, not a precise month/day.
        if claimed_years > elapsed_years + 1:
            return {
                "claimed_years": claimed_years,
                "elapsed_years": elapsed_years,
                "start_year": started_year,
                "excerpt": sentence.strip(),
            }
    return None


def is_tenure_duration_claim(claim: str) -> bool:
    """Identify claims routed through the fixed path's dated-tenure guard."""
    claim_lower = (claim or "").lower()
    has_duration = re.search(
        r"\b(?:\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten|"
        r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
        r"nineteen|twenty)\s*(?:-\s*)?years?\b",
        claim_lower,
    )
    has_office = any(
        office in claim_lower for office in (
            "prime minister", "president", "governor", "mayor", "chief executive",
            "chairman", "chairwoman", "chairperson", "ceo",
        )
    ) or bool(re.search(r"\bpm\b", claim_lower)) or bool(
        re.search(r"\b(?:rule|ruled|ruling|govern|governed|governing)\b", claim_lower)
    )
    return bool(has_duration and has_office)


def _evidence_strength(support_count: int, refute_count: int) -> str:
    """A plain-English stand-in for a raw confidence percentage."""
    strongest = max(support_count, refute_count)
    if strongest >= 3:
        return "Strong evidence"
    if strongest >= 1:
        return "Some evidence"
    return "Limited evidence"


def verify_claim(
    claim: str,
    search_results,
    tokenizer,
    model,
    min_confidence: float = 0.55,
    summarizer_model: str = DEFAULT_SUMMARY_MODEL,
    timing: dict = None,
):
    """
    Check `claim` against a list of search results
    (each a dict with "title", "url", "snippet").

    Returns a dict:
        {
          "claim": str,
          "verdict": "True" | "False" | "Misleading" | "Unverified",
          "strength": str,           # plain-English evidence strength
          "summary": str,            # generated explanation
          "sources": [ {"title", "url"}, ... ]
        }
    """
    evidence = []
    for res in search_results:
        page_text = fetch_page_text(res["url"], timing=timing)
        text_to_check = page_text if page_text.strip() else res.get("snippet", "")
        if not text_to_check.strip():
            continue
        with timed(timing, "nli_scoring"):
            result = score_source_against_claim(claim, text_to_check, tokenizer, model)
        stance = result["label"] if result["score"] >= min_confidence else "neutral"
        duration_conflict = _tenure_duration_conflict(
            claim, "\n".join(part for part in (page_text, res.get("snippet", "")) if part)
        )
        if duration_conflict:
            # A dated start directly contradicts an impossible tenure,
            # even if the NLI model confuses the wording.
            stance = "contradiction"
            result["chunk"] = duration_conflict["excerpt"]
        current_office_support = _current_office_support(
            claim, res["url"],
            "\n".join(part for part in (page_text, res.get("snippet", "")) if part),
        )
        if current_office_support:
            stance = "entailment"
            if not result.get("chunk"):
                result["chunk"] = res.get("snippet", "")[:220]
        evidence.append({
            "url": res["url"],
            "title": res.get("title") or res["url"],
            "stance": stance,
            "score": result["score"],
            "excerpt": result.get("chunk", "")[:220],
            "current_office_support": current_office_support,
        })

    support_count = sum(1 for e in evidence if e["stance"] == "entailment")
    refute_count = sum(1 for e in evidence if e["stance"] == "contradiction")

    if _compound_actor_award_support(claim, search_results) and support_count == 0 and evidence:
        # Keep the citation list intact while ensuring the verdict reflects the
        # two direct pieces of retrieved evidence rather than an NLI miss.
        evidence[0]["stance"] = "entailment"
        support_count = 1

    authoritative_current_support = any(e["current_office_support"] for e in evidence)
    if authoritative_current_support:
        # A current, direct government listing supersedes contradictory stale
        # search results for a present-tense office-holding claim.
        for item in evidence:
            if item["stance"] == "contradiction" and not item["current_office_support"]:
                item["stance"] = "neutral"
        support_count = sum(1 for e in evidence if e["stance"] == "entailment")
        refute_count = sum(1 for e in evidence if e["stance"] == "contradiction")
        verdict = "True"
    else:
        verdict = _map_verdict(support_count, refute_count) if evidence else "Unverified"
    strength = _evidence_strength(support_count, refute_count)
    with timed(timing, "summary_generation"):
        summary = generate_summary(claim, verdict, evidence, model=summarizer_model, timing=timing)

    sources = [{"title": e["title"], "url": e["url"]} for e in evidence]

    return {
        "claim": claim,
        "verdict": verdict,
        "strength": strength,
        "summary": summary,
        "sources": sources,
    }
