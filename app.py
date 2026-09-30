"""
AI Lecture Analyzer
Turn your lectures into structured study material.

Pipeline:  PDF -> extract text -> clean -> chunk -> Hugging Face model
           -> structured analysis -> Streamlit UI (+ chat + download)

The file has two parts:
  PART 1: core logic (no UI). Every function can be imported and tested
          one by one in Google Colab:   import app
  PART 2: the Streamlit UI (only runs when started with `streamlit run app.py`).
"""

import html
import json
import os
import re
import time

import pymupdf  # PyMuPDF
import requests
import streamlit as st

# =============================================================================
# CONFIGURATION  (the only place you need to edit to change the model)
# =============================================================================

# Change the model here, or set the HF_MODEL environment variable.
# The model must be an instruction/chat model that is available through
# Hugging Face Inference Providers. Availability changes over time, so test it
# with Colab Cell 4 before relying on it.
MODEL_NAME = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")

# OpenAI-compatible chat endpoint of Hugging Face Inference Providers.
API_URL = "https://router.huggingface.co/v1/chat/completions"

REQUEST_TIMEOUT = 120        # seconds to wait for one model call
MAX_CHUNK_CHARS = 6000       # size of one chunk sent to the model
MAX_DIRECT_CHARS = 12000     # lectures shorter than this are sent whole
MAX_NOTES_CHARS = 12000      # size limit for the combined chunk notes
MIN_TEXT_CHARS = 200         # below this we assume the PDF is scanned

NOT_FOUND_MESSAGE = "I couldn't find this information in the uploaded lecture."

DEFAULT_OPTIONS = {
    "summary": True,
    "key_points": True,
    "concepts": True,
    "definitions": True,
    "exam_focus": True,
    "questions": True,
    "mcqs": True,
    "quick_revision": True,
}

PROGRESS_STAGES = [
    "PDF uploaded",
    "Text extracted",
    "Text cleaned",
    "Content analyzed",
    "Generating structured notes",
    "Finalizing revision material",
]

# =============================================================================
# PART 1A: HUGGING FACE API
# =============================================================================


class HFError(Exception):
    """Friendly error from the Hugging Face API. fatal=True means retrying
    or continuing is pointless (bad token, missing token, unknown model)."""

    def __init__(self, message, fatal=False):
        super().__init__(message)
        self.fatal = fatal


def get_hf_token():
    """Read the token from the HF_TOKEN environment variable, then from
    Streamlit secrets. The token is never printed or stored in code."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token.strip()
    try:
        return str(st.secrets["HF_TOKEN"]).strip()
    except Exception:
        return None


def call_hf(messages, max_tokens=1500, temperature=0.2, model=None, retries=2):
    """Send a chat request to Hugging Face and return the reply text.
    Raises HFError with a user friendly message on any failure."""
    token = get_hf_token()
    if not token:
        raise HFError("Hugging Face API key is missing. Please configure HF_TOKEN.", fatal=True)

    model = model or MODEL_NAME
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    headers = {"Authorization": f"Bearer {token}"}
    last_error = HFError("The model did not respond. Please try again.")

    for attempt in range(retries + 1):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
        except requests.exceptions.Timeout:
            last_error = HFError("The Hugging Face request timed out. Try again, or use a smaller PDF.")
            time.sleep(2)
            continue
        except requests.exceptions.RequestException:
            last_error = HFError("Could not reach Hugging Face. Check your internet connection.")
            time.sleep(2)
            continue

        code = resp.status_code
        if code == 200:
            try:
                text = resp.json()["choices"][0]["message"]["content"]
            except (ValueError, KeyError, IndexError, TypeError):
                raise HFError("Hugging Face returned a response in an unexpected format.")
            if not text or not text.strip():
                last_error = HFError("The model returned an empty response.")
                continue
            return text.strip()
        if code in (401, 403):
            raise HFError("Hugging Face rejected the API key. Check that HF_TOKEN is valid "
                          "and has permission to call Inference Providers.", fatal=True)
        if code == 404:
            raise HFError(f"The model '{model}' was not found or is not available through the API. "
                          "Change MODEL_NAME in app.py.", fatal=True)
        if code == 400:
            raise HFError(f"The model '{model}' rejected the request. It may not support chat "
                          f"requests. Details: {resp.text[:200]}", fatal=True)
        if code == 429:
            last_error = HFError("Rate limit reached. Wait a minute and try again.")
            time.sleep(5 * (attempt + 1))
            continue
        if code in (502, 503, 504):
            last_error = HFError("The model is loading or temporarily unavailable. Try again shortly.")
            time.sleep(8)
            continue
        raise HFError(f"Hugging Face returned an unexpected error (HTTP {code}).")

    raise last_error


# =============================================================================
# PART 1B: PDF EXTRACTION, CLEANING, CHUNKING
# =============================================================================


def clean_text(text):
    """Remove noise but keep the content: odd characters, extra spaces,
    broken hyphenation, and long runs of blank lines."""
    text = text.replace("\x00", "").replace("\xa0", " ")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)          # "classifi-\ncation" -> "classification"
    text = "\n".join(line.strip() for line in text.splitlines())
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf(file_bytes):
    """Return (pages, total_page_count). Empty pages are skipped, and every
    kept page remembers its real page number: [{"page": 1, "text": "..."}]"""
    doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    try:
        if doc.needs_pass:
            raise ValueError("This PDF is password protected. Remove the password and upload it again.")
        pages = []
        for number, page in enumerate(doc, start=1):
            text = clean_text(page.get_text("text"))
            if text:
                pages.append({"page": number, "text": text})
        return pages, doc.page_count
    finally:
        doc.close()


def load_pdf(file_bytes, name):
    """Extract a PDF and return a small dict describing the document."""
    pages, total = extract_pdf(file_bytes)
    chars = sum(len(p["text"]) for p in pages)
    return {
        "name": name,
        "num_pages": total,
        "pages": pages,
        "chars": chars,
        "scanned": chars < MIN_TEXT_CHARS,
    }


def _split_long(text, limit):
    """Split text longer than `limit` at paragraph/line boundaries."""
    if len(text) <= limit:
        return [text]
    pieces, current = [], ""
    for part in re.split(r"(?<=\n)", text):          # keep line breaks
        while len(part) > limit:                     # one huge line: hard cut
            if current:
                pieces.append(current)
                current = ""
            pieces.append(part[:limit])
            part = part[limit:]
        if len(current) + len(part) > limit and current:
            pieces.append(current)
            current = ""
        current += part
    if current.strip():
        pieces.append(current)
    return [p.strip() for p in pieces if p.strip()]


def pages_to_text(pages):
    """Join pages into one string with [Page N] markers."""
    return "\n\n".join(f"[Page {p['page']}]\n{p['text']}" for p in pages)


def chunk_pages(pages, max_chars=MAX_CHUNK_CHARS):
    """Group consecutive pages into chunks of about `max_chars` characters.
    Each chunk keeps its page numbers, so results can point back to pages."""
    chunks, current, current_len = [], [], 0

    def flush():
        nonlocal current, current_len
        if current:
            numbers = sorted({n for n, _ in current})
            chunks.append({
                "text": "".join(block for _, block in current).strip(),
                "pages": numbers,
                "start_page": numbers[0],
                "end_page": numbers[-1],
            })
        current, current_len = [], 0

    for p in pages:
        for piece in _split_long(p["text"], max_chars - 50):
            block = f"[Page {p['page']}]\n{piece}\n\n"
            if current and current_len + len(block) > max_chars:
                flush()
            current.append((p["page"], block))
            current_len += len(block)
    flush()
    return chunks


# =============================================================================
# PART 1C: SAFE JSON HANDLING + NORMALIZING MODEL OUTPUT
# =============================================================================


def parse_json_safe(raw):
    """Turn model output into a dict. Returns None instead of raising."""
    if not raw:
        return None
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return None
    candidate = raw[start:end + 1]
    attempts = [candidate, re.sub(r",\s*([}\]])", r"\1", candidate)]   # 2nd: remove trailing commas
    for text in attempts:
        try:
            data = json.loads(text, strict=False)
            return data if isinstance(data, dict) else None
        except (ValueError, TypeError):
            continue
    return None


# Which fields each kind of item has, and which are required to keep it.
FIELDS = {
    "key_points": ["text"],
    "concepts": ["name", "simple", "technical"],
    "definitions": ["term", "definition", "type"],
    "examples": ["topic", "example", "explanation"],
    "formulas": ["formula", "meaning"],
    "exam_focus": ["topic", "reason"],
}
REQUIRED = {
    "key_points": ["text"],
    "concepts": ["name"],
    "definitions": ["term", "definition"],
    "examples": ["example"],
    "formulas": ["formula"],
    "exam_focus": ["topic"],
}


def valid_page(value, allowed):
    """Return the page number only if it really exists; never keep a made-up one."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number in allowed else None


def normalize_items(kind, items, allowed_pages):
    """Clean a list from the model: drop broken entries, force strings,
    and replace invalid page numbers with None."""
    result = []
    if not isinstance(items, list):
        return result
    allowed = set(allowed_pages)
    for item in items:
        if isinstance(item, str):
            item = {REQUIRED[kind][0]: item}
        if not isinstance(item, dict):
            continue
        clean = {f: str(item.get(f) or "").strip() for f in FIELDS[kind]}
        if not all(clean[f] for f in REQUIRED[kind]):
            continue
        if kind == "definitions" and clean["type"].lower() != "definition":
            clean["type"] = "explanation"          # be conservative: only label direct definitions
        clean["page"] = valid_page(item.get("page"), allowed)
        result.append(clean)
    return result


def merge_items(kind, groups):
    """Combine lists from several chunks and remove duplicates."""
    seen, merged = set(), []
    key_field = REQUIRED[kind][0]
    for group in groups:
        for item in group:
            key = re.sub(r"\W+", " ", item[key_field].lower()).strip()[:80]
            if key and key not in seen:
                seen.add(key)
                merged.append(item)
    return merged


def string_list(value):
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if str(v).strip()]


# =============================================================================
# PART 1D: PROMPTS
# =============================================================================

SYSTEM_PROMPT = (
    "You are a careful study assistant. Use ONLY the lecture text you are given. "
    "Never invent content, definitions, formulas, examples, page numbers, or facts from outside "
    "the lecture. If something is not in the text, leave it out (use null or an empty list). "
    "Page numbers must come from the [Page N] markers in the text. "
    "When asked for JSON, reply with ONLY one valid JSON object: no markdown, no commentary."
)

CHUNK_PROMPT = """Read this part of a lecture (pages <<START>> to <<END>>) and extract study information.

Return ONLY a JSON object with exactly this shape:
{
  "key_points": [{"text": "one important point", "page": 3}],
  "concepts": [{"name": "", "simple": "beginner-friendly explanation", "technical": "more precise explanation, or empty", "page": 3}],
  "definitions": [{"term": "", "definition": "", "type": "definition or explanation", "page": 3}],
  "examples": [{"topic": "", "example": "example exactly as given in the lecture", "explanation": "", "page": 3}],
  "formulas": [{"formula": "copied as accurately as possible", "meaning": "", "page": 3}]
}

Rules:
- Only include things that appear in the text. Empty list if none.
- "type" is "definition" only if the lecture states it directly; otherwise "explanation".
- Do not create new examples. Do not invent formulas.
- "page" must be a number from a [Page N] marker.

LECTURE TEXT:
<<TEXT>>"""

OVERVIEW_PROMPT = """Below is material from a lecture. Produce an overview.

Return ONLY a JSON object:
{
  "title": "lecture title, or null if not clearly stated",
  "subject": "school subject / field, or null if unsure",
  "topic": "specific topic of this lecture, or null if unsure",
  "main_theme": "one sentence, or null if unsure",
  "summary": "a concise, useful summary in simple language covering the major ideas, keeping important technical information, without repetition and without outside information",
  "key_points": [{"text": "", "page": 3}]
}

Rules:
- Do not guess the title or subject. Use null if they are not supported by the material.
- key_points: the 8 to 12 most important points. Reuse page numbers from the material only.

FIRST PAGE OF THE DOCUMENT (for title detection):
<<FIRST>>

MATERIAL:
<<MATERIAL>>"""

EXAM_PROMPT = """From the lecture material below, list topics that look important for revision.
Base your choice only on signals visible in the material: repetition, headings, definitions,
examples, explicit emphasis, fundamental concepts.

Return ONLY a JSON object:
{"exam_focus": [{"topic": "", "reason": "which signal made it stand out", "page": 3}]}

Rules:
- Give 5 to 10 topics.
- Never claim something will appear in an exam. Describe it as potentially important for revision.
- Page numbers only from the material.

MATERIAL:
<<MATERIAL>>"""

QUESTIONS_PROMPT = """Write possible exam questions based ONLY on the lecture material below.
Do not ask about anything that is not covered in the material.

Return ONLY a JSON object:
{"short": ["3 to 5 short questions"], "conceptual": ["2 to 4 conceptual questions"], "long": ["1 to 3 long questions"]}

MATERIAL:
<<MATERIAL>>"""

MCQ_PROMPT = """Write <<N>> multiple choice questions based ONLY on the lecture material below.
Every question and its correct answer must be supported by the material.

Return ONLY a JSON object:
{"mcqs": [{"question": "", "options": ["option A text", "option B text", "option C text", "option D text"], "answer": "A", "explanation": "why it is correct, based on the lecture", "page": 3}]}

Rules:
- Exactly 4 options per question, without the letters in the text.
- "answer" is one letter: A, B, C or D.
- Wrong options must be plausible but clearly wrong according to the lecture.
- Page numbers only from the material.

MATERIAL:
<<MATERIAL>>"""

REVISION_PROMPT = """Create a quick revision sheet from the lecture material below. It must be readable in under 2 minutes.

Return ONLY a JSON object:
{"quick_revision": [{"term": "", "gist": "very short explanation (under 15 words)"}]}

Rules: 10 to 15 entries, only from the material.

MATERIAL:
<<MATERIAL>>"""

CHAT_MODES = {
    "Lecture Mode": "Answer strictly using the lecture text. Quote or closely follow its wording.",
    "Simple Explanation": "Explain the idea from the lecture in beginner-friendly language, "
                          "using short sentences. Do not add facts that are not in the lecture.",
    "Exam Mode": "Answer the way a good exam answer would be written, using only lecture content. "
                 "Mention key terms. You may point out what looks important for revision.",
    "Quick Revision": "Answer in one to three short lines suitable for last-minute revision.",
}

CHAT_SYSTEM = (
    "You answer questions about ONE uploaded lecture. Use only the lecture excerpts below. "
    "If the answer cannot be found or reasonably derived from them, reply exactly: "
    f"{NOT_FOUND_MESSAGE} "
    "Never invent facts or page numbers. When possible, mention the page like (Page 4), using "
    "only the [Page N] markers. Do not use outside knowledge to answer.\n\n"
    "STYLE: <<MODE>>\n\nLECTURE EXCERPTS:\n<<CONTEXT>>"
)


def fill(template, **values):
    """Fill <<PLACEHOLDERS>> in a prompt. (Plain replace, so JSON braces are safe.)"""
    for key, value in values.items():
        template = template.replace(f"<<{key.upper()}>>", str(value))
    return template


# =============================================================================
# PART 1E: ANALYSIS PIPELINE
# =============================================================================


def ask_json(prompt, max_tokens=2000):
    """Call the model and parse JSON. If the first answer is not valid JSON,
    ask once more. Returns a dict or None (never raises for bad JSON)."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    raw = call_hf(messages, max_tokens=max_tokens)
    data = parse_json_safe(raw)
    if data is None:
        messages += [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": "That was not valid JSON. Reply again with ONLY the corrected JSON object."},
        ]
        data = parse_json_safe(call_hf(messages, max_tokens=max_tokens))
    return data


def analyze_chunk(chunk):
    """Extract key points, concepts, definitions, examples, formulas from one chunk."""
    prompt = fill(CHUNK_PROMPT, start=chunk["start_page"], end=chunk["end_page"], text=chunk["text"])
    data = ask_json(prompt, max_tokens=2000)
    if data is None:
        raise HFError("The model returned invalid output for one section.")
    return {kind: normalize_items(kind, data.get(kind), chunk["pages"])
            for kind in ("key_points", "concepts", "definitions", "examples", "formulas")}


def merge_chunk_results(results):
    """Combine chunk-level results into one set of lists (duplicates removed)."""
    return {kind: merge_items(kind, [r.get(kind, []) for r in results])
            for kind in ("key_points", "concepts", "definitions", "examples", "formulas")}


def build_notes(merged, max_chars=MAX_NOTES_CHARS):
    """Turn merged chunk results into compact text notes for the final steps."""
    def tag(item):
        return f"(Page {item['page']}) " if item["page"] else ""

    lines = ["KEY POINTS:"]
    lines += [f"- {tag(i)}{i['text']}" for i in merged["key_points"]]
    lines.append("CONCEPTS:")
    lines += [f"- {tag(i)}{i['name']}: {i['simple']}" for i in merged["concepts"]]
    lines.append("DEFINITIONS:")
    lines += [f"- {tag(i)}{i['term']}: {i['definition']}" for i in merged["definitions"]]
    lines.append("EXAMPLES:")
    lines += [f"- {tag(i)}{i['topic']}: {i['example']}" for i in merged["examples"]]
    lines.append("FORMULAS:")
    lines += [f"- {tag(i)}{i['formula']} {i['meaning']}" for i in merged["formulas"]]
    return "\n".join(lines)[:max_chars]


def get_material(pages, merged=None):
    """What the final steps read: the full lecture if short, else the chunk notes."""
    full = pages_to_text(pages)
    if len(full) <= MAX_DIRECT_CHARS or not merged:
        return full[:MAX_DIRECT_CHARS]
    return build_notes(merged)


def gen_overview(pages, material):
    data = ask_json(fill(OVERVIEW_PROMPT, first=pages[0]["text"][:800], material=material), max_tokens=2500)
    if data is None:
        raise HFError("The model returned invalid output for the overview.")
    allowed = [p["page"] for p in pages]

    def text_or_empty(key):
        value = data.get(key)
        return str(value).strip() if value and str(value).strip().lower() != "null" else ""

    return {
        "title": text_or_empty("title"),
        "subject": text_or_empty("subject"),
        "topic": text_or_empty("topic"),
        "main_theme": text_or_empty("main_theme"),
        "summary": text_or_empty("summary"),
        "key_points": normalize_items("key_points", data.get("key_points"), allowed),
    }


def gen_exam_focus(pages, material):
    data = ask_json(fill(EXAM_PROMPT, material=material))
    if data is None:
        raise HFError("The model returned invalid output for exam focus.")
    return normalize_items("exam_focus", data.get("exam_focus"), [p["page"] for p in pages])


def gen_questions(material):
    data = ask_json(fill(QUESTIONS_PROMPT, material=material))
    if data is None:
        raise HFError("The model returned invalid output for questions.")
    return {k: string_list(data.get(k)) for k in ("short", "conceptual", "long")}


def gen_mcqs(pages, material, n=None):
    if n is None:                                   # 5 to 15 depending on lecture size
        n = max(5, min(15, len(material) // 1500))
    data = ask_json(fill(MCQ_PROMPT, n=n, material=material), max_tokens=3500)
    if data is None:
        raise HFError("The model returned invalid output for MCQs.")
    allowed = {p["page"] for p in pages}
    result = []
    for item in data.get("mcqs") or []:
        if not isinstance(item, dict):
            continue
        options = string_list(item.get("options"))
        answer = str(item.get("answer") or "").strip().upper()[:1]
        question = str(item.get("question") or "").strip()
        if len(options) != 4 or answer not in "ABCD" or not answer or not question:
            continue
        options = [re.sub(r"^[A-Da-d][\).:]\s+", "", o) for o in options]   # drop "A. " if model added it
        result.append({
            "question": question,
            "options": options,
            "answer": answer,
            "explanation": str(item.get("explanation") or "").strip(),
            "page": valid_page(item.get("page"), allowed),
        })
    return result


def gen_quick_revision(material):
    data = ask_json(fill(REVISION_PROMPT, material=material))
    if data is None:
        raise HFError("The model returned invalid output for quick revision.")
    result = []
    for item in data.get("quick_revision") or []:
        if isinstance(item, dict) and str(item.get("term") or "").strip() and str(item.get("gist") or "").strip():
            result.append({"term": str(item["term"]).strip(), "gist": str(item["gist"]).strip()})
    return result


def empty_analysis():
    return {
        "title": "", "subject": "", "topic": "", "main_theme": "", "summary": "",
        "key_points": [], "concepts": [], "definitions": [], "examples": [], "formulas": [],
        "exam_focus": [], "questions": {"short": [], "conceptual": [], "long": []},
        "mcqs": [], "quick_revision": [], "warnings": [],
    }


def analyze_lecture(pages, options=None, progress=None, chunk_results=None):
    """Run the whole analysis and return the structured dict.

    pages          list from extract_pdf
    options        dict of booleans (see DEFAULT_OPTIONS)
    progress       optional function progress(stage_index, message)
    chunk_results  optional results of analyze_chunk (skips re-analysis in Colab)

    A failure in one step is recorded in analysis["warnings"] and the other
    steps still run. Only problems that make every step fail (missing or invalid
    token, unknown model) are raised as HFError.
    """
    options = {**DEFAULT_OPTIONS, **(options or {})}
    report = progress or (lambda stage, message="": None)
    analysis = empty_analysis()
    warnings = analysis["warnings"]

    def step(name, function):
        try:
            return function()
        except HFError as error:
            if error.fatal:
                raise
            warnings.append(f"{name}: {error}")
        except Exception as error:  # never let one step crash the app
            warnings.append(f"{name}: unexpected error ({type(error).__name__}).")
        return None

    # --- Stage 3: analyze each chunk ---
    chunks = chunk_pages(pages)
    report(3, f"Analyzing {len(chunks)} section(s) of the lecture...")
    if chunk_results is None:
        chunk_results = []
        for index, chunk in enumerate(chunks, start=1):
            report(3, f"Analyzing section {index} of {len(chunks)} (pages {chunk['start_page']}-{chunk['end_page']})")
            result = step(f"Section {index}", lambda c=chunk: analyze_chunk(c))
            if result:
                chunk_results.append(result)
    if not chunk_results:
        raise HFError("The model could not analyze any part of this lecture. Try again or change MODEL_NAME.")

    merged = merge_chunk_results(chunk_results)
    material = get_material(pages, merged)

    # --- Stage 4: overview, concepts, definitions ---
    report(4, "Writing overview and summary...")
    overview = step("Overview", lambda: gen_overview(pages, material))
    if overview:
        for key in ("title", "subject", "topic", "main_theme"):
            analysis[key] = overview[key]
        if options["summary"]:
            analysis["summary"] = overview["summary"]
        if options["key_points"]:
            analysis["key_points"] = overview["key_points"]
    if options["key_points"] and not analysis["key_points"]:
        analysis["key_points"] = merged["key_points"][:12]     # fallback: use chunk level points
    if options["concepts"]:
        analysis["concepts"] = merged["concepts"]
    if options["definitions"]:
        analysis["definitions"] = merged["definitions"]
    analysis["examples"] = merged["examples"]
    analysis["formulas"] = merged["formulas"]

    # --- Stage 4b: exam focus, questions, MCQs ---
    if options["exam_focus"]:
        report(4, "Finding exam focus topics...")
        analysis["exam_focus"] = step("Exam focus", lambda: gen_exam_focus(pages, material)) or []
    if options["questions"]:
        report(4, "Writing possible exam questions...")
        analysis["questions"] = step("Questions", lambda: gen_questions(material)) or analysis["questions"]
    if options["mcqs"]:
        report(4, "Writing MCQs...")
        analysis["mcqs"] = step("MCQs", lambda: gen_mcqs(pages, material)) or []

    # --- Stage 5: revision sheet ---
    if options["quick_revision"]:
        report(5, "Building quick revision sheet...")
        analysis["quick_revision"] = step("Quick revision", lambda: gen_quick_revision(material)) or []

    report(6, "Done")
    return analysis


# =============================================================================
# PART 1F: CHAT WITH THE LECTURE
# =============================================================================

STOPWORDS = set(
    "a an the is are was were be been of to in on at for and or but if then this that these those "
    "it its as by with from what which who whom how why when where do does did can could should would "
    "me my you your i we they he she about explain tell give simple words please".split()
)


def select_context(pages, query, max_chars=9000):
    """Very simple retrieval: pick the pages that share the most words with the
    question. No embeddings or vector database needed."""
    full = pages_to_text(pages)
    if len(full) <= max_chars:
        return full
    words = {w for w in re.findall(r"[a-z0-9]+", query.lower()) if w not in STOPWORDS and len(w) > 2}
    scored = []
    for p in pages:
        text = p["text"].lower()
        score = sum(text.count(w) for w in words)
        if score:
            scored.append((score, p))
    scored.sort(key=lambda pair: -pair[0])
    picked, used = [], 0
    for _, p in scored:
        block = f"[Page {p['page']}]\n{p['text']}"
        if used + len(block) > max_chars:
            continue
        picked.append((p["page"], block))
        used += len(block)
    picked.sort()
    return "\n\n".join(block for _, block in picked)


def answer_question(question, history, mode, pages, analysis=None):
    """Answer a question about the lecture. `history` is a list of
    {"role": "user"/"assistant", "content": "..."}."""
    recent_user = " ".join(m["content"] for m in history[-4:] if m["role"] == "user")
    context = select_context(pages, f"{question} {recent_user}")
    if not context and analysis and analysis.get("summary"):
        context = "[Summary of the lecture]\n" + analysis["summary"]
    if not context:
        return NOT_FOUND_MESSAGE
    system = fill(CHAT_SYSTEM, mode=CHAT_MODES.get(mode, CHAT_MODES["Lecture Mode"]), context=context)
    messages = [{"role": "system", "content": system}] + history[-6:] + [{"role": "user", "content": question}]
    return call_hf(messages, max_tokens=700, temperature=0.2)


# =============================================================================
# PART 1G: DOWNLOAD (Markdown)
# =============================================================================


def build_markdown(analysis, doc_info=None):
    """Create the study notes as a Markdown string."""
    def pg(item):
        return f" _(Page {item['page']})_" if item.get("page") else ""

    a = analysis
    lines = [f"# {a['title'] or (doc_info or {}).get('name', 'Lecture Study Notes')}", ""]
    if doc_info:
        lines += [f"Source file: {doc_info['name']} ({doc_info['num_pages']} pages)", ""]
    for label, key in (("Subject", "subject"), ("Topic", "topic"), ("Main theme", "main_theme")):
        if a.get(key):
            lines.append(f"**{label}:** {a[key]}  ")
    lines += ["", "## Summary", a["summary"] or "_Not generated._", "", "## Key points"]
    lines += [f"{i}. {k['text']}{pg(k)}" for i, k in enumerate(a["key_points"], 1)] or ["_None._"]

    lines += ["", "## Important concepts"]
    for c in a["concepts"]:
        lines.append(f"- **{c['name']}**{pg(c)}: {c['simple']}")
        if c["technical"]:
            lines.append(f"  - Technical: {c['technical']}")
    if not a["concepts"]:
        lines.append("_None._")

    lines += ["", "## Definitions"]
    for d in a["definitions"]:
        kind = "" if d["type"] == "definition" else " (explanation, not a direct definition)"
        lines.append(f"- **{d['term']}**{kind}{pg(d)}: {d['definition']}")
    if not a["definitions"]:
        lines.append("_None._")

    lines += ["", "## Examples"]
    for e in a["examples"]:
        lines.append(f"- **{e['topic'] or 'Example'}**{pg(e)}: {e['example']}")
        if e["explanation"]:
            lines.append(f"  - {e['explanation']}")
    if not a["examples"]:
        lines.append("_None._")

    lines += ["", "## Formulas / equations"]
    lines += [f"- `{f['formula']}`{pg(f)} {f['meaning']}" for f in a["formulas"]] or ["No important formulas detected."]

    lines += ["", "## Exam focus (potentially important for revision)"]
    lines += [f"- **{x['topic']}**{pg(x)}: {x['reason']}" for x in a["exam_focus"]] or ["_None._"]

    lines += ["", "## Possible exam questions"]
    for label, key in (("Short questions", "short"), ("Conceptual questions", "conceptual"), ("Long questions", "long")):
        lines += ["", f"### {label}"]
        lines += [f"{i}. {q}" for i, q in enumerate(a["questions"][key], 1)] or ["_None._"]

    lines += ["", "## MCQs"]
    for i, m in enumerate(a["mcqs"], 1):
        lines += ["", f"**Q{i}. {m['question']}**"]
        lines += [f"- {'ABCD'[j]}. {opt}" for j, opt in enumerate(m["options"])]
        lines += [f"- **Answer:** {m['answer']}", f"- Explanation: {m['explanation']}{pg(m)}"]
    if not a["mcqs"]:
        lines.append("_None._")

    lines += ["", "## Quick revision"]
    lines += [f"- {q['term']} → {q['gist']}" for q in a["quick_revision"]] or ["_None._"]

    if a.get("warnings"):
        lines += ["", "---", "Some sections could not be generated:"] + [f"- {w}" for w in a["warnings"]]
    lines += ["", "---", "_Generated by AI Lecture Analyzer. Always verify against the original lecture._"]
    return "\n".join(lines)


# =============================================================================
# PART 2: STREAMLIT UI
# =============================================================================

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@500;600&family=Source+Sans+3:wght@400;500;600&display=swap');
:root { --ink:#1c2430; --muted:#5b6675; --paper:#f4f6f8; --line:#dde2e8; --teal:#0f6b63; --mark:#f7dd72; }
html, body, [class*="css"], .stMarkdown, button, input, textarea { font-family:'Source Sans 3', sans-serif; color:var(--ink); }
h1, h2, h3, .app-title { font-family:'Source Serif 4', serif !important; letter-spacing:-0.01em; }
#MainMenu, footer { visibility:hidden; }
.block-container { padding-top:2rem; max-width:1100px; }
.app-title { font-size:2.3rem; font-weight:600; margin:0; }
.app-tag { color:var(--muted); font-size:1.05rem; margin:.15rem 0 1.4rem 0; }
.card { background:#fff; border:1px solid var(--line); border-left:4px solid var(--teal); border-radius:6px; padding:.85rem 1.1rem; margin:.6rem 0; line-height:1.55; }
.card b.name { font-weight:600; }
.card .sub { color:var(--muted); font-size:.93rem; margin-top:.25rem; }
.pg { background:var(--mark); color:#3b3200; font-size:.78rem; font-weight:600; padding:1px 7px; border-radius:3px; margin-left:.5rem; white-space:nowrap; }
.stat { background:#fff; border:1px solid var(--line); border-radius:6px; padding:.7rem 1rem; }
.stat .v { font-family:'Source Serif 4', serif; font-size:1.5rem; font-weight:600; }
.stat .l { color:var(--muted); font-size:.85rem; }
.stages div { padding:.12rem 0; color:var(--muted); }
.stages .done { color:var(--teal); }
.stages .now { color:var(--ink); font-weight:600; }
.stTabs [data-baseweb="tab"] { font-weight:500; }
.stButton > button, .stDownloadButton > button { border-radius:6px; font-weight:600; }
</style>
"""


def esc(text):
    """Escape model/PDF text before putting it inside HTML."""
    return html.escape(str(text or ""))


def page_badge(page):
    return f'<span class="pg">Page {page}</span>' if page else ""


def card(inner_html):
    st.markdown(f'<div class="card">{inner_html}</div>', unsafe_allow_html=True)


def render_progress(placeholder, current, message=""):
    """Show the checklist:  ✓ done   ● in progress   ○ waiting."""
    rows = []
    for index, name in enumerate(PROGRESS_STAGES):
        if index < current:
            rows.append(f'<div class="done">✓ {name}</div>')
        elif index == current:
            rows.append(f'<div class="now">● {name}</div>')
        else:
            rows.append(f"<div>○ {name}</div>")
    extra = f'<div style="margin-top:.4rem;color:#5b6675">{esc(message)}</div>' if message else ""
    placeholder.markdown(f'<div class="stages">{"".join(rows)}{extra}</div>', unsafe_allow_html=True)


def init_state():
    defaults = {"doc": None, "file_key": None, "analysis": None, "chat": [], "mode": "Lecture Mode",
                "status": "Waiting for a PDF", "error": None}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def empty_note(text):
    st.info(text)


def show_overview(a, doc):
    cols = st.columns(3)
    for col, (value, label) in zip(cols, [(doc["num_pages"], "Pages"), (f"{doc['chars']:,}", "Extracted characters"),
                                        (st.session_state.status, "Status")]):
        col.markdown(f'<div class="stat"><div class="v">{esc(value)}</div><div class="l">{label}</div></div>',
                     unsafe_allow_html=True)
    st.markdown("### Lecture overview")
    not_stated = "Not clearly stated in the document"
    card(f"<b class='name'>Title:</b> {esc(a['title'] or not_stated)}<br>"
         f"<b class='name'>Topic:</b> {esc(a['topic'] or not_stated)}<br>"
         f"<b class='name'>Subject:</b> {esc(a['subject'] or not_stated)}<br>"
         f"<b class='name'>Main theme:</b> {esc(a['main_theme'] or not_stated)}")
    st.markdown("### Summary")
    if a["summary"]:
        st.write(a["summary"])
    else:
        empty_note("No summary was generated.")
    for warning in a["warnings"]:
        st.warning(warning)


def show_key_points(a):
    if not a["key_points"]:
        return empty_note("No key points were generated.")
    for i, k in enumerate(a["key_points"], 1):
        card(f"{i}. {esc(k['text'])}{page_badge(k['page'])}")


def show_concepts(a):
    if not a["concepts"]:
        return empty_note("No concepts were found.")
    for c in a["concepts"]:
        tech = f"<div class='sub'><b>Technical:</b> {esc(c['technical'])}</div>" if c["technical"] else ""
        card(f"<b class='name'>{esc(c['name'])}</b>{page_badge(c['page'])}"
             f"<div class='sub'><b>Simple:</b> {esc(c['simple'] or '—')}</div>{tech}")


def show_definitions(a):
    if not a["definitions"]:
        return empty_note("No definitions were found in the lecture.")
    for d in a["definitions"]:
        note = "" if d["type"] == "definition" else "<div class='sub'>Explanation from the lecture, not a direct definition.</div>"
        card(f"<b class='name'>{esc(d['term'])}</b>{page_badge(d['page'])}<br>{esc(d['definition'])}{note}")


def show_examples_formulas(a):
    st.markdown("### Examples")
    if not a["examples"]:
        empty_note("No examples were found in the lecture.")
    for e in a["examples"]:
        expl = f"<div class='sub'>{esc(e['explanation'])}</div>" if e["explanation"] else ""
        card(f"<b class='name'>{esc(e['topic'] or 'Example')}</b>{page_badge(e['page'])}<br>{esc(e['example'])}{expl}")
    st.markdown("### Formulas / equations")
    if not a["formulas"]:
        st.write("No important formulas detected.")
    for f in a["formulas"]:
        st.code(f["formula"], language="text")
        st.caption(f"{f['meaning']}  " + (f"(Page {f['page']})" if f["page"] else ""))


def show_exam_focus(a):
    st.caption("Topics that look potentially important for revision. This is a suggestion, not a prediction.")
    if not a["exam_focus"]:
        return empty_note("No exam focus topics were generated.")
    for x in a["exam_focus"]:
        card(f"<b class='name'>{esc(x['topic'])}</b>{page_badge(x['page'])}<div class='sub'>{esc(x['reason'])}</div>")


def show_questions(a):
    labels = (("Short questions", "short"), ("Conceptual questions", "conceptual"), ("Long questions", "long"))
    if not any(a["questions"][k] for _, k in labels):
        return empty_note("No questions were generated.")
    for label, key in labels:
        if a["questions"][key]:
            with st.expander(label, expanded=True):
                for i, q in enumerate(a["questions"][key], 1):
                    st.markdown(f"{i}. {q}")


def show_mcqs(a):
    if not a["mcqs"]:
        return empty_note("No MCQs were generated.")
    for i, m in enumerate(a["mcqs"], 1):
        with st.expander(f"Q{i}. {m['question']}", expanded=(i == 1)):
            for j, option in enumerate(m["options"]):
                st.markdown(f"**{'ABCD'[j]}.** {option}")
            if st.checkbox("Show answer", key=f"ans_{i}"):
                letter = m["answer"]
                st.success(f"Correct answer: {letter}. {m['options']['ABCD'.index(letter)]}")
                if m["explanation"]:
                    st.write(m["explanation"] + (f"  (Page {m['page']})" if m["page"] else ""))


def show_revision(a):
    if not a["quick_revision"]:
        return empty_note("No revision notes were generated.")
    for q in a["quick_revision"]:
        card(f"<b class='name'>{esc(q['term'])}</b> → {esc(q['gist'])}")


def show_chat(doc, a):
    st.session_state.mode = st.radio("Answer style", list(CHAT_MODES), horizontal=True,
                                     index=list(CHAT_MODES).index(st.session_state.mode))
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.write(m["content"])
    question = st.chat_input("Ask a question about this lecture")
    if question:
        with st.chat_message("user"):
            st.write(question)
        try:
            with st.spinner("Searching the lecture..."):
                reply = answer_question(question, st.session_state.chat, st.session_state.mode, doc["pages"], a)
        except HFError as error:
            reply = f"⚠️ {error}"
        with st.chat_message("assistant"):
            st.write(reply)
        st.session_state.chat += [{"role": "user", "content": question}, {"role": "assistant", "content": reply}]


def main():
    st.set_page_config(page_title="AI Lecture Analyzer", page_icon="📘", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()

    st.markdown('<p class="app-title">AI Lecture Analyzer</p>'
                '<p class="app-tag">Upload a lecture. Understand it faster.</p>', unsafe_allow_html=True)

    # ---------------- Sidebar ----------------
    with st.sidebar:
        st.header("Your lecture")
        uploaded = st.file_uploader("Upload lecture PDF", type=["pdf"])
        st.caption(f"Model: `{MODEL_NAME}`")
        st.subheader("Analysis options")
        options = {
            "summary": st.checkbox("Summary", True),
            "key_points": st.checkbox("Key points", True),
            "concepts": st.checkbox("Concepts", True),
            "definitions": st.checkbox("Definitions", True),
            "exam_focus": st.checkbox("Exam focus", True),
            "questions": st.checkbox("Questions", True),
            "mcqs": st.checkbox("MCQs", True),
            "quick_revision": st.checkbox("Quick revision", True),
        }
        analyze_clicked = st.button("Analyze lecture", type="primary", use_container_width=True,
                                    disabled=uploaded is None)
        if st.session_state.analysis and st.session_state.doc:
            st.download_button("Download study notes (.md)",
                               build_markdown(st.session_state.analysis, st.session_state.doc),
                               file_name="lecture_study_notes.md", mime="text/markdown",
                               use_container_width=True)

    # ---------------- Load the PDF (once per new file) ----------------
    if uploaded is not None:
        key = (uploaded.name, uploaded.size)
        if st.session_state.file_key != key:
            st.session_state.file_key = key
            st.session_state.analysis, st.session_state.chat, st.session_state.error = None, [], None
            try:
                st.session_state.doc = load_pdf(uploaded.getvalue(), uploaded.name)
                st.session_state.status = "PDF loaded"
            except Exception as error:
                st.session_state.doc = None
                st.session_state.error = f"Could not read this PDF: {error}"
    else:
        st.session_state.doc, st.session_state.file_key = None, None
        st.session_state.analysis, st.session_state.chat = None, []

    doc = st.session_state.doc

    if st.session_state.error:
        st.error(st.session_state.error)
    if doc is None:
        if not st.session_state.error:
            st.markdown("#### Upload your lecture PDF and turn it into structured study material.")
            st.write("Use the sidebar to upload slides or a chapter exported as PDF, then choose **Analyze lecture**.")
        return

    cols = st.columns(3)
    for col, (value, label) in zip(cols, [(esc(doc["name"]), "File"), (doc["num_pages"], "Pages"),
                                        (f"{doc['chars']:,}", "Extracted characters")]):
        col.markdown(f'<div class="stat"><div class="v" style="font-size:1.1rem;word-break:break-word">{value}</div>'
                     f'<div class="l">{label}</div></div>', unsafe_allow_html=True)

    if doc["scanned"]:
        st.error("This PDF appears to contain scanned/image based pages. "
                 "Text extraction could not find enough readable text.")
        return

    # ---------------- Run the analysis ----------------
    if analyze_clicked:
        st.session_state.chat = []
        progress_box = st.empty()
        render_progress(progress_box, 3, "Starting...")
        try:
            st.session_state.analysis = analyze_lecture(
                doc["pages"], options, progress=lambda stage, msg="": render_progress(progress_box, stage, msg))
            st.session_state.status = "Analysis complete"
            st.session_state.error = None
        except HFError as error:
            st.session_state.analysis = None
            st.session_state.status = "Analysis failed"
            st.session_state.error = str(error)
        progress_box.empty()
        st.rerun()

    analysis = st.session_state.analysis
    if analysis is None:
        st.info("Ready. Choose **Analyze lecture** in the sidebar to begin.")
        st.markdown('<div class="stages"><div class="done">✓ PDF uploaded</div><div class="done">✓ Text extracted</div>'
                    '<div class="done">✓ Text cleaned</div><div>○ Content analyzed</div></div>', unsafe_allow_html=True)
        return

    tabs = st.tabs(["Overview", "Key points", "Concepts", "Definitions", "Examples & formulas",
                    "Exam focus", "Questions", "MCQs", "Quick revision", "Chat"])
    with tabs[0]:
        show_overview(analysis, doc)
    with tabs[1]:
        show_key_points(analysis)
    with tabs[2]:
        show_concepts(analysis)
    with tabs[3]:
        show_definitions(analysis)
    with tabs[4]:
        show_examples_formulas(analysis)
    with tabs[5]:
        show_exam_focus(analysis)
    with tabs[6]:
        show_questions(analysis)
    with tabs[7]:
        show_mcqs(analysis)
    with tabs[8]:
        show_revision(analysis)
    with tabs[9]:
        show_chat(doc, analysis)


# Streamlit runs this file as __main__; importing it in Colab does not start the UI.
if __name__ == "__main__":
    main()
