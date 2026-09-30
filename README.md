# AI Lecture Analyzer

*Turn your lectures into structured study material.*

Upload a lecture PDF (slides, a textbook chapter, notes). The app extracts the text, sends it to a Hugging Face model, and gives you a summary, key points, concepts, definitions, examples, formulas, exam focus topics, possible exam questions, MCQs, a quick revision sheet, and a chat that answers questions about your lecture. Results carry page references so you can verify them.

## Features

- PDF upload with page-by-page text extraction (empty pages skipped, real page numbers kept)
- Clear message for scanned/image-only PDFs (no OCR in version 1)
- Simple chunking for long lectures, then a combine step (not a blind concatenation)
- Structured output: overview, summary, key points, concepts, definitions, examples, formulas, exam focus, questions, MCQs, quick revision
- Page references. Invalid page numbers from the model are replaced with "none", never kept
- Chat with the lecture in four styles: Lecture Mode, Simple Explanation, Exam Mode, Quick Revision
- Answers "I couldn't find this information in the uploaded lecture." when the lecture doesn't contain the answer
- Markdown download of the study notes
- Friendly errors: missing/invalid token, rate limit, model loading or unavailable, timeout, empty or malformed output
- One malformed model response never crashes the app. That step is skipped and shown as a warning

## Architecture

```
PDF upload
   -> extract_pdf()      PyMuPDF, page by page
   -> clean_text()       whitespace, hyphenation
   -> chunk_pages()      ~6000 characters per chunk, page numbers kept
   -> analyze_chunk()    per chunk: key points, concepts, definitions, examples, formulas
   -> merge_chunk_results()   combine + remove duplicates
   -> gen_overview / gen_exam_focus / gen_questions / gen_mcqs / gen_quick_revision
   -> structured dict  ->  Streamlit tabs, chat, Markdown download
```

If the whole lecture is under ~12,000 characters, the final steps read the full text instead of the chunk notes, which gives better quality on short lectures.

Chat uses a deliberately simple keyword match to pick the most relevant pages. There are no embeddings, vector databases, or LangChain.

## Technology stack

Python, Streamlit, PyMuPDF, `requests`, and the Hugging Face Inference API (OpenAI-compatible chat endpoint at `router.huggingface.co`).

## Project structure

```
app.py             all logic (Part 1) and the Streamlit UI (Part 2)
requirements.txt   streamlit, pymupdf, requests
README.md          this file
```

`import app` in Colab loads only Part 1. The UI starts only with `streamlit run app.py`.

## Installation (local)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Hugging Face API setup

1. Create a free account at https://huggingface.co
2. Go to **Settings -> Access Tokens -> Create new token**. A fine-grained token with permission to **make calls to Inference Providers** is enough.
3. Copy the token. You will only see it once.

### Configure `HF_TOKEN` securely

Never paste the token into `app.py` and never commit it.

| Where | How |
|---|---|
| Local (macOS/Linux) | `export HF_TOKEN="hf_..."` |
| Local (Windows PowerShell) | `$env:HF_TOKEN="hf_..."` |
| Local, Streamlit secrets | create `.streamlit/secrets.toml` containing `HF_TOKEN = "hf_..."` |
| Streamlit Community Cloud | App settings -> **Secrets** -> `HF_TOKEN = "hf_..."` |
| Google Colab | Key icon (Secrets) in the left bar -> add `HF_TOKEN`, enable notebook access |

Add this to `.gitignore` before your first commit:

```
.streamlit/secrets.toml
.env
.venv/
```

## Run locally

```bash
streamlit run app.py
```

## Choosing and changing the model

The model name lives in **one place**, at the top of `app.py`:

```python
MODEL_NAME = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")
```

You can also switch without editing code: `export HF_MODEL="another/model"`.

Important things to know:

- Models differ in quality, context length, and whether they are served through the API at all. Availability changes over time.
- The default above is a reasonable starting point for instruction following and JSON output, but **you must confirm it works for your account** (Colab Cell 4) before trusting it.
- The model must be a **chat/instruction** model available through Hugging Face **Inference Providers**. Base (non-instruct) models will not work well.
- Larger models follow the "use only the lecture" and "return only JSON" rules more reliably. If you see many "invalid output" warnings, try a larger instruct model.
- If a model has a small context window, lower `MAX_CHUNK_CHARS` and `MAX_DIRECT_CHARS` at the top of `app.py`.

## Google Colab testing (do this before launching Streamlit)

Run the cells **in order**, one at a time. Each cell tests one piece, so if something breaks you know exactly which piece. You need: `app.py`, a Hugging Face token saved as the Colab secret `HF_TOKEN`, and one lecture PDF.

Each cell below lists: what it does, why, expected output, success signs, common errors, fixes.

---

### Cell 1: Install packages

```python
# CELL 1
!pip install -q streamlit pymupdf requests
```

- **What:** installs the three dependencies.
- **Why:** everything else depends on them.
- **Expected:** a short install log, no red errors.
- **Success:** the cell finishes.
- **Errors:** `Could not find a version` or network errors.
- **Fix:** re-run; check Colab has internet (Runtime -> Restart session if needed).

### Cell 2: Load the token securely

```python
# CELL 2
import os
from google.colab import userdata

os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
print("HF_TOKEN loaded:", bool(os.environ.get("HF_TOKEN")))   # prints True/False, never the token
```

- **What:** copies your Colab secret into the `HF_TOKEN` environment variable that `app.py` reads.
- **Why:** keeps the token out of the notebook text.
- **Expected:** `HF_TOKEN loaded: True`
- **Errors:** `SecretNotFoundError` or `NotebookAccessError`.
- **Fix:** open the key icon in the left bar, add a secret named exactly `HF_TOKEN`, and switch on **Notebook access**.

### Cell 3: Import libraries and `app.py`

```python
# CELL 3
from google.colab import files
import importlib, json, os, sys
import pymupdf

if not os.path.exists("app.py"):
    print("Select app.py from your computer:")
    files.upload()

sys.path.insert(0, os.getcwd())
import app
importlib.reload(app)          # re-run this cell after editing app.py
print("Imported OK. Model:", app.MODEL_NAME)
```

- **What:** uploads `app.py` (once) and imports it. This loads only the logic, not the UI.
- **Why:** all later cells call functions from `app.py`, so you test the real code.
- **Expected:** `Imported OK. Model: ...`
- **Errors:** `ModuleNotFoundError: app` or `pymupdf`.
- **Fix:** make sure the uploaded file is named exactly `app.py`; re-run Cell 1.

### Cell 4: Test Hugging Face authentication

```python
# CELL 4
try:
    reply = app.call_hf([{"role": "user", "content": "Reply with the single word: ready"}], max_tokens=20)
    print("Model replied:", reply)
except app.HFError as e:
    print("FAILED:", e)
```

- **What:** the smallest possible API request.
- **Why:** proves the token, the model name, and the endpoint all work before anything else.
- **Expected:** `Model replied: ready` (or similar).
- **Success:** any non-empty reply.
- **Errors and fixes:**
  - "API key is missing": Cell 2 was skipped or the runtime restarted. Re-run Cell 2.
  - "rejected the API key": the token is wrong or lacks Inference Providers permission. Create a new token.
  - "not found / not available" or "rejected the request": the model isn't served for your account. Change `MODEL_NAME` in `app.py`, re-upload, re-run Cell 3.
  - "loading" or "Rate limit": wait a minute and retry.

### Cell 5: Sample lecture and a first model test

```python
# CELL 5
SAMPLE = """Data Mining - Lecture 1
Data mining is the process of discovering useful patterns in large datasets.
Classification is a supervised learning technique. A model learns from labeled data to predict predefined classes.
For example, an email filter classifies messages as spam or not spam.
Clustering groups similar data points without predefined labels. It is an unsupervised technique.
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Key steps of the data mining process: data cleaning, data selection, pattern discovery, evaluation."""

msgs = [{"role": "system", "content": app.SYSTEM_PROMPT},
        {"role": "user", "content": "In two sentences, what is this lecture about?\n\n" + SAMPLE}]
print(app.call_hf(msgs))
```

- **What:** sends a small hand-written lecture to the model.
- **Why:** checks the model gives a sensible answer on known text.
- **Expected:** two sentences about data mining, classification, clustering.
- **Errors:** an answer that mentions things not in the sample means the model ignores instructions. Try a bigger instruct model.

### Cell 6: Summarization

```python
# CELL 6
msgs = [{"role": "system", "content": app.SYSTEM_PROMPT},
        {"role": "user", "content": "Summarize this lecture in simple language using ONLY the text below.\n\n" + SAMPLE}]
print(app.call_hf(msgs))
```

- **What:** asks for a summary.
- **Why:** the summary is the core output.
- **Expected:** a short paragraph.
- **Success:** every statement can be found in `SAMPLE`. **Read it and check for invented facts** (for example extra algorithms or history that isn't in the text).
- **Errors:** invented details. **Fix:** use a stronger model.

### Cell 7: Key points

```python
# CELL 7
msgs = [{"role": "system", "content": app.SYSTEM_PROMPT},
        {"role": "user", "content": "List the key points as a numbered list, using ONLY the text below.\n\n" + SAMPLE}]
print(app.call_hf(msgs))
```

- **What:** asks for a numbered list.
- **Expected:** about 4 to 7 numbered points.
- **Errors:** paragraph instead of a list. Minor. The app itself asks for JSON, which Cell 8 tests.

### Cell 8: Structured extraction

```python
# CELL 8
chunk = {"text": "[Page 1]\n" + SAMPLE, "pages": [1], "start_page": 1, "end_page": 1}
raw = app.call_hf([{"role": "system", "content": app.SYSTEM_PROMPT},
                   {"role": "user", "content": app.fill(app.CHUNK_PROMPT, start=1, end=1, text=chunk["text"])}],
                  max_tokens=2000)
print("RAW OUTPUT:\n", raw[:1500])

result = app.analyze_chunk(chunk)          # the app's real function (parses + cleans)
for kind, items in result.items():
    print(f"{kind}: {len(items)} item(s)")

focus = app.gen_exam_focus([{"page": 1, "text": SAMPLE}], SAMPLE)
print("exam_focus:", focus)
```

- **What:** requests the JSON structure used by the app, then runs it through the app's cleaning.
- **Why:** the whole app depends on structured output.
- **Expected:** counts such as `key_points: 5`, `concepts: 2`, `definitions: 2`, `examples: 1`, `formulas: 1`; a non-empty `exam_focus`.
- **Success:** the formula `Accuracy = ...` appears under formulas, the spam example under examples, and all page numbers are `1` or `None`.
- **Errors:** "The model returned invalid output for one section".
- **Fix:** run the cell again (models vary), or use a stronger model.

### Cell 9: JSON parsing and malformed output

```python
# CELL 9
tests = {
    "clean JSON":        '{"a": 1}',
    "markdown fenced":   '\x60\x60\x60json\n{"a": 1}\n\x60\x60\x60',   # \x60 is the backtick character
    "trailing comma":    '{"a": [1, 2,],}',
    "text around JSON":  'Sure! Here you go: {"a": 1} Hope that helps.',
    "truncated":         '{"a": [1, 2',
    "not JSON at all":   'I cannot do that.',
    "empty":             '',
}
for name, text in tests.items():
    print(f"{name:18} -> {app.parse_json_safe(text)}")
```

- **What:** feeds good and bad text to `parse_json_safe`.
- **Why:** proves bad model output cannot crash the app.
- **Expected:** the first four give `{'a': ...}`; the last three give `None`. No exception.

### Cell 10: Upload a real PDF

```python
# CELL 10
up = files.upload()                       # choose your lecture PDF
name = list(up)[0]
doc = app.load_pdf(up[name], name)

print("File:", doc["name"])
print("Pages:", doc["num_pages"], "| pages with text:", len(doc["pages"]))
print("Extracted characters:", doc["chars"])
if doc["scanned"]:
    print("This PDF appears to contain scanned/image based pages. Text extraction could not find enough readable text.")
else:
    print("\nFirst 400 characters:\n", doc["pages"][0]["text"][:400])
```

- **What:** extracts the text with PyMuPDF.
- **Why:** everything downstream uses this text.
- **Expected:** file name, page count, characters, readable text.
- **Errors:**
  - Scanned-PDF message: the PDF has images instead of text. OCR isn't in version 1. Try a text-based export of the slides.
  - "password protected": remove the password first.
  - Garbled text: the PDF uses unusual font encoding. Re-export it from PowerPoint or Word.

### Cell 11: Page boundaries

```python
# CELL 11
for p in doc["pages"][:5]:
    print("=" * 30, "Page", p["page"], "=" * 30)
    print(p["text"][:300])
```

- **What:** prints the first pages separately.
- **Why:** page numbers are used for source references.
- **Success:** the page numbers match the real PDF (blank pages are skipped, so numbers may jump).

### Cell 12: Text cleaning

```python
# CELL 12
pdf_doc = pymupdf.open(stream=up[name], filetype="pdf")
first_page = doc["pages"][0]["page"] - 1
raw_text = pdf_doc[first_page].get_text("text")
print("BEFORE (repr):\n", repr(raw_text[:400]))
print("\nAFTER:\n", app.clean_text(raw_text)[:400])
```

- **What:** compares raw and cleaned text.
- **Expected:** extra spaces, odd characters, and repeated blank lines are gone; the words are unchanged.

### Cell 13: Chunking

```python
# CELL 13
chunks = app.chunk_pages(doc["pages"])
print("Number of chunks:", len(chunks))
print("Chunk sizes:", [len(c["text"]) for c in chunks])
print("\nFIRST chunk (pages", chunks[0]["start_page"], "-", chunks[0]["end_page"], "):\n", chunks[0]["text"][:300])
print("\nLAST chunk (pages", chunks[-1]["start_page"], "-", chunks[-1]["end_page"], "):\n", chunks[-1]["text"][:300])
```

- **What:** splits the lecture into chunks.
- **Why:** long lectures don't fit in one model call.
- **Expected:** sizes at or below about 6000; a short lecture gives just 1 chunk; page ranges cover the whole document.
- **Errors:** chunks too big for your model. Lower `MAX_CHUNK_CHARS`.

### Cell 14: Analyze chunks with the model

```python
# CELL 14
chunk_results = []
for i, chunk in enumerate(chunks, start=1):
    print(f"Analyzing chunk {i}/{len(chunks)} (pages {chunk['start_page']}-{chunk['end_page']})...")
    try:
        r = app.analyze_chunk(chunk)
        chunk_results.append(r)
        print("   ok:", {k: len(v) for k, v in r.items()})
    except app.HFError as e:
        print("   FAILED:", e)
        if e.fatal:
            break
```

- **What:** sends each chunk to the model.
- **Why:** this is the slowest step; testing it alone saves time.
- **Expected:** one `ok:` line per chunk with counts.
- **Errors:** rate limit or timeout. Wait and re-run. A failed chunk is skipped so the notebook keeps going.

### Cell 15: Combine into the final analysis

```python
# CELL 15
analysis = app.analyze_lecture(
    doc["pages"],
    options={"questions": False, "mcqs": False, "quick_revision": False},   # these come in Cells 17-19
    progress=lambda stage, msg="": print(f"[{stage}] {msg}"),
    chunk_results=chunk_results,       # reuse Cell 14 results, no repeat calls
)
print("Warnings:", analysis["warnings"])
```

- **What:** merges chunk results, writes the overview, summary, key points, and exam focus.
- **Expected:** progress lines, then `Warnings: []` (or a few warnings for steps that failed).

### Cell 16: Show the results

```python
# CELL 16
print("TITLE:", analysis["title"] or "(not clearly stated)")
print("SUBJECT:", analysis["subject"] or "(not clearly stated)")
print("\nSUMMARY:\n", analysis["summary"])
print("\nKEY POINTS:")
for k in analysis["key_points"]:
    print(f"  - {k['text']}  [page {k['page']}]")
print("\nCONCEPTS:")
for c in analysis["concepts"][:8]:
    print(f"  - {c['name']}: {c['simple']}  [page {c['page']}]")
print("\nDEFINITIONS:")
for d in analysis["definitions"][:8]:
    print(f"  - {d['term']} ({d['type']}): {d['definition']}  [page {d['page']}]")
print("\nEXAM FOCUS:")
for x in analysis["exam_focus"]:
    print(f"  - {x['topic']}: {x['reason']}  [page {x['page']}]")
```

- **What:** prints each section.
- **Success:** open your PDF and spot-check 3 or 4 items and their page numbers. **This is your hallucination check.** Titles/subjects the PDF doesn't state should show "(not clearly stated)".

### Cell 17: Exam questions

```python
# CELL 17
material = app.get_material(doc["pages"], app.merge_chunk_results(chunk_results))
analysis["questions"] = app.gen_questions(material)
for label, qs in analysis["questions"].items():
    print(label.upper())
    for i, q in enumerate(qs, 1):
        print(f"  {i}. {q}")
```

- **Expected:** short, conceptual, and long questions about topics in your lecture.
- **Check:** no question about things the lecture doesn't cover.

### Cell 18: MCQs

```python
# CELL 18
analysis["mcqs"] = app.gen_mcqs(doc["pages"], material)
for i, m in enumerate(analysis["mcqs"], 1):
    print(f"\nQ{i}. {m['question']}")
    for letter, opt in zip("ABCD", m["options"]):
        print(f"   {letter}. {opt}")
    print("   Answer:", m["answer"], "|", m["explanation"])
```

- **Expected:** 5 to 15 questions, 4 options each, one correct answer.
- **Check:** the marked answer is right according to the lecture. Malformed MCQs are dropped automatically.

### Cell 19: Quick revision

```python
# CELL 19
analysis["quick_revision"] = app.gen_quick_revision(material)
for q in analysis["quick_revision"]:
    print(f"{q['term']} -> {q['gist']}")
```

- **Expected:** 10 to 15 short "term -> gist" lines you can read in under 2 minutes.

### Cell 20: Chat and the hallucination test

```python
# CELL 20
history = []
def ask(question, mode="Lecture Mode"):
    answer = app.answer_question(question, history, mode, doc["pages"], analysis)
    history.extend([{"role": "user", "content": question}, {"role": "assistant", "content": answer}])
    print(f"\nQ: {question}  [{mode}]\nA: {answer}")

ask("What is the main topic of this lecture?")                 # replace with a real question from your PDF
ask("Explain it in simple words.", "Simple Explanation")
ask("Who won the FIFA World Cup in 2010?")                     # NOT in the lecture
ask("What is the capital of France?")                          # NOT in the lecture
```

- **What:** asks in-lecture and out-of-lecture questions.
- **Success:** the first two answers use lecture content. The last two must answer `I couldn't find this information in the uploaded lecture.`
- **If the model answers the out-of-lecture questions:** this is a model limitation. Switch to a stronger instruct model. Do not ship with a model that fails this test.

### Cell 21: Download study notes

```python
# CELL 21
markdown = app.build_markdown(analysis, doc)
with open("lecture_study_notes.md", "w", encoding="utf-8") as f:
    f.write(markdown)
print(markdown[:1500])
files.download("lecture_study_notes.md")
```

- **Expected:** a Markdown preview and a file download containing every section.

### Cell 22: Launch Streamlit

Only run this after Cells 1-21 pass.

```python
# CELL 22
!npm install -q localtunnel
!streamlit run app.py &>/content/streamlit_logs.txt &
import urllib.request, time
time.sleep(5)
print("Tunnel password (IP):", urllib.request.urlopen("https://ipv4.icanhazip.com").read().decode().strip())
!npx localtunnel --port 8501
```

- **What:** starts the real app and exposes it through a temporary public URL.
- **Expected:** a `https://....loca.lt` link. Open it and enter the printed IP as the password.
- **Errors:** page won't load. Run `!tail /content/streamlit_logs.txt`. `HF_TOKEN` from Cell 2 is inherited automatically because it is set in `os.environ`.
- This URL is temporary and public. Don't share it. For real deployment use Streamlit Community Cloud.

## Deploy to Streamlit Community Cloud

1. Push `app.py`, `requirements.txt`, `README.md` to GitHub (**without** any token or `secrets.toml`).
2. Create the app at https://share.streamlit.io and select `app.py`.
3. In **Settings -> Secrets** add `HF_TOKEN = "hf_..."`.

## Troubleshooting

| Problem | Likely cause | Fix |
|---|---|---|
| "API key is missing" | `HF_TOKEN` not set in this environment | Set the env variable or Streamlit secret; restart the app |
| "rejected the API key" | Wrong token or missing Inference Providers permission | Create a new token with that permission |
| "model was not found or not available" | Model not served through the API for your account | Change `MODEL_NAME` |
| "Rate limit reached" | Too many requests | Wait, or use fewer options / a shorter PDF |
| "model is loading" | Provider is starting the model | Wait 30 to 60 seconds and retry |
| "invalid output" warnings | Model doesn't follow JSON instructions | Use a larger instruct model |
| "scanned/image based" message | PDF has no text layer | Use a text-based PDF (OCR is a future improvement) |
| Empty tabs | That step failed. See warnings in the Overview tab | Re-run **Analyze lecture** |

## Limitations

- No OCR: scanned PDFs are not supported.
- Text only: diagrams, images, and charts in slides are ignored. Formulas that PDF extraction breaks up (fractions, symbols) may be garbled.
- Quality depends on the model. Small models can miss items or occasionally ignore rules. **Always verify with the page references.**
- "Exam focus" and questions are suggestions based on the lecture's structure, not predictions.
- Very long lectures are analyzed through condensed chunk notes, so small details can be lost.
- Chat retrieval is keyword based, so a question using different words than the lecture may miss the right page.
- Free Hugging Face usage has rate limits.

## Future improvements

- OCR for scanned PDFs (e.g. Tesseract or a vision model)
- PDF export of study notes
- Semantic search for chat (embeddings), once the simple version is working well
- Anki/flashcard export
- Support for PowerPoint and Word files

## Security

`HF_TOKEN` is read only from the environment or Streamlit secrets. It is never printed, logged, or shown in error messages. Never commit it.
