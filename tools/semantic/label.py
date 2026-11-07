"""Semantic category of every recall, labelled by Claude Haiku.

usage: label.py semantic.json labels.json (--dry-run | --sample N | --all | --open N) [--model M] [--workers K]
--open writes free codes (no taxonomy) to its own file; pass a different labels path for it.
Only the recall text, the fragment text and the taxonomy leave the laptop; subject IDs are
cache keys here and never part of a request (--dry-run checks that on every payload).
API key: ANTHROPIC_API_KEY, else the gitignored .env at the repo root. Never printed.
"""
import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

# v2: Stawarczyk et al. (2011) on-task / task-related interference / external distraction / mind wandering,
# MW split by time, plus story-cued associations (a group the open coding found and v1 had no place for)
TAX = "v2"
CATS = ["story", "association", "task", "environment", "self_past", "self_present", "self_future", "blank"]
PRICE = {"claude-haiku-5-5": (0.10, 0.50), "claude-haiku-4-5": (1.00, 5.00)}   # USD per 1M input / output tokens
MW_BINS = [(0, 10), (11, 25), (26, 50), (51, 75), (76, 99)]

SYSTEM = """You label short thought reports from a psychology experiment on mind wandering.
A participant listened to a ~45-second fragment of a Polish audio story, was stopped, and said what they had been thinking about. You get the story fragment and the participant's report (both in Polish). Choose the ONE category for the content that takes most of the report, and list any other categories that are also clearly present.

Categories:
- story: the story itself: retelling it, imagining its scenes or characters, guessing what happens next, judging the characters, the plot or the narration, or a plain statement of having followed the story ("I was focused on the story").
- association: the story reminded the person of something outside it: another book, film or fairy tale, or their own life, and the report stays tied to that link ("the cottage looked like the one in a film I saw", "Janek reminded me of my grandfather").
- task: the experiment and the person's own performance in it: how focused they are, whether they remember the story, how long it will last, wanting it to end, boredom with the task, what they will say or said at a stop, the procedure, the fixation cross, the equipment.
- environment: sensations and surroundings: pain, itching, cold or heat, tiredness or sleepiness, hunger, sounds, the light, the room.
- self_past: own memories and past events, not reached through the story.
- self_present: own life with no clear time: people they care about, their feelings, daydreams, fantasies. Opinions, feelings or images about the story and its characters are story, not self_present.
- self_future: own plans and things ahead: meals, shopping, travel, exams, work, errands, worries about what is coming.
- blank: no content: "nothing", "I don't know", "I don't remember", or drifting off without saying what about.

Rules:
- category is the content that takes most of the report. When the story and another category take about the same space, choose the other category: a departure from the story is the event of interest.
- also: the other categories clearly present in the report, without category itself; an empty list when there are none.
- Judge only what is said; do not infer from the fragment what the person "must" have thought.
- topic: at most 5 words in Polish naming what the report is about.
- confidence: your probability (0-1) that category is right.

Examples (report -> category, also):
"Karolina chciała wypłynąć statkiem, ale bała się, co powie ojciec." -> story, []
"Byłem cały czas skupiony na historii, zastanawiałem się, czy Jan dostanie tę pracę." -> story, []
"Ta chatka przypominała mi dom z bajki o Jasiu i Małgosi, którą czytała mi mama." -> association, [story]
"Kiedy mówili o młynie, przypomniał mi się młyn koło domu dziadków." -> association, []
"Zastanawiałam się, czy dobrze zapamiętuję szczegóły, bo potem trzeba to opowiedzieć." -> task, []
"Myślałem, ile jeszcze tych fragmentów zostało, trochę mi się już nudzi." -> task, []
"Swędziało mnie oko i cały czas o tym myślałam, prawie nie słuchałam." -> environment, [task]
"Było mi zimno w ręce." -> environment, []
"Przypomniały mi się wakacje nad morzem z dziadkiem." -> self_past, []
"Myślałem o tym, jak wczoraj pokłóciłem się z bratem." -> self_past, []
"Myślałam o mojej przyjaciółce i o tym, jak jej się teraz układa." -> self_present, []
"Wyobrażałem sobie, że gram koncert na dużej scenie." -> self_present, []
"Myślałam, co zrobić na obiad i co kupić po drodze." -> self_future, []
"Martwię się egzaminem w przyszłym tygodniu, a Janek coś tam rzeźbił." -> self_future, [story]
"Nic." -> blank, []
"Trochę odpłynęłam, nie wiem nawet o czym." -> blank, []"""

SCHEMA = {"type": "object",
          "properties": {"category": {"type": "string", "enum": CATS},
                         "also": {"type": "array", "items": {"type": "string", "enum": CATS}},
                         "confidence": {"type": "number"},
                         "topic": {"type": "string"}},
          "required": ["category", "also", "confidence", "topic"], "additionalProperties": False}

OPEN_SYSTEM = """You code short thought reports from a psychology experiment on mind wandering, as a qualitative researcher doing open coding.
A participant listened to a ~45-second fragment of a Polish audio story, was stopped, and said what they had been thinking about. You get the story fragment and the participant's report (both in Polish).
Give 1-3 codes for the report, ordered by how much of the report each takes. A code is 2-5 Polish words naming the KIND of thought at a general level, as you would code many reports, not its specific details (write «plany na resztę dnia», not «kupić chleb w Biedronce»). There is no predefined list; name what you see.
Judge only what is said; do not infer from the fragment what the person "must" have thought."""

OPEN_SCHEMA = {"type": "object",
               "properties": {"codes": {"type": "array", "items": {"type": "string"}}},
               "required": ["codes"], "additionalProperties": False}

CLUSTER_SYSTEM = """You get open codes that a coder gave to thought reports from a mind-wandering experiment (people listened to an audio story and said what they had been thinking about). Each line: code number, the code, and in how many reports it appeared.
Propose 5-10 categories that a researcher could then use to code every report with one category, so that together they cover all the codes. Name each category in Polish (2-4 words) and define it in one Polish sentence. Do not list the codes."""

CLUSTER_SCHEMA = {"type": "object",
                  "properties": {"categories": {"type": "array", "items": {
                      "type": "object",
                      "properties": {"name": {"type": "string"}, "definition": {"type": "string"}},
                      "required": ["name", "definition"], "additionalProperties": False}}},
                  "required": ["categories"], "additionalProperties": False}

ASSIGN_SYSTEM = """You get numbered categories for coding thought reports from a mind-wandering experiment, and a numbered list of open codes. For every code return one pair: the code number and the number of the category it belongs to."""

ASSIGN_SCHEMA = {"type": "object",
                 "properties": {"assign": {"type": "array", "items": {
                     "type": "object", "properties": {"code": {"type": "integer"}, "category": {"type": "integer"}},
                     "required": ["code", "category"], "additionalProperties": False}}},
                 "required": ["assign"], "additionalProperties": False}

# ticket 11: the participants' own probe, worded as on their screen; self-report and categories never sent
RATE_SYSTEM = """You read thought reports from a psychology experiment on mind wandering.
A participant listened to a ~45-second fragment of a Polish audio story. At a stop the screen said «Opowiedz przebieg swoich myśli od ostatniego przerwania historii.» and they answered aloud: that answer is the report. Then the screen asked «W jakim stopniu byłeś w tym momencie rozproszony podczas słuchania historii?» and they marked a continuous scale from 0 «Brak rozproszenia» to 100 «Pełne rozproszenie».
You get the story fragment and the report. Estimate the number this person marked on that scale. Use the whole scale. Judge only what is said in the report.
reason: one short Polish sentence saying what in the report your number rests on."""

RATE_SCHEMA = {"type": "object",
               "properties": {"mw": {"type": "integer"}, "reason": {"type": "string"}},
               "required": ["mw", "reason"], "additionalProperties": False}

ap = argparse.ArgumentParser()
ap.add_argument("semantic")
ap.add_argument("out")
mode = ap.add_mutually_exclusive_group(required=True)
mode.add_argument("--dry-run", action="store_true")
mode.add_argument("--sample", type=int)
mode.add_argument("--all", action="store_true")
mode.add_argument("--open", type=int, help="open coding of N recalls, then one call that groups the codes")
ap.add_argument("--rate", action="store_true", help="with --sample/--all: rate distraction 0-100 instead of the category")
ap.add_argument("--model", default="claude-haiku-5-5", choices=sorted(PRICE))
ap.add_argument("--workers", type=int, default=6)
a = ap.parse_args()

sem = json.load(open(a.semantic, encoding="utf-8"))
frag_text = {fr["name"]: " ".join(fr["ss"]) for fr in sem["fragments"]}
R = sem["recalls"]
key = lambda r: f"{r['s']}|{r['t']}"
subjects = {r["s"] for r in R}
cache = json.load(open(a.out, encoding="utf-8")) if os.path.exists(a.out) else {}


def user_msg(r):
    return (f"<fragment>\n{frag_text[r['f']]}\n</fragment>\n\n"
            f"<report>\n{' '.join(r['rs'])}\n</report>")


def check_payload(text):
    leaked = [s for s in subjects if s in text]
    assert not leaked, f"subject ID in payload: {leaked[:3]}"


if a.dry_run:
    r = R[0]
    for x in R:
        check_payload(user_msg(x))
    print(f"system prompt: {len(SYSTEM)} chars; {len(R)} payloads checked, no subject ID in any")
    print(f"model {a.model}; one payload as sent:\n---\n{user_msg(r)}\n---")
    sys.exit(0)

if not os.environ.get("ANTHROPIC_API_KEY"):
    env = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env")
    if os.path.exists(env):
        for line in open(env, encoding="utf-8"):
            k, _, v = line.strip().partition("=")
            if k == "ANTHROPIC_API_KEY" and v:
                os.environ[k] = v.strip().strip('"').strip("'")
if not os.environ.get("ANTHROPIC_API_KEY"):
    sys.exit("no API key: export ANTHROPIC_API_KEY or put it in .env at the repo root")

import anthropic   # after the key check, so --dry-run needs no SDK

client = anthropic.Anthropic(max_retries=5)
usage = {"in": 0, "out": 0, "calls": 0}


def call(system, schema, content, effort="low", max_tokens=2000):
    """One structured request; returns the parsed JSON, None on refusal or unparsable text."""
    resp = client.messages.create(
        model=a.model, max_tokens=max_tokens, system=system,
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
        messages=[{"role": "user", "content": content}])
    usage["in"] += resp.usage.input_tokens + (resp.usage.cache_read_input_tokens or 0) + (resp.usage.cache_creation_input_tokens or 0)
    usage["out"] += resp.usage.output_tokens
    usage["calls"] += 1
    text = next((b.text for b in resp.content if b.type == "text"), "")
    try:
        if resp.stop_reason != "refusal":
            return json.loads(text)
    except ValueError:
        pass
    print(f"no JSON: stop_reason {resp.stop_reason}, {resp.usage.output_tokens} output tokens, text {text[:60]!r}", file=sys.stderr)
    return None


def ask(r):
    for attempt in range(2):   # one re-ask on an invalid answer
        d = call(SYSTEM, SCHEMA, user_msg(r))
        try:
            if d["category"] in CATS and 0 <= float(d["confidence"]) <= 1:
                also = [c for c in dict.fromkeys(d.get("also", [])) if c in CATS and c != d["category"]]
                return {"category": d["category"], "also": also, "confidence": round(float(d["confidence"]), 2),
                        "topic": str(d["topic"]).strip(), "model": a.model, "tax": TAX}
        except (KeyError, TypeError, ValueError):
            pass
    return {"error": f"invalid answer: {str(d)[:80]}"}


def ask_open(r):
    for attempt in range(2):
        d = call(OPEN_SYSTEM, OPEN_SCHEMA, user_msg(r))
        codes = [str(c).strip().lower() for c in (d or {}).get("codes", []) if str(c).strip()][:3]
        if codes:
            return {"codes": codes, "model": a.model}
    return {"error": f"invalid answer: {str(d)[:80]}"}


def ask_rate(r):
    for attempt in range(2):
        d = call(RATE_SYSTEM, RATE_SCHEMA, user_msg(r))
        try:
            if 0 <= int(d["mw"]) <= 100:
                return {"mw": int(d["mw"]), "reason": str(d["reason"]).strip(), "model": a.model}
        except (KeyError, TypeError, ValueError):
            pass
    return {"error": f"invalid answer: {str(d)[:80]}"}


def stratified(n, seed):
    rng = random.Random(seed)
    per = max(1, n // len(MW_BINS))
    return [x for lo, hi in MW_BINS for x in rng.sample([r for r in R if r["mw"] is not None and lo <= r["mw"] <= hi], per)]


def save():
    tmp = a.out + ".tmp"
    json.dump(cache, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    os.replace(tmp, a.out)


def report_cost(n):
    pin, pout = PRICE[a.model]
    cost = usage["in"] / 1e6 * pin + usage["out"] / 1e6 * pout
    print(f"{a.model}: {usage['calls']} calls for {n} recalls, {usage['in']} input + {usage['out']} output tokens, "
          f"cost ${cost:.4f}, {time.time() - t0:.0f} s -> {a.out}")
    return cost


t0 = time.time()
if a.open:
    # seed 1, not 0: the open pass should not reuse the reports the fixed taxonomy was tuned on
    sel = stratified(a.open, seed=1)
    todo = [r for r in sel if key(r) not in cache or "error" in cache[key(r)]]
    for r in todo:
        check_payload(user_msg(r))
    with ThreadPoolExecutor(a.workers) as ex:
        for r, lab in zip(todo, ex.map(ask_open, todo)):
            cache[key(r)] = lab
    save()
    coded = [(r, cache[key(r)]["codes"]) for r in sel if "codes" in cache.get(key(r), {})]
    freq = {}
    for _, cs in coded:
        for c in cs:
            freq[c] = freq.get(c, 0) + 1
    codes = sorted(freq, key=lambda c: -freq[c])
    listing = "\n".join(f"{i}. {c} ({freq[c]})" for i, c in enumerate(codes))
    check_payload(listing)
    # two steps: one list of code numbers per category ran into a repetition loop at 468 codes
    d = call(CLUSTER_SYSTEM, CLUSTER_SCHEMA, listing, effort="medium", max_tokens=4000)
    assert d, "cluster call failed"
    cats = "\n".join(f"{i}. {c['name']}: {c['definition']}" for i, c in enumerate(d["categories"]))
    of = {}
    for attempt in range(3):   # pairs, not a positional list: a positional list came back 94-110 long for 100 codes
        left = [i for i, c in enumerate(codes) if c not in of]
        for b in range(0, len(left), 50):
            chunk = left[b:b + 50]
            got = call(ASSIGN_SYSTEM, ASSIGN_SCHEMA, f"<categories>\n{cats}\n</categories>\n\n<codes>\n" +
                       "\n".join(f"{i}. {codes[i]}" for i in chunk) + "\n</codes>", max_tokens=4000)
            for p in (got or {}).get("assign", []):
                if p.get("code") in chunk and 0 <= p.get("category", -1) < len(d["categories"]):
                    of.setdefault(codes[p["code"]], p["category"])
    for ci, cat in enumerate(d["categories"]):
        cat["codes"] = [codes.index(c) for c in codes if of.get(c) == ci]
    cache["_clusters"] = d
    save()
    print(f"{len(coded)} recalls coded, {len(codes)} distinct codes, {len(codes) - len(of)} left ungrouped")
    for ci, cat in enumerate(d["categories"]):
        first = [r for r, cs in coded if of.get(cs[0]) == ci]
        anyc = [r for r, cs in coded if any(of.get(c) == ci for c in cs)]
        mw = sorted(r["mw"] for r in first)
        print(f"\n## {cat['name']} — first code {len(first)}, any code {len(anyc)}, "
              f"median MW {mw[len(mw) // 2] if mw else '-'}\n   {cat['definition']}")
        print("   codes: " + "; ".join(codes[k] for k in cat["codes"][:12] if 0 <= k < len(codes)))
        for r in first[:3]:
            print(f"   · MW {r['mw']:>2.0f} {' '.join(r['rs'])[:120]}")
    report_cost(len(todo))
    sys.exit(0)

if a.sample:
    todo = stratified(a.sample, seed=0)
else:
    todo = R
sel = todo
if a.rate:
    todo = [r for r in todo if "mw" not in cache.get(key(r), {})]
else:
    todo = [r for r in todo if cache.get(key(r), {}).get("tax") != TAX]   # errors and older taxonomies are redone
for r in todo:
    check_payload(user_msg(r))

with ThreadPoolExecutor(a.workers) as ex:
    for r, lab in zip(todo, ex.map(ask_rate if a.rate else ask, todo)):
        cache[key(r)] = lab
save()

if a.rate:
    import statistics
    got = [(r, cache[key(r)]) for r in sorted(sel, key=lambda r: r["mw"]) if "mw" in cache.get(key(r), {})]
    if a.sample:
        for r, lab in got:
            print(f"self {r['mw']:>2.0f} · model {lab['mw']:>3} · {lab['reason'][:70]:<70} · {' '.join(r['rs'])[:90]}")
    xs, ys = [r["mw"] for r, _ in got], [lab["mw"] for _, lab in got]
    print(f"r(self, model) = {statistics.correlation(xs, ys):.3f} over {len(got)} recalls (pooled, between and within subject)")
    cost = report_cost(len(todo))
    if a.sample:
        print(f"full run estimate: ${cost / max(1, len(todo)) * len(R):.2f} for {len(R)} recalls")
    sys.exit(0)

shown = todo if a.sample else []
for r in sorted(shown, key=lambda r: r["mw"]):
    lab = cache[key(r)]
    cat = lab.get("category", "ERROR") + "".join(f"+{c}" for c in lab.get("also", []))
    print(f"MW {r['mw']:>2.0f} · {cat:<26} · {lab.get('confidence', ''):<4} · "
          f"{lab.get('topic', lab.get('error', ''))[:40]:<40} · {' '.join(r['rs'])[:110]}")
cost = report_cost(len(todo))
print(f"cache {len(cache)} labels ({sum(1 for k in cache if 'error' in cache[k])} errors)")
if a.sample:
    print(f"full run estimate: ${cost / max(1, len(todo)) * len(R):.2f} for {len(R)} recalls")
