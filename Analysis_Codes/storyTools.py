"""
storyTools.py -- tools for analysing participants' recalls (utterances) in the Strategic MW experiment.

Sources:
  * GitHub/Experiment_MW_Story/analysis/storyTools.py          (story loading, spaCy, autobiography, WMD, viewer)
  * GitHub/startegicMW_lite/.../NEW_WORD_ANALYSIS/analysis_functions.py
        (NEWER: preNorm/lems, dictStemmer with part-of-speech filter, TF-IDF, bertSentenceCosineSim with GloVe)

Heavy libraries (morfeusz2, spacy, nltk, gensim, sklearn) are imported inside the functions,
so importing this module is fast and does not require the whole NLP stack.

Contents
  1. Normalisation dictionaries   preNorm, lems, CONTENT_TAGS, NAMES
  2. Story                        loadStory, cleanFragment, splitStoryEntity
  3. Tokenisation / lemmatisation load_stopwords, preprocess, dictStemmer, spacyMorph
  4. Utterance features           lexical_diversity, lexical_diversity_storywise, autobiographyAnalysis
  5. Semantic decoupling          bertSentenceCosineSim, wmd_gensim
  6. TF-IDF                       tf_idf_analysis
  7. Browsing                     history_viewer
  8. Metrics report               utterance_report
"""
import os
import ast
import re

import numpy as np
import pandas as pd

os.environ.setdefault("USE_TF", "0")   # make transformers ignore TensorFlow


# =============================================================================
# 1. Normalisation dictionaries (version from NEW_WORD_ANALYSIS)
# =============================================================================

# Inflected forms of the characters' names -> nominative (applied BEFORE Morfeusz)
preNorm = {
    "Janka": "Janek", "Jankowi": "Janek", "Janku": "Janek", "Jana": "Janek", "Jan": "Janek",
    "Janowi": "Janek", "Janie": "Janek", "Janem": "Janek", "Jankiem": "Janek",
    "Karoliny": "Karolina", "Karolinie": "Karolina", "Karolino": "Karolina",
    "Karolinę": "Karolina", "Karoliną": "Karolina",
}

# Fixes for wrong Morfeusz lemmas (applied AFTER Morfeusz)
lems = {
    "trocha": "trochę", "troszka": "trochę", "myślić": "myśli", "oczyć": "oczy", "swoić": "swój",
    "tyli": "tyle", "pewne": "pewny", "dzienić": "dzień", "miecić": "mieć", "Janko": "Janek",
    "słowić": "słowo", "spokoić": "spokój", "wzorzać": "wzór", "stola": "stół", "ciszyć": "cisza",
    "wieczor": "wieczór", "potrzeb": "potrzeba", "got": "gotowy", "ciężeć": "ciężko",
    "odbiegły": "odbiegać", "rozprószać": "rozproszyć", "rozprószyć": "rozproszyć",
    "drewny": "drewno", "pracowny": "pracownia", "szwalny": "szwalnia", "światły": "światło",
    "rad": "rada", "wzorać": "wzór", "chęcić": "chęć", "płaszczyć": "płaszcz", "dusić": "dusza",
    "dumać": "duma", "zlecieć": "zlecenie", "kieszenia": "kieszeń", "powierzchni": "powierzchnia",
    "ranka": "ranek", "prace": "praca", "rozmowy": "rozmowa", "statki": "statek", "szopy": "szopa",
    "żagle": "żagiel", "szaty": "szata", "wpływy": "wpływ", "bracić": "brat", "cienić": "cień",
    "farbić": "farba", "gotowić": "gotowy", "głębić": "głębia", "kopić": "kopia", "żalić": "żal",
    "złościć": "złość",
}

# Morfeusz (NKJP) tags of content words: nouns, adjectives, verbs, adverbs
CONTENT_TAGS = {"subst", "depr", "adj", "adja", "adjc", "adjp", "fin", "praet", "inf", "impt",
                "imps", "ger", "pact", "ppas", "pcon", "pant", "adv"}

# Names from the story (not lower-cased before analysis)
NAMES = {"Janek", "Jan", "Karolina", "Magda"}

# aliases matching the names used in analysis_functions.py
content = CONTENT_TAGS
names = NAMES


# =============================================================================
# 2. Story
# =============================================================================

def loadStory(filename):
    """
        Loads the raw story and gets rid of the section labels

    Returns: (list_of_paragraphs, list_of_fragment_names) -> e.g. ['KAROLINA_1', 'JANEK_1', ...]
    """
    skip = {'\n', '#Wstęp\n', '#Janek\n', '#Kasia\n', '#Karolina\n', '#Koniec\n'}
    with open(filename, 'r', encoding='utf-8') as f:
        content_ = f.readlines()
    newContent, partList = [], []
    karolina, janek = 1, 1
    for line in content_:
        if line not in skip:
            newContent.append(line)
        if line == '#Janek\n':
            partList.append(f"JANEK_{janek}"); janek += 1
        if line == '#Karolina\n':
            partList.append(f"KAROLINA_{karolina}"); karolina += 1
    return newContent, partList


def cleanFragment(text):
    """
    Tokenizes words in a Polish text
    (punctuation is removed from the tokens)
    """
    from nltk import word_tokenize
    ensure_nltk()
    words = word_tokenize(text, language='polish')
    return [re.sub(r'[^\w\s]', '', t) for t in words if re.sub(r'[^\w\s]', '', t)]


def splitStoryEntity(cont):
    """
    needs adjusting depending on whether the story has an introduction or not. It splits the story into two lists: firstEntity and secondEntity, which contain the fragments of the story for each character.
    """
    noHeader = [cont[i] for i in range(1, len(cont), 2)]
    return {"firstEntity": noHeader[0::2], "secondEntity": noHeader[1::2]}


# =============================================================================
# 3. Tokenisation / lemmatisation
# =============================================================================

# Shipped with the code (Data/ is not in git)
STOPWORDS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'polish.stopwords.txt')


def ensure_nltk():
    """Downloads the NLTK tokenizer models (punkt, punkt_tab) if they are missing. Runs once."""
    import nltk
    for res in ('punkt', 'punkt_tab'):
        try:
            nltk.data.find(f'tokenizers/{res}')
        except LookupError:
            nltk.download(res, quiet=True)
_STOPWORDS_CACHE = {}
PUNCT = {',', '.', ':', ';', '?', '!', ')', '(', '...', '„', '”', '—', '-'}


def load_stopwords(path=STOPWORDS_PATH):
    """Polish stopwords loaded from a file (cached after the first call)."""
    if path not in _STOPWORDS_CACHE:
        with open(path, encoding='utf-8') as f:
            _STOPWORDS_CACHE[path] = {l.strip().lower() for l in f if l.strip()}
    return _STOPWORDS_CACHE[path]


def preprocess(text, additonal_stopwords=None, stopwords_path=STOPWORDS_PATH):
    """
        Preprocessing: tokenizing words with NLTK and removing Polish stopwords
        (Polish stopwords are loaded from an external file),
        then filtering out non-word characters.
    """
    from nltk.tokenize import word_tokenize
    ensure_nltk()
    tokens = word_tokenize(text, language='polish')
    sw = set(load_stopwords(stopwords_path))
    if isinstance(additonal_stopwords, (list, set)):
        sw |= {w.lower() for w in additonal_stopwords}
    return [w for w in tokens if w.lower() not in sw and w.lower() not in PUNCT]


def dictStemmer(text, morf, names=NAMES, content=CONTENT_TAGS, norms=preNorm, lems=lems, stopwords=None):
    """
        This function lemmatizes the current text using Morfeusz:

    (NEW version from NEW_WORD_ANALYSIS)
    1. preprocess (stopwords, punctuation)
    2. norms: inflected forms of names -> nominative
    3. Morfeusz: keep only candidates whose tag is in `content`; prefer common (non-proper) words
    4. lems: fixes for wrong lemmas
    Returns a list of lemmas (str).

    Example:
        import morfeusz2; morf = morfeusz2.Morfeusz()
        dictStemmer("Karolina szła do pracowni", morf)
    """
    tokens = preprocess(text, additonal_stopwords=stopwords)
    if norms:
        tokens = [norms.get(t, t) for t in tokens]
    out = []
    for tok in tokens:
        cands = [a[2] for a in morf.analyse(tok if tok in names else tok.lower())]
        cont = [c for c in cands if c[2].split(":")[0] in content]
        if not cont:
            continue
        common = [c for c in cont if not c[3]]          # without a proper-name qualifier
        c = (common or cont)[0]
        if c is None or c[2] == "ign":
            continue
        lemma = c[1].split(":")[0].lower()
        if lems:
            lemma = lems.get(lemma, lemma)
        if stopwords and lemma in stopwords:
            continue
        out.append(lemma)
    return out


def spacyMorph(text, NLP, morf, polish_stopwords=None, verbosity=0):
    """
        This function uses spaCy- and Morfeusz-based lemmatisation to get
        syntactic and semantic features of the current text

    NLP = spacy.load("pl_core_news_sm"), morf = morfeusz2.Morfeusz()
    Returns: verb_phrases, noun_phrases, lemmas, sentences
    """
    def is_participle_adj(tok):
        return tok.pos_ == "ADJ" and tok.morph.get("VerbForm") == ["Part"]

    doc = NLP(text)
    noun_phrases, verb_phrases = [], []
    lemmas = dictStemmer(text, morf, norms=preNorm, lems=lems)
    sentences = [s.text for s in doc.sents]

    for token in doc:
        if verbosity:
            print(f"{token.text:15} | {token.lemma_:15} | {token.pos_} | {token.tag_} | {token.dep_}")

        # Noun phrase: [ADJ/NUM/DET/participle on the left] + NOUN + [simple PP on the right]
        if token.pos_ in ("NOUN", "PROPN"):
            left = [t.text for t in token.lefts if t.pos_ in ("ADJ", "NUM", "DET") or is_participle_adj(t)]
            right = []
            for t in token.rights:
                if t.pos_ == "ADP":
                    right += [t.text] + [ch.text for ch in t.rights if ch.pos_ in ("NOUN", "PROPN", "ADJ")]
            phrase = " ".join(left + [token.text] + right)
            if phrase.strip():
                noun_phrases.append(phrase)

        # Verb phrase: advmod + VERB + 'się' + obj/obl (with their dependents)
        if token.pos_ == "VERB":
            parts = [t.text for t in token.children if t.dep_ == "advmod"] + [token.text]
            parts += [t.text for t in token.children if t.dep_ == "expl"]
            for t in token.children:
                if t.dep_ in ("obj", "obl"):
                    parts += [t.text] + [ch.text for ch in t.children if ch.pos_ in ("NOUN", "PROPN", "ADJ")]
            phrase = " ".join(parts)
            if phrase.strip():
                verb_phrases.append(phrase)

        # Adjectival participle as a mini-verb ("wdowę wchodzącą z dziećmi")
        if is_participle_adj(token):
            head = token.head.text if token.head.pos_ in ("NOUN", "PROPN") else ""
            parts = [head, token.text]
            for t in token.children:
                if t.dep_ in ("obj", "obl"):
                    parts += [t.text] + [ch.text for ch in t.children if ch.pos_ in ("NOUN", "PROPN", "ADJ")]
            phrase = " ".join(p for p in parts if p)
            if phrase.strip():
                verb_phrases.append(phrase)

    return verb_phrases, noun_phrases, lemmas, sentences


# =============================================================================
# 4. Utterance features
# =============================================================================

def lexical_diversity(tokens):
    """
        Calculates Lexical Diversity as the number of unique tokens divided by the total number of tokens
    """
    return len(set(tokens)) / len(tokens) if len(tokens) > 0 else 0


def lexical_diversity_storywise(tokens, storyTokens):
    """
        Calculates Lexical Diversity relative to the story: the number of unique tokens of the utterance
        divided by the number of unique tokens of the story fragment
    """
    return len(set(tokens)) / len(set(storyTokens)) if len(tokens) > 0 else 0


PERSONAL_PRONOUNS = ["ja", "mnie", "mi", "mną", "moim", "mój", "moją", "moimi", "moja", "moje", "moich"]


def autobiographyAnalysis(transkrypcja, morf, pronouns=PERSONAL_PRONOUNS):
    """Autobiographical references in an utterance.

    personalVerbs: 1st-person singular verbs (Morfeusz tag contains 'sg:pri')
    personalPron : 1st-person personal/possessive pronouns (list `pronouns`)
    Typical measures: len(personalVerbs), len(personalPron), optionally normalised by utterance length.
    """
    personalVerbs = [a[2][0] for a in morf.analyse(transkrypcja) if "sg:pri" in a[2][2]]
    personalPron = [w for w in (x.strip("';:.,!?\"").lower() for x in transkrypcja.split()) if w in pronouns]
    return personalVerbs, personalPron


# =============================================================================
# 5. Semantic decoupling (similarity utterance <-> story fragment)
# =============================================================================

def bertSentenceCosineSim(text1, text2, model, aggregation="mean_pairs", mode=1):
    """
    Quick BERT cosine similarity, catered towards sentence embeddings, but it also works for lemmas
    (provided they have been properly filtered).

    It is not fancy at all.

    text1 : STORY items      (e.g. StorySentences / StoryLemmas)
    text2 : UTTERANCE items  (e.g. Sentences / Lemmas)
    mode  : 0 = SentenceTransformer (sentences)  -> model.encode
            1 = fastText / Word2Vec (gensim, single lemmas) -> model.wv[lemma]
            2 = GloVe / KeyedVectors (out-of-vocabulary lemmas are skipped)
    aggregation:
            'centroid'   : cos(mean utterance vector, mean story vector)
            'mean_pairs' : mean over all pairs
            'sym_max'    : symmetric best-match
            'max'        : for every utterance item the best match in the story,
                           identical tokens excluded (used as semDecoupling)
    Higher = closer to the story (lower = decoupling).
    """
    text1 = [t for t in text1 if str(t).strip()]
    text2 = [t for t in text2 if str(t).strip()]
    if mode == 2:                                                 # GloVe: skip out-of-vocabulary tokens
        text1 = [w for w in text1 if w in model]
        text2 = [w for w in text2 if w in model]
    if not text1 or not text2:
        return float("nan")

    if mode == 0:
        E1 = model.encode(text1, convert_to_numpy=True)
        E2 = model.encode(text2, convert_to_numpy=True)
    elif mode == 1:
        E1 = np.vstack([model.wv[w] for w in text1])
        E2 = np.vstack([model.wv[w] for w in text2])
    elif mode == 2:
        E1 = np.vstack([model[w] for w in text1])
        E2 = np.vstack([model[w] for w in text2])
    else:
        raise ValueError("mode must be 0, 1 or 2")

    unit = lambda X: X / np.linalg.norm(X, axis=-1, keepdims=True)
    F = unit(np.asarray(E1, dtype=np.float64))                    # story     [F, D]
    U = unit(np.asarray(E2, dtype=np.float64))                    # utterance [U, D]

    if aggregation == "centroid":
        return float(unit(U.mean(0)) @ unit(F.mean(0)))

    S = U @ F.T                                                    # [U, F] cosine matrix
    if aggregation == "mean_pairs":
        return float(S.mean())
    if aggregation == "sym_max":
        return float(0.5 * (S.max(1).mean() + S.max(0).mean()))
    if aggregation == "max":
        for i, u in enumerate(text2):                              # identical tokens do not count
            for j, f in enumerate(text1):
                if u == f:
                    S[i, j] = -np.inf
        row = S.max(1)
        row = row[np.isfinite(row)]
        return float(row.mean()) if row.size else float("nan")
    raise ValueError(f"unknown aggregation: {aggregation}")


def wmd_gensim(doc1_tokens, doc2_tokens, kv):
    """
    Word Mover's Distance between two documents (lists of lemmas/tokens).
    - Uses gensim's exact WMD (Earth Mover's Distance over word2vec space).
    - Duplicates matter (term frequency -> transport mass).
    - Unknown tokens are ignored.
    Returns:
      distance (float) — lower is more similar (0 = identical bags).
    """
    d1 = [t for t in doc1_tokens if t in kv]
    d2 = [t for t in doc2_tokens if t in kv]
    if not d1 or not d2:
        return float("nan")
    return kv.wmdistance(d1, d2)


# =============================================================================
# 6. TF-IDF (version from NEW_WORD_ANALYSIS)
# =============================================================================

def tf_idf_analysis(story_df, K, data_path=None, stopwords_add=None):
    """TF-IDF of lemmas for the 40 story fragments + top-K words for Janek, Karolina and the whole story.

    story_df: columns 'Fragment', 'fragment_hero' (or 'Hero'), 'StoryFrag'
    Returns: tfidf_long, Janek_top, Karolina_top, Story_top
    """
    import morfeusz2
    from sklearn.feature_extraction.text import TfidfVectorizer
    morf = morfeusz2.Morfeusz()

    story = story_df.drop_duplicates("Fragment").copy()
    if "fragment_hero" not in story and "Hero" in story:
        story["fragment_hero"] = story["Hero"]
    story["StoryLemmas"] = [dictStemmer(t, morf, stopwords=stopwords_add) for t in story["StoryFrag"]]

    vec = TfidfVectorizer(analyzer=lambda x: x)
    X = vec.fit_transform(story["StoryLemmas"])
    tfidf_long = (pd.DataFrame(X.toarray(), columns=vec.get_feature_names_out())
                  .assign(Fragment=story["Fragment"].values, fragment_hero=story["fragment_hero"].values)
                  .melt(id_vars=["Fragment", "fragment_hero"], var_name="lemma", value_name="tfidf")
                  .query("tfidf > 0"))
    if data_path:
        tfidf_long.to_excel(os.path.join(data_path, 'TFIDF_long.xlsx'), index=False)

    top = lambda d: d.groupby("lemma")["tfidf"].sum().nlargest(K)
    return (tfidf_long,
            top(tfidf_long[tfidf_long.fragment_hero == "JANEK"]),
            top(tfidf_long[tfidf_long.fragment_hero == "KAROLINA"]),
            top(tfidf_long))


# =============================================================================
# 7. Browsing utterances
# =============================================================================

def _as_list(x):
    """List-like columns (Lemmas/Sentences) are stored in the CSV as text "['a', 'b']"."""
    if isinstance(x, list):
        return x
    if isinstance(x, str) and x.startswith('['):
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return [x]
    return [] if pd.isna(x) else [str(x)]


def history_viewer(df, errors_dict=None,
                   extra_cols=('MW', 'MW_Estimate', 'Tracking', 'Preference')):
    """
    Interactive story/recall viewer in Jupyter.

    df: e.g. behavioral.csv -- columns Subject, Fragment (or fragment), Utterance,
        optionally StoryFrag, Lemmas, TrialNum and the columns in `extra_cols`.
    Shows: story fragment | participant's utterance | lemmas | trial metrics.
    Buttons: previous / next / mark error, plus a dropdown to pick a subject.
    errors_dict: optional pre-filled dictionary {Subject:{fragment: True/False}}
                 If None, a dictionary with all False is created.

    Returns: the updated errors_dict
    """
    import ipywidgets as widgets
    from IPython.display import display, HTML
    from html import escape

    df = df.rename(columns={'fragment': 'Fragment', 'utterance': 'Utterance'}).reset_index(drop=True)
    if errors_dict is None:
        errors_dict = {s: {f: False for f in g['Fragment']} for s, g in df.groupby('Subject')}

    state = {'rows': df, 'i': 0}
    # The content is NOT a widget: a plain display(HTML) with a display_id, replaced via .update().
    # Output/HTML widgets can render twice in VS Code.
    handle = None
    subj_dd = widgets.Dropdown(options=['(all)'] + sorted(df['Subject'].unique()), description='Subject:')

    def box(title, body, bg):
        return (f'<div style="margin-top:12px;padding:10px;background:{bg};color:#1a1a1a;border-radius:8px;">'
                f'<b>{title}</b><br>{body}</div>')

    def render():
        rows, i = state['rows'], state['i']
        if rows.empty:
            handle.update(HTML('<i>no rows</i>'))
            return
        r = rows.iloc[i]
        meta = ' | '.join(f'{c}: <b>{r[c]}</b>' for c in extra_cols if c in rows and pd.notna(r[c]))
        trial = f" | trial {r['TrialNum']}" if 'TrialNum' in rows else ''
        html = (f'<div style="max-width:950px;font-family:Arial;line-height:1.55;">'
                f'<h3>{r["Subject"]}{trial} | {r["Fragment"]}</h3><div>{meta}</div>')
        if 'StoryFrag' in rows and pd.notna(r['StoryFrag']):
            html += box('Story fragment:', escape(str(r['StoryFrag'])), '#eef3fb')
        html += box('Participant recall:', escape(str(r.get('Utterance', ''))), '#f5f5f5')
        if 'Lemmas' in rows:
            html += box('Lemmas:', escape(', '.join(map(str, _as_list(r['Lemmas'])))), '#fbf7ee')
        html += (f'<hr><div>Record {i + 1} of {len(rows)} | '
                 f'error marked: {errors_dict.get(r["Subject"], {}).get(r["Fragment"], False)}</div></div>')
        handle.update(HTML(html))

    def step(d):
        state['i'] = int(np.clip(state['i'] + d, 0, max(len(state['rows']) - 1, 0)))
        render()

    def on_subject(change):
        s = change['new']
        state['rows'] = df if s == '(all)' else df[df['Subject'] == s].reset_index(drop=True)
        state['i'] = 0
        render()

    def mark_error(_):
        r = state['rows'].iloc[state['i']]
        d = errors_dict.setdefault(r['Subject'], {})
        d[r['Fragment']] = not d.get(r['Fragment'], False)
        render()

    b_prev = widgets.Button(description='⬅ Previous'); b_prev.on_click(lambda _: step(-1))
    b_next = widgets.Button(description='Next ➡'); b_next.on_click(lambda _: step(+1))
    b_err = widgets.Button(description='Mark error', button_style='danger'); b_err.on_click(mark_error)
    subj_dd.observe(on_subject, names='value')

    display(widgets.HBox([subj_dd, b_prev, b_next, b_err]))
    handle = display(HTML(''), display_id=True)
    render()
    return errors_dict


# =============================================================================
# 8. Metrics report for a single utterance
# =============================================================================

def utterance_report(row, morf=None, sbert=None):
    """Prints the metrics of a single utterance (a row of behavioral.csv) and returns them as a dict.

    morf  = morfeusz2.Morfeusz()        -> autobiographical references  (None = skip)
    sbert = SentenceTransformer(...)    -> semantic coupling            (None = skip)
    """
    utt = str(row['Utterance'])
    lemmas = _as_list(row['Lemmas'])
    story_lemmas = _as_list(row['StoryLemmas'])
    n_words = len(utt.split())
    m = {}

    print('=' * 70)
    print(f"{row['Subject']} | trial {row['TrialNum']} | {row['Fragment']} | MW = {row['MW']}")
    print('=' * 70)
    print('RECALL:', utt)
    print('-' * 70)

    # 1) Length
    m['n_words'] = n_words
    m['n_lemmas'] = len(lemmas)
    print(f"Words / lemmas:            {n_words} / {len(lemmas)}")

    # 2) Lexical diversity
    m['lexdiv'] = lexical_diversity(lemmas)
    m['lexdiv_story'] = lexical_diversity_storywise(lemmas, story_lemmas)
    print(f"Lexical diversity:         {m['lexdiv']:.2f}   (unique lemmas / all lemmas; 1 = no repetitions)")
    print(f"Lexical div. vs story:     {m['lexdiv_story']:.2f}   (unique recall lemmas / unique story-fragment lemmas)")

    # 3) Autobiographical references
    if morf is not None:
        verbs, prons = autobiographyAnalysis(utt, morf)
        m['autobio'] = len(verbs) + len(prons)
        print(f"Self-references:           {m['autobio']}   (1st-person verbs: {verbs}, pronouns: {prons})")

    # 4) Semantic coupling (cosine similarity between utterance sentences and story-fragment sentences)
    if sbert is not None:
        m['cos_sim'] = bertSentenceCosineSim(_as_list(row['StorySentences']), _as_list(row['Sentences']),
                                             sbert, mode=0, aggregation='max')
        print(f"Cosine similarity (sBERT): {m['cos_sim']:.2f}   (higher = closer to the story, lower = decoupling)")

    print('=' * 70)
    return m
