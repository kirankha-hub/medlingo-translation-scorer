"""
MedLingo Translation Scorer — Streamlit app.

Upload a spreadsheet with the original script and the MedLingo output;
get corpus-level and per-sentence translation scores:
BLEU, chrF, TER (sacrebleu), semantic similarity (sentence embeddings,
per TextSim_MTQE) and COMET (Unbabel wmt22-comet-da).

Run locally:  streamlit run streamlit_app.py
"""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import sacrebleu
import streamlit as st

from file_loader import read_table

st.set_page_config(page_title="MedLingo Translation Scorer",
                   page_icon="🩺", layout="wide")


# ---------------------------------------------------------------- models

@st.cache_resource(show_spinner="Loading semantic similarity model…")
def embedder():
    from sentence_transformers import SentenceTransformer
    # Same model as https://github.com/fivehills/TextSim_MTQE
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


@st.cache_resource(show_spinner="Loading COMET model (large — first time takes a while)…")
def comet_model():
    from comet import download_model, load_from_checkpoint
    return load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))


# COMET (torch + a ~2.3 GB model) needs several GB of RAM. On a small host it
# would be killed by the OS out-of-memory reaper (a hard crash a try/except
# can't catch), so we check available memory first and skip COMET if it's low.
COMET_MIN_GB = 3.0


def available_ram_gb():
    """Best-effort available system RAM in GB, or None if it can't be read."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 * 1024)  # kB -> GB
    except Exception:
        pass
    return None  # e.g. macOS: no /proc — assume enough and let it run


# ---------------------------------------------------------------- scoring

def interpret(score):
    if score < 10: return "Almost no overlap"
    if score < 20: return "Low overlap"
    if score < 30: return "Gist preserved, heavily reworded"
    if score < 40: return "Moderate overlap"
    if score < 50: return "High overlap"
    if score < 60: return "Very high overlap"
    return "Near-identical"


def meaning_verdict(sim):
    if sim >= 0.75: return "Meaning preserved"
    if sim >= 0.55: return "Mostly preserved — review"
    return "Possible meaning change"


@st.cache_data(show_spinner=False)
def score_pairs(refs: tuple, cands: tuple, use_comet: bool):
    refs, cands = list(refs), list(cands)

    corpus = sacrebleu.corpus_bleu(cands, [refs])
    corpus_chrf = sacrebleu.corpus_chrf(cands, [refs])
    corpus_ter = sacrebleu.corpus_ter(cands, [refs])

    model = embedder()
    ref_emb = model.encode(refs, batch_size=64, show_progress_bar=False)
    cand_emb = model.encode(cands, batch_size=64, show_progress_bar=False)
    ref_emb = ref_emb / np.linalg.norm(ref_emb, axis=1, keepdims=True)
    cand_emb = cand_emb / np.linalg.norm(cand_emb, axis=1, keepdims=True)
    cosines = np.sum(ref_emb * cand_emb, axis=1).clip(-1, 1)

    comet_scores, comet_system, comet_error = None, None, None
    if use_comet:
        try:
            avail = available_ram_gb()
            if avail is not None and avail < COMET_MIN_GB:
                raise MemoryError(
                    f"Only {avail:.1f} GB RAM available; COMET needs about "
                    f"{COMET_MIN_GB:.0f} GB. Run locally or on a larger host "
                    "to enable COMET.")
            data = [{"src": r, "mt": c, "ref": r} for r, c in zip(refs, cands)]
            out = comet_model().predict(data, batch_size=8, gpus=0,
                                        num_workers=1, progress_bar=False)
            comet_scores = list(out.scores)
            comet_system = float(out.system_score)
        except Exception as e:
            # COMET is heavy (torch + a ~2 GB model) and can fail to load on
            # memory-limited hosts like Streamlit Community Cloud. Don't let
            # that take down the whole app — skip COMET and report why.
            comet_error = str(e)

    rows = []
    for i, (r, c) in enumerate(zip(refs, cands)):
        s = sacrebleu.sentence_bleu(c, [r], smooth_method="exp").score
        sim = float(cosines[i])
        row = {"#": i + 1, "Original script": r, "MedLingo output": c,
               "BLEU": round(s, 1), "Wording": interpret(s),
               "Semantic (%)": round(sim * 100),
               "Meaning": meaning_verdict(sim),
               "chrF": round(sacrebleu.sentence_chrf(c, [r]).score, 1),
               "TER": round(sacrebleu.sentence_ter(c, [r]).score, 1)}
        if comet_scores is not None:
            row["COMET"] = round(comet_scores[i] * 100)
        rows.append(row)

    summary = {"bleu": corpus.score, "bleu_label": interpret(corpus.score),
               "bp": corpus.bp, "precisions": list(corpus.precisions),
               "chrf": corpus_chrf.score, "ter": corpus_ter.score,
               "sem_mean": float(np.mean(cosines)),
               "sent_bleu_mean": float(np.mean([r["BLEU"] for r in rows])),
               "comet": comet_system, "comet_error": comet_error}
    return pd.DataFrame(rows), summary


def autodetect(cols):
    def find(keywords, exclude=None):
        for c in cols:
            if c == exclude:
                continue
            if any(k in str(c).lower() for k in keywords):
                return c
        return None
    ref = find(["original", "script", "reference", "source", "doctor"])
    cand = find(["medlingo", "output", "candidate", "translation", "patient"],
                exclude=ref)
    if ref is None or cand is None or ref == cand:
        ref, cand = cols[0], cols[1]
    return ref, cand


# ---------------------------------------------------------------- UI

st.title("🩺 MedLingo Translation Scorer")
st.caption("Upload a spreadsheet with the original script and the MedLingo output — "
           "get overall translation scores (BLEU, chrF, TER, semantic similarity, "
           "COMET) and a score for every sentence.")

uploaded = st.file_uploader("Excel or CSV with the two columns",
                            type=["xlsx", "xlsm", "xls", "csv", "tsv", "txt"])

# Default COMET on only where there's enough RAM for it (e.g. local machines);
# off on memory-limited hosts like Streamlit Community Cloud, where it can't run.
_ram = available_ram_gb()
_comet_ok = _ram is None or _ram >= COMET_MIN_GB
use_comet = st.toggle(
    "Include COMET (slower; needs ~3 GB RAM and a ~2.3 GB model)",
    value=_comet_ok,
    help=None if _comet_ok else
    f"Only ~{_ram:.1f} GB RAM here — COMET is skipped on this host. "
    "Run locally for COMET.")

if uploaded:
    try:
        # Detect the real format by content, not the extension, so a
        # spreadsheet mislabeled .csv (or a CSV named .xlsx) still loads.
        df = read_table(uploaded.getvalue(), uploaded.name)
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        st.stop()

    if len(df.columns) < 2:
        st.error("The file needs at least two columns "
                 "(original script and MedLingo output).")
        st.stop()

    cols = list(df.columns)
    ref_default, cand_default = autodetect(cols)
    c1, c2 = st.columns(2)
    ref_col = c1.selectbox("Original script (reference) column", cols,
                           index=cols.index(ref_default))
    cand_col = c2.selectbox("MedLingo output column", cols,
                            index=cols.index(cand_default))
    if ref_col == cand_col:
        st.error("Reference and MedLingo columns must be different.")
        st.stop()

    sub = df[[ref_col, cand_col]].dropna()
    refs = sub[ref_col].astype(str).str.strip()
    cands = sub[cand_col].astype(str).str.strip()
    mask = (refs != "") & (cands != "")
    refs, cands = refs[mask].tolist(), cands[mask].tolist()
    if not refs:
        st.error("No usable rows (empty cells were removed).")
        st.stop()

    with st.spinner(f"Scoring {len(refs)} sentences…"):
        table, s = score_pairs(tuple(refs), tuple(cands), use_comet)

    if use_comet and s.get("comet_error"):
        st.warning("COMET could not be computed, so it's omitted below. "
                   "All other scores are unaffected. Common causes: the host "
                   "ran out of memory loading the ~2 GB model, or a COMET "
                   "dependency failed to import.\n\n"
                   f"Details: {s['comet_error']}")

    # ---- headline scores, one row
    labels = ["Overall BLEU (corpus)", "Mean semantic similarity",
              "chrF (corpus)", "TER (corpus, lower = closer)"]
    values = [f"{s['bleu']:.1f}", f"{s['sem_mean'] * 100:.0f}%",
              f"{s['chrf']:.1f}", f"{s['ter']:.1f}"]
    helps = [s["bleu_label"], "meaning", "", ""]
    if s["comet"] is not None:
        labels.insert(2, "COMET (system, 0–100)")
        values.insert(2, f"{s['comet'] * 100:.0f}")
        helps.insert(2, "")
    for col, lab, val, hlp in zip(st.columns(len(labels)), labels, values, helps):
        col.metric(lab, val, help=hlp or None)

    with st.expander("More statistics"):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Sentences scored", len(refs))
        m2.metric("Mean sentence BLEU", f"{s['sent_bleu_mean']:.1f}")
        m3.metric("Brevity penalty", f"{s['bp']:.2f}")
        m4.metric("1–4-gram precision",
                  " / ".join(f"{p:.0f}" for p in s["precisions"]))

    # ---- filters (chips)
    st.subheader("Per-sentence scores")
    f1, f2 = st.columns(2)
    verdict_opts = ["All"] + [v for v in ["Possible meaning change",
                                          "Mostly preserved — review",
                                          "Meaning preserved"]
                              if v in set(table["Meaning"])]
    label_opts = ["All"] + [l for l in ["Almost no overlap", "Low overlap",
                                        "Gist preserved, heavily reworded",
                                        "Moderate overlap", "High overlap",
                                        "Very high overlap", "Near-identical"]
                            if l in set(table["Wording"])]
    pick_verdict = f1.radio("Filter by meaning", verdict_opts, horizontal=True)
    pick_label = f2.radio("Filter by wording overlap (BLEU)", label_opts,
                          horizontal=True)

    view = table
    if pick_verdict != "All":
        view = view[view["Meaning"] == pick_verdict]
    if pick_label != "All":
        view = view[view["Wording"] == pick_label]
    st.caption(f"{len(view)} of {len(table)} sentences shown")

    st.dataframe(view, use_container_width=True, hide_index=True, height=520)

    # ---- download
    buf = io.BytesIO()
    summary_df = pd.DataFrame({
        "Metric": ["Corpus BLEU", "Sentences scored", "Brevity penalty",
                   "1-gram precision", "2-gram precision", "3-gram precision",
                   "4-gram precision", "Mean sentence BLEU",
                   "Mean semantic similarity", "COMET system score",
                   "Corpus chrF", "Corpus TER"],
        "Value": [round(s["bleu"], 2), len(refs), round(s["bp"], 3),
                  *[round(p, 1) for p in s["precisions"]],
                  round(s["sent_bleu_mean"], 2), round(s["sem_mean"], 3),
                  round(s["comet"], 3) if s["comet"] is not None else "n/a",
                  round(s["chrf"], 2), round(s["ter"], 2)],
    })
    with pd.ExcelWriter(buf) as xl:
        table.to_excel(xl, sheet_name="Per-sentence scores", index=False)
        summary_df.to_excel(xl, sheet_name="Summary", index=False)
    st.download_button("⬇️ Download full results (.xlsx)", buf.getvalue(),
                       file_name="translation_scores.xlsx",
                       mime="application/vnd.openxmlformats-officedocument"
                            ".spreadsheetml.sheet")

    st.info("**Reading the scores:** BLEU, chrF and TER measure *surface* overlap — "
            "they penalize reworded text even when the rewording is a perfect "
            "simplification (TER: lower = closer, 0 = identical). Semantic "
            "similarity and COMET look past wording toward *meaning*: "
            "“myocardial infarction” → “heart attack” scores low on BLEU but high "
            "on both. The sweet spot for MedLingo: **high semantic/COMET + "
            "low/moderate BLEU** = meaning preserved, wording simplified. Rows "
            "flagged *Possible meaning change* deserve a manual read.")

# ---- legend & references (always visible)
st.divider()
st.subheader("Score legend & references")
st.markdown("""
| Score | Range | What it measures | Code | Publication |
|---|---|---|---|---|
| **BLEU** | 0–100, higher = more similar wording | Overlap of word sequences (1–4-gram precision) with the original, plus a brevity penalty. Standard MT metric, per the [Microsoft Translator methodology](https://learn.microsoft.com/azure/ai-services/translator/custom-translator/concepts/bleu-score). | [mjpost/sacrebleu](https://github.com/mjpost/sacrebleu); methodology: [MicrosoftDocs/azure-ai-docs](https://github.com/MicrosoftDocs/azure-ai-docs/blob/main/articles/ai-services/translator/custom-translator/concepts/bleu-score.md) | [Papineni et al. (2002)](https://aclanthology.org/P02-1040/), ACL; implementation: [Post (2018)](https://aclanthology.org/W18-6319/), WMT |
| **chrF** | 0–100, higher = more similar wording | Character n-gram F-score — like BLEU but at character level; more forgiving of small word changes and morphology. | [m-popovic/chrF](https://github.com/m-popovic/chrF) (computed via sacrebleu) | [Popović (2015)](https://aclanthology.org/W15-3049/), WMT |
| **TER** | 0–100+, **lower** = closer (0 = identical) | Translation Edit Rate: edits (insert/delete/substitute/shift) needed to turn the MedLingo output back into the original. | [mjpost/sacrebleu](https://github.com/mjpost/sacrebleu) | [Snover et al. (2006)](https://aclanthology.org/2006.amta-papers.25/), AMTA |
| **Semantic similarity** | 0–100%, higher = same meaning | Cosine similarity of sentence embeddings (paraphrase-multilingual-MiniLM-L12-v2); measures whether *meaning* is preserved regardless of wording. Drives the meaning verdicts (≥75% preserved, 55–75% review, <55% possible change). | [fivehills/TextSim_MTQE](https://github.com/fivehills/TextSim_MTQE) / [UKPLab/sentence-transformers](https://github.com/UKPLab/sentence-transformers) | [Reimers & Gurevych (2019)](https://aclanthology.org/D19-1410/), EMNLP |
| **COMET** | 0–100, higher = better quality | Neural metric (wmt22-comet-da) trained on human quality judgments of translations; sensitive to meaning errors rather than wording changes. | [Unbabel/COMET](https://github.com/Unbabel/COMET) | [Rei et al. (2020)](https://aclanthology.org/2020.emnlp-main.213/), EMNLP; model: [Rei et al. (2022)](https://aclanthology.org/2022.wmt-1.52/), WMT |
""")
