#!/usr/bin/env python3
"""
FM Literature Classifier — Final (Architecture-aware Improve)
Unsupervised, ChatGPT API + Embeddings, with Offline Fallback

What it does
------------
Marks a paper as `Valuable = True` only if BOTH:
  (A) FM-related (about / centrally relies on FMs), and
  (B) Improve-related: EITHER
        • classic FM method improvements (PEFT/LoRA, RLHF/DPO, distillation, quantization, decoding, etc.), OR
        • architectural upgrades layered around FMs (RAG, planners, agents, MoE, hierarchical controllers, tool-use, memory).

Run
---
pip install -U "openai>=1.40" pandas numpy openpyxl tenacity tqdm
Set OPENAI_API_KEY or fill API_KEY below. Put your Excel in the same folder and click ▶ in VS Code.
"""
from __future__ import annotations
import os, re, sys, json, math, hashlib, sqlite3, argparse
from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm

# === USER CONFIG ===
API_KEY = ""                                # e.g., "sk-..."; recommended: leave empty & use env var
INPUT_XLSX = "LR_Foundationmodel.xlsx"      # Excel file in the same folder
OUTPUT_XLSX = "classified_output.xlsx"
SHEET = 0                                   # sheet name or index
EMBED_MODEL = "text-embedding-3-large"      # or "text-embedding-3-small"
USE_LLM_JUDGE = False                       # True to enable LLM judge for borderline rows
JUDGE_MODEL = None                          # e.g., "gpt-5" when USE_LLM_JUDGE is True
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or None

# Offline fallback if quota is exhausted (cue-only classification still runs)
OFFLINE_FALLBACK = True
EMBED_BATCH = 32
EMBED_DIMS = {"text-embedding-3-large": 3072, "text-embedding-3-small": 1536}

try:
    from openai import OpenAI
except Exception:
    raise SystemExit("Please install the OpenAI Python SDK: pip install openai>=1.40")

# -----------------------------
# Concept probes & lexical cues
# -----------------------------
FOUNDATION_PROBES = [
    "foundation model", "large language model", "LLM", "multimodal foundation model",
    "vision-language model", "VLM", "base model", "general-purpose pretrained model",
    "self-supervised pretraining", "instruction tuning", "alignment tuning",
    "pretrained transformer", "scaling laws", "in-context learning",
    "GPT", "BERT", "T5", "LLaMA", "Mistral", "Phi", "Gemma", "CLIP", "DINOv2", "SAM (Segment Anything)", "Whisper",
]

# (1) Classic FM improvement anchors
IMPROVE_METHOD_ANCHORS = [
    r"\bPEFT\b", r"\bLoRA\b", r"\badapter(s)?\b", r"\bprefix[- ]tuning\b",
    r"\bRLHF\b", r"\bDPO\b", r"\bORPO\b", r"\bSFT\b", r"\binstruction[- ]tuning\b",
    r"\bdistillation\b", r"\bquantization\b", r"\bpruning\b",
    r"\bspeculative decoding\b", r"\bKV cache\b", r"\bmixture[- ]of[- ]experts\b|\bMoE\b",
    r"\bdata (curriculum|selection|filtering|augmentation)\b",
    r"\blayerwise (freezing|unfreezing)\b", r"\blow[- ]rank\b",
]

# (2) Architecture-layer anchors around FMs (what you asked to count as "Improve")
IMPROVE_ARCH_ANCHORS = [
    r"\bRAG\b|\bretrieval[- ]augmented\b",          # Retrieval-Augmented Generation
    r"\bplanner\b|\bplanning\b|\bhierarchical\b",   # planners / hierarchy
    r"\bagent(s)?\b|\bmulti[- ]agent\b|\bagentic\b",
    r"\btool[- ]use\b|\btool(s)?\b|\bAPI calling\b|\btoolformer\b",
    r"\bmemory\b|\bvector store\b|\bknowledge graph\b",
    r"\bcontroller\b|\bpolicy orchestration\b|\bcoordinator\b",
    r"\bchain[- ]of[- ]thought\b|\bCoT\b",
    r"\bMixture[- ]of[- ]Experts\b|\bMoE\b",
    r"\breplay buffer\b|\bcontinual learning\b",     # system learning loop around FM
    r"\bRAG[- ]based\b|\bRAG-enabled\b",
]

# Probes (semantic) for the improve dimension (tightened)
IMPROVE_PROBES = [
    "fine-tuning method", "continued pretraining", "instruction tuning method",
    "parameter-efficient fine-tuning (PEFT)", "LoRA", "prefix tuning",
    "alignment techniques", "RLHF", "DPO", "ORPO",
    "training efficiency", "inference optimization", "quantization", "pruning",
    "speculative decoding", "KV cache reuse", "mixture-of-experts (MoE)",
    "retrieval-augmented generation (RAG)", "hierarchical planner", "multi-agent LLM",
]

# Cues (lexical) for improve — no broad 'accuracy/performance/benchmark' words
IMPROVE_CUES = [
    r"\bfine[- ]tuning\b|\bfinetun(e|ing)\b",
    r"\btraining efficiency\b|\binference\b|\blatency\b|\bthroughput\b",
] + IMPROVE_METHOD_ANCHORS + IMPROVE_ARCH_ANCHORS

FOUNDATION_CUES = [
    r"\bfoundation model(s)?\b", r"\bLLM(s)?\b", r"\bGPT[-\s]?\d?\b",
    r"\bpretrain(ed|ing)?\b", r"\bself[- ]supervised\b", r"\btransformer\b",
    r"\bvision[- ]language\b", r"\bmultimodal\b", r"\bbase model\b",
    r"\bCLIP\b|\bDINOv2\b|\bSAM\b|\bWhisper\b|\bLLaMA\b|\bT5\b|\bBERT\b",
]

# FM centrality helpers (to avoid counting generic platforms as FM papers)
FM_ANCHORS = [
    r"\bfoundation model(s)?\b", r"\bLLM(s)?\b", r"\bGPT[-\s]?\d?\b", r"\bLLaMA\b",
    r"\bT5\b", r"\bBERT\b", r"\bMistral\b", r"\bPhi\b", r"\bGemma\b", r"\bCLIP\b",
    r"\bDINOv2\b", r"\bSAM\b", r"\bWhisper\b", r"\bvision[- ]language\b", r"\bmultimodal\b",
]
PLATFORM_BLOCKERS = [
    r"\bplatform\b", r"\btoolkit\b", r"\binterface\b", r"\bworkflow\b",
    r"\bframework\b", r"\bGUI\b", r"\benvironment\b", r"\bmodule\b",
]

# -----------------------------
# Utilities
# -----------------------------
def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def _count_hits(patterns: List[str], text: str) -> int:
    c = 0
    for pat in patterns:
        if re.search(pat, text, flags=re.IGNORECASE):
            c += 1
    return c

@dataclass
class Models:
    embed: str = "text-embedding-3-large"
    judge: Optional[str] = None  # e.g., "gpt-5"

class Cache:
    def __init__(self, path: str = "classifier_cache.sqlite"):
        self.path = path
        self._init_db()
    def _init_db(self):
        con = sqlite3.connect(self.path); cur = con.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS embeddings (hash TEXT PRIMARY KEY, model TEXT, vec BLOB)")
        cur.execute("CREATE TABLE IF NOT EXISTS judgments (hash TEXT PRIMARY KEY, model TEXT, json TEXT)")
        con.commit(); con.close()
    def get_embedding(self, key: str) -> Optional[np.ndarray]:
        con = sqlite3.connect(self.path); cur = con.cursor()
        cur.execute("SELECT vec FROM embeddings WHERE hash=?", (key,))
        row = cur.fetchone(); con.close()
        return None if row is None else np.frombuffer(row[0], dtype=np.float32)
    def set_embedding(self, key: str, model: str, vec: np.ndarray):
        con = sqlite3.connect(self.path); cur = con.cursor()
        cur.execute("REPLACE INTO embeddings(hash, model, vec) VALUES (?,?,?)",
                    (key, model, vec.astype(np.float32).tobytes()))
        con.commit(); con.close()
    def get_judgment(self, key: str) -> Optional[Dict]:
        con = sqlite3.connect(self.path); cur = con.cursor()
        cur.execute("SELECT json FROM judgments WHERE hash=?", (key,))
        row = cur.fetchone(); con.close()
        try:
            return None if row is None else json.loads(row[0])
        except Exception:
            return None
    def set_judgment(self, key: str, model: str, payload: Dict):
        con = sqlite3.connect(self.path); cur = con.cursor()
        cur.execute("REPLACE INTO judgments(hash, model, json) VALUES (?,?,?)",
                    (key, model, json.dumps(payload, ensure_ascii=False)))
        con.commit(); con.close()

class OpenAIClient:
    def __init__(self):
        api_key = (API_KEY or os.getenv("OPENAI_API_KEY"))
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set (fill API_KEY at top of file or set env var)")
        base_url = (OPENAI_BASE_URL or os.getenv("OPENAI_BASE_URL"))
        self.client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=30),
           retry=retry_if_exception_type(Exception))
    def embed(self, texts: List[str], model: str) -> List[List[float]]:
        outputs: List[List[float]] = []; BATCH = EMBED_BATCH
        for i in range(0, len(texts), BATCH):
            batch = texts[i:i + BATCH]
            resp = self.client.embeddings.create(model=model, input=batch)
            outputs.extend([d.embedding for d in resp.data])
        return outputs
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=30),
           retry=retry_if_exception_type(Exception))
    def judge(self, model: str, prompt: str, schema: Dict) -> Dict:
        resp = self.client.responses.create(
            model=model,
            input=[{"role":"system","content":"You are a precise academic classifier. Only output valid JSON matching the provided schema."},
                   {"role":"user","content":prompt}],
            response_format={"type":"json_schema","json_schema":{"name":"literature_classification","schema":schema,"strict":True}},
        )
        try:
            text = resp.output_text
        except Exception:
            if hasattr(resp, "output") and resp.output:
                parts = resp.output[0].content
                text = "".join(getattr(p, "text", "") for p in parts)
            else:
                text = "{}"
        try:
            return json.loads(text)
        except Exception:
            return {"error": "Invalid JSON from model", "raw": text}

# ---------------------------------
# Core classifier (unsupervised)
# ---------------------------------
class FMLiteratureClassifier:
    def __init__(self, models: Models, use_llm_judge: bool = False, cache_path: str = "classifier_cache.sqlite"):
        self.models = models
        self.use_llm_judge = use_llm_judge and (models.judge is not None)
        self.cache = Cache(cache_path)
        self.oai = OpenAIClient()
        self.fm_proto = None
        self.improve_proto = None
        self.embed_dim = EMBED_DIMS.get(models.embed, 3072)

    def _concat_fields(self, title: str, abstract: str, keywords: str) -> str:
        parts = [title or "", abstract or "", keywords or ""]
        text = "\n".join(p.strip() for p in parts if p and isinstance(p, str))
        return re.sub(r"\s+", " ", text)[:20000]

    def _embed_with_cache(self, text: str) -> np.ndarray:
        key = _hash(self.models.embed + "::" + text)
        cached = self.cache.get_embedding(key)
        if cached is not None:
            return cached
        try:
            vec = np.array(self.oai.embed([text], model=self.models.embed)[0], dtype=np.float32)
            self.cache.set_embedding(key, self.models.embed, vec)
            return vec
        except Exception as e:
            if OFFLINE_FALLBACK and ("insufficient_quota" in str(e) or "429" in str(e)):
                return np.zeros((self.embed_dim,), dtype=np.float32)
            raise

    def _proto(self, probes: List[str]) -> np.ndarray:
        joined = " | ".join(probes)
        key = _hash("PROTO::" + self.models.embed + "::" + joined)
        cached = self.cache.get_embedding(key)
        if cached is not None:
            return cached
        try:
            vecs = np.array(self.oai.embed(probes, model=self.models.embed), dtype=np.float32)
            centroid = vecs.mean(axis=0)
            self.cache.set_embedding(key, self.models.embed, centroid.astype(np.float32))
            return centroid
        except Exception as e:
            if OFFLINE_FALLBACK and ("insufficient_quota" in str(e) or "429" in str(e)):
                return np.zeros((self.embed_dim,), dtype=np.float32)
            raise

    def _cue_hits(self, cues: List[str], text: str) -> List[str]:
        hits = []
        for pat in cues:
            if re.search(pat, text, flags=re.IGNORECASE):
                hits.append(pat)
        return hits

    def _score(self, doc_vec: np.ndarray, proto_vec: np.ndarray, cue_hits: int) -> float:
        sim = (1 + _cos_sim(doc_vec, proto_vec)) / 2
        boost = 1 - math.exp(-0.6 * cue_hits)
        w_sim, w_boost = (0.8, 0.2)
        if OFFLINE_FALLBACK and (np.allclose(doc_vec, 0) or np.allclose(proto_vec, 0)):
            w_sim, w_boost = (0.4, 0.6)  # lean on cues when offline
        score = w_sim * sim + w_boost * boost
        return max(0.0, min(1.0, score))

    def _quantile_flags(self, scores: List[float], abs_thresh: float, q: float) -> List[bool]:
        if len(scores) == 0:
            return []
        qv = float(np.quantile(scores, q))
        thresh = max(abs_thresh, qv)
        return [s >= thresh for s in scores]

    def _llm_judge(self, row_text: str) -> Dict:
        schema = {
            "type": "object",
            "properties": {
                "foundation_model_related": {"type": "boolean"},
                "improves_model_performance": {"type": "boolean"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "rationale": {"type": "string"},
            },
            "required": ["foundation_model_related", "improves_model_performance"],
            "additionalProperties": False,
        }
        prompt = (
            "You are classifying an academic paper from title/abstract/keywords.\n"
            "Return JSON only.\n\n"
            "Definitions:\n"
            "- foundation_model_related: true if the work is ABOUT foundation models (e.g., LLMs/VLMs) — their training, fine-tuning, evaluation, alignment, scaling, safety, distillation, adapters, inference, data curation, architectures, or applications that centrally rely on them.\n"
            "- improves_model_performance: true if the paper proposes methods or evidence that improve model performance/efficiency/robustness OR introduces architectural layers around FMs (e.g., RAG, planners, agents, MoE, hierarchical controllers, tool-use, durable memory) that materially improve system capability.\n\n"
            f"Text:\n{row_text}\n"
        )
        return self.oai.judge(self.models.judge, prompt, schema)

    def fit_and_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        texts = [self._concat_fields(r.get("Title", ""), r.get("Abstract", ""), r.get("Keywords", "")) for _, r in df.iterrows()]

        # Prototypes
        self.fm_proto = self._proto(FOUNDATION_PROBES)
        self.improve_proto = self._proto(IMPROVE_PROBES)

        # Embeddings for documents
        doc_vecs = [self._embed_with_cache(t) for t in tqdm(texts, desc="Embedding docs")]  # type: ignore

        # Scores and lexical cues
        fm_scores, improve_scores = [], []
        fm_hits_list, improve_hits_list = [], []
        for text, vec in zip(texts, doc_vecs):
            fm_hits = self._cue_hits(FOUNDATION_CUES, text)
            im_hits = self._cue_hits(IMPROVE_CUES, text)
            fm_hits_list.append("; ".join(fm_hits))
            improve_hits_list.append("; ".join(im_hits))
            fm_scores.append(self._score(vec, self.fm_proto, len(fm_hits)))
            improve_scores.append(self._score(vec, self.improve_proto, len(im_hits)))

        # Dynamic thresholds: absolute + batch quantile (top 40%)
        fm_flags = self._quantile_flags(fm_scores, abs_thresh=0.35, q=0.60)
        improve_flags = self._quantile_flags(improve_scores, abs_thresh=0.35, q=0.60)

        # --- Post-hoc corrections: centrality & method/architecture requirement ---
        for i, text in enumerate(texts):
            title = (df.iloc[i].get("Title") or "")
            title_has_fm = _count_hits(FM_ANCHORS, title) > 0
            fm_anchor_hits = _count_hits(FM_ANCHORS, text)
            improve_method_hits = _count_hits(IMPROVE_METHOD_ANCHORS, text)
            improve_arch_hits = _count_hits(IMPROVE_ARCH_ANCHORS, text)
            platform_hits = _count_hits(PLATFORM_BLOCKERS, text)

            # Rule A: "Improve" now accepts classic methods OR architecture layers
            if improve_flags[i] and (improve_method_hits + improve_arch_hits == 0):
                improve_flags[i] = False

            # Rule B: FM centrality — demote platform/tool papers unless FM is central
            fm_central = title_has_fm or (fm_anchor_hits >= 2) or (improve_method_hits + improve_arch_hits > 0)
            if fm_flags[i] and (platform_hits > 0) and not fm_central:
                fm_flags[i] = False
        # -------------------------------------------------------------------------

        # Optional LLM judge for borderline cases
        rationales = [""] * len(df); used_llm = [False] * len(df)
        if self.use_llm_judge:
            fm_thresh = max(0.35, float(np.quantile(fm_scores, 0.60)))
            im_thresh = max(0.35, float(np.quantile(improve_scores, 0.60)))
            for i, text in enumerate(texts):
                borderline = (abs(fm_scores[i] - fm_thresh) <= 0.05) or (abs(improve_scores[i] - im_thresh) <= 0.05)
                if borderline:
                    used_llm[i] = True
                    jkey = _hash((self.models.judge or "") + "::" + text)
                    cached = self.cache.get_judgment(jkey)
                    result = cached if cached is not None else self._llm_judge(text)
                    if cached is None: self.cache.set_judgment(jkey, self.models.judge or "", result)
                    if isinstance(result, dict):
                        fm_flags[i] = bool(result.get("foundation_model_related", fm_flags[i]))
                        improve_flags[i] = bool(result.get("improves_model_performance", improve_flags[i]))
                        rat = result.get("rationale") or ""
                        if rat: rationales[i] = rat[:500]

        valuable = [f and g for f, g in zip(fm_flags, improve_flags)]

        out = df.copy()
        out["fm_score"] = np.round(fm_scores, 4)
        out["improve_score"] = np.round(improve_scores, 4)
        out["fm_flag"] = fm_flags
        out["improve_flag"] = improve_flags
        out["Valuable"] = valuable
        out["triggers_foundation"] = fm_hits_list
        out["triggers_improvement"] = improve_hits_list
        out["llm_judge_used"] = used_llm
        out["rationale"] = rationales
        return out

# -----------------------------
# CLI
# -----------------------------
def parse_args(argv: List[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Unsupervised FM Literature Classifier (ChatGPT API)")
    ap.add_argument("input", help="Input Excel (.xlsx) with Title, Abstract, Keywords, Link")
    ap.add_argument("-o", "--output", default="classified_output.xlsx", help="Output Excel path")
    ap.add_argument("--sheet", default=0, help="Sheet name or index")
    ap.add_argument("--embed-model", default="text-embedding-3-large", help="Embedding model name")
    ap.add_argument("--judge-model", default=None, help="LLM model for JSON judgments (e.g., gpt-5, gpt-4.1)")
    ap.add_argument("--use-llm-judge", action="store_true", help="Enable LLM judge for borderline cases")
    ap.add_argument("--cache", default="classifier_cache.sqlite", help="SQLite cache file path")
    return ap.parse_args(argv)

def main(argv: List[str]) -> int:
    if len(argv) == 0 and INPUT_XLSX:
        args = argparse.Namespace(
            input=INPUT_XLSX, output=OUTPUT_XLSX, sheet=SHEET,
            embed_model=EMBED_MODEL, judge_model=JUDGE_MODEL,
            use_llm_judge=bool(USE_LLM_JUDGE), cache="classifier_cache.sqlite",
        )
        print("Running with in-file config (no CLI args detected)...")
    else:
        args = parse_args(argv)

    try:
        df = pd.read_excel(args.input, sheet_name=args.sheet)
    except Exception as e:
        print(f"Failed to read Excel: {e}")
        return 2

    for col in ["Title", "Abstract", "Keywords", "Link"]:
        if col not in df.columns:
            print(f"Missing required column: {col}")
            return 2

    models = Models(embed=args.embed_model, judge=args.judge_model)
    clf = FMLiteratureClassifier(models=models, use_llm_judge=args.use_llm_judge, cache_path=args.cache)
    out = clf.fit_and_predict(df)

    try:
        out.to_excel(args.output, index=False)
        print(f"Saved -> {args.output}")
    except Exception as e:
        print(f"Failed to write output: {e}")
        return 2

    total = len(out); val_cnt = int(out["Valuable"].sum())
    print(f"Total: {total} | Valuable: {val_cnt} ({(val_cnt/total*100 if total else 0):.1f}%)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
