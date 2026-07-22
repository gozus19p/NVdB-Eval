"""
evaluate.py — Definition Generation Evaluation Pipeline
Evaluates LLM-generated word definitions against Senso Comune ground truth.

Usage:
    python evaluate.py --embedding-model multilingual-e5-base
    python evaluate.py --embedding-model all-MiniLM-L6-v2
    python evaluate.py --embedding-model Qwen3-Embedding-4B
    python evaluate.py  # runs all embedding models sequentially
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Protocol, runtime_checkable

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & paths (edit these to match your project layout)
# ---------------------------------------------------------------------------
_RESOURCES_PATH: Path = (Path(__file__).parent / "../../resources").resolve()
GT_BASE_PATH = (_RESOURCES_PATH / "sensocomune/sc_base_271217_no_ext").resolve()
PREDICTIONS_BASE_PATH = (
    _RESOURCES_PATH / "generated/DefinitionGeneratorBatch"
).resolve()
PREDICTIONS_V2_BASE_PATH = (
    _RESOURCES_PATH / "generated/DefinitionGeneratorV2Batch"
).resolve()
OUTPUT_BASE = _RESOURCES_PATH / "generated" / "evaluation-report"
TDM_SAMPLED_PATH = (_RESOURCES_PATH / "sensocomune" / "tdm.sampled.json").resolve()

# Ordered cluster labels (as they appear in tdm.sampled.json)
CLUSTERS: list[str] = ["FIRST", "SECOND", "THIRD", "FOURTH"]

LLM_MODELS: dict[str, str] = {
    "mistral-large-2512": "mistralai%2Fmistral-large-2512",
    "llama-4-scout": "meta-llama%2Fllama-4-scout",
    "qwen3.6-plus": "qwen%2Fqwen3.6-plus",
    "gemini-2.5-flash": "google%2Fgemini-2.5-flash",
    "deepseek-v3.2": "deepseek%2Fdeepseek-v3.2",
    "claude-sonnet-4.6": "anthropic%2Fclaude-sonnet-4.6",
}

# Similarity rescaling parameters (same as original notebook)
SIM_MIN = 0.7
SIM_MAX = 1.0

# Bootstrap configuration
N_BOOTSTRAP = 2000
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Embedding model registry
# ---------------------------------------------------------------------------
class EmbeddingModel(Enum):
    """Registry of supported sentence embedding models."""

    MULTILINGUAL_E5_BASE = "intfloat/multilingual-e5-base"
    ALL_MINILM_L6_V2 = "sentence-transformers/all-MiniLM-L6-v2"
    QWEN3_SEMANTIC_EMBEDDER = "Qwen/Qwen3-Embedding-4B"

    @classmethod
    def from_cli(cls, name: str) -> "EmbeddingModel":
        mapping = {
            "multilingual-e5-base": cls.MULTILINGUAL_E5_BASE,
            "all-MiniLM-L6-v2": cls.ALL_MINILM_L6_V2,
            "Qwen3-Embedding-4B": cls.QWEN3_SEMANTIC_EMBEDDER,
        }
        if name not in mapping:
            raise argparse.ArgumentTypeError(
                f"Unknown embedding model '{name}'. Choices: {list(mapping)}"
            )
        return mapping[name]

    @property
    def short_name(self) -> str:
        return self.value.split("/")[-1]

    def prefix_sentence(self, sentence: str) -> str:
        """Some models require a task prefix for better performance."""
        if self == EmbeddingModel.MULTILINGUAL_E5_BASE:
            return f"sentence: {sentence}"
        if self == EmbeddingModel.QWEN3_SEMANTIC_EMBEDDER:
            return f"Instruct: Given a gloss definition as query, retrieve relevant passages similar to that query.\nQuery: {sentence}"
        return sentence


# ---------------------------------------------------------------------------
# Embedding backend
# ---------------------------------------------------------------------------
@runtime_checkable
class EmbedderProtocol(Protocol):
    def encode(self, sentences: list[str]) -> torch.Tensor: ...


class HuggingFaceEmbedder:
    """Loads a HuggingFace model and encodes sentences into normalised embeddings."""

    def __init__(self, model_enum: EmbeddingModel) -> None:
        self.model_enum = model_enum
        log.info("Loading tokenizer/model: %s", model_enum.value)
        self.tokenizer = AutoTokenizer.from_pretrained(model_enum.value)
        self.model = AutoModel.from_pretrained(model_enum.value)
        self.model.eval()
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.model.to(self.device)
        log.info("Model loaded on device: %s", self.device)

    def _average_pool(
        self, last_hidden: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        if self.model_enum == EmbeddingModel.MULTILINGUAL_E5_BASE:
            last_hidden = last_hidden.masked_fill(~mask[..., None].bool(), 0.0)
            return last_hidden.sum(dim=1) / mask.sum(dim=1)[..., None]
        if self.model_enum == EmbeddingModel.QWEN3_SEMANTIC_EMBEDDER:
            left_padding = mask[:, -1].sum() == mask.shape[0]
            if left_padding:
                return last_hidden[:, -1]
            else:
                sequence_lengths = mask.sum(dim=1) - 1
                batch_size = last_hidden.shape[0]
                return last_hidden[
                    torch.arange(batch_size, device=last_hidden.device),
                    sequence_lengths,
                ]
        if self.model_enum == EmbeddingModel.ALL_MINILM_L6_V2:
            token_embeddings = last_hidden  # First element of model_output contains all token embeddings
            input_mask_expanded = (
                mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9
            )

    def encode(self, sentences: list[str]) -> torch.Tensor:
        prefixed = [self.model_enum.prefix_sentence(s) for s in sentences]
        batch = self.tokenizer(
            prefixed,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = self.model(**batch)
        embeddings = self._average_pool(
            outputs.last_hidden_state, batch["attention_mask"]
        )
        return F.normalize(embeddings, p=2, dim=1).cpu()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
class GroundTruthLoader:
    """Loads and normalises Senso Comune ground truth into a lookup DataFrame."""

    def __init__(self, base_path: Path) -> None:
        self.base_path = base_path

    def load(self) -> pd.DataFrame:
        json_files = list(self.base_path.glob("*.json"))
        log.info("Found %d ground-truth JSON files", len(json_files))
        raw = [
            lemma["lemma"]
            for jf in json_files
            for lemma in json.loads(jf.read_text(encoding="utf-8"))["lemmas"]
        ]

        df = pd.DataFrame(raw).explode("meanings").rename(columns={"id": "lemma_id"})
        df = pd.concat(
            [df.drop(columns="meanings"), df["meanings"].apply(pd.Series)], axis=1
        )
        # Ground-truth lemma ids are formatted like "a#2"; predictions use plain lemma.
        df["lemma"] = df["lemma_id"].astype(str).str.split("#", n=1).str[0]
        df = df[["lemma_id", "lemma", "gloss"]].drop_duplicates(
            subset=["lemma", "gloss"]
        )
        df = (
            df.groupby("lemma")
            .agg(id=("lemma_id", "first"), glosses=("gloss", list))
            .dropna()
            .reset_index()
        )
        log.info("Ground truth: %d unique lemmas", len(df))
        return df


class PredictionLoader:
    """Loads LLM prediction JSONs for a single LLM model."""

    def __init__(self, base_path: Path) -> None:
        self.base_path = base_path

    def load(self, llm_name: str, subdir: str) -> pd.DataFrame:
        json_files = list((self.base_path / subdir).glob("*.json"))
        log.info("Loading %d prediction files for %s", len(json_files), llm_name)
        records = []
        for jf in json_files:
            data = json.loads(jf.read_text())
            data["hash_key"] = jf.stem.removesuffix(".json")
            data["model_name"] = llm_name
            records.append(data)
        return pd.DataFrame(records)


class ClusterLoader:
    """Loads tdm.sampled.json and provides a lemma → cluster lookup."""

    def __init__(self, tdm_path: Path) -> None:
        self.tdm_path = tdm_path
        self._lookup: dict[str, str] | None = None

    def load(self) -> dict[str, str]:
        if self._lookup is not None:
            return self._lookup
        if not self.tdm_path.exists():
            log.warning(
                "TDM sampled file not found at %s — cluster metrics will be skipped.",
                self.tdm_path,
            )
            return {}
        raw: list[dict] = json.loads(self.tdm_path.read_text(encoding="utf-8"))
        self._lookup = {item["lemma"]: item["cluster"] for item in raw}
        log.info(
            "Loaded cluster info for %d lemmas from tdm.sampled.json", len(self._lookup)
        )
        return self._lookup


# ---------------------------------------------------------------------------
# Evaluation cache
# ---------------------------------------------------------------------------
class EvaluationCache:
    """Saves/loads per-LLM eval DataFrames as Parquet to avoid recomputation."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, llm_name: str) -> Path:
        safe = llm_name.replace("/", "_").replace(" ", "_")
        return self.cache_dir / f"cache_{safe}.parquet"

    def get(self, llm_name: str) -> pd.DataFrame | None:
        p = self._path(llm_name)
        if p.exists():
            log.info("Cache HIT for %s — loading from %s", llm_name, p)
            return pd.read_parquet(p)
        return None

    def set(self, llm_name: str, df: pd.DataFrame) -> None:
        p = self._path(llm_name)
        df.to_parquet(p, index=False)
        log.info("Cache written: %s", p)


# ---------------------------------------------------------------------------
# Similarity evaluation
# ---------------------------------------------------------------------------
@dataclass
class LemmaResult:
    lemma: str
    similarity: float  # recall-weighted optimal assignment score
    matched_similarity: float  # mean similarity for matched pairs only
    false_negatives: int
    false_positives: int
    false_positives_instances: list[dict]


class SimilarityEvaluator:
    """
    Evaluates predicted senses against ground-truth glosses using the
    Hungarian (optimal assignment) algorithm on embedding similarities.
    """

    def __init__(self, embedder: HuggingFaceEmbedder) -> None:
        self.embedder = embedder

    def _rescale(self, raw: float) -> float:
        """Rescale cosine similarity from [SIM_MIN, SIM_MAX] to [0, 1]."""
        if self.embedder.model_enum == EmbeddingModel.MULTILINGUAL_E5_BASE:
            return max(0.0, min(1.0, (raw - SIM_MIN) / (SIM_MAX - SIM_MIN)))
        return raw  # No rescaling for other models (already in [0, 1])

    def _similarity_matrix(
        self, predictions: list[str], glosses: list[str]
    ) -> list[list[float]]:
        all_sents = predictions + glosses
        embeddings = self.embedder.encode(all_sents)
        pred_emb = embeddings[: len(predictions)]
        gloss_emb = embeddings[len(predictions) :]
        raw_matrix = (pred_emb @ gloss_emb.T).tolist()
        return [[self._rescale(v) for v in row] for row in raw_matrix]

    def evaluate_lemma(
        self, predicted_senses: list[str], glosses: list[str], lemma: str
    ) -> LemmaResult:
        if (
            not predicted_senses
            or not glosses
            or not all(isinstance(s, str) for s in predicted_senses + glosses)
            or not all(s.strip() for s in predicted_senses + glosses)
            or not lemma
            or not lemma.strip()
        ):
            return LemmaResult(
                lemma=lemma,
                similarity=0.0,
                matched_similarity=0.0,
                false_negatives=len(glosses),
                false_positives=len(predicted_senses),
                false_positives_instances=[
                    {"definition": s, "index": i}
                    for i, s in enumerate(predicted_senses)
                ],
            )

        matrix = self._similarity_matrix(predicted_senses, glosses)
        row_ind, col_ind = linear_sum_assignment(matrix, maximize=True)
        matched_sims = [matrix[i][j] for i, j in zip(row_ind, col_ind)]

        unmatched_gt = len(glosses) - len(row_ind)
        all_sims = matched_sims + [0.0] * unmatched_gt
        recall_score = sum(all_sims) / len(glosses)
        matched_mean = float(np.mean(matched_sims)) if matched_sims else 0.0
        # Get unmatched predictions instances for error analysis
        matched_pred_indices = set(row_ind)
        false_positives_instances = [
            {"definition": predicted_senses[i], "index": i}
            for i in range(len(predicted_senses))
            if i not in matched_pred_indices
        ]

        return LemmaResult(
            lemma=lemma,
            similarity=recall_score,
            matched_similarity=matched_mean,
            false_negatives=unmatched_gt,
            false_positives=max(0, len(predicted_senses) - len(glosses)),
            false_positives_instances=false_positives_instances,
        )

    def evaluate_dataframe(
        self, pred_df: pd.DataFrame, gt_df: pd.DataFrame
    ) -> pd.DataFrame:
        gt_lookup: dict[str, list[str]] = dict(zip(gt_df["lemma"], gt_df["glosses"]))
        results: list[LemmaResult] = []
        for _, row in tqdm(pred_df.iterrows(), total=len(pred_df), desc="Evaluating"):
            lemma: str = row.get("lemma", "")
            if lemma not in gt_lookup:
                continue
            glosses = gt_lookup[lemma]
            predicted = [s.get("definition", "") for s in row.get("senses", [])]
            results.append(self.evaluate_lemma(predicted, glosses, lemma))

        return pd.DataFrame(
            [
                {
                    "lemma": r.lemma,
                    "similarity": r.similarity,
                    "matched_similarity": r.matched_similarity,
                    "false_negatives": r.false_negatives,
                    "false_positives": r.false_positives,
                    "false_positives_instances": r.false_positives_instances,
                }
                for r in results
            ]
        )


# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------
@dataclass
class ModelStats:
    model_name: str
    n: int
    mean_similarity: float
    mean_matched_similarity: float
    ci_low: float
    ci_high: float
    ci_matched_low: float
    ci_matched_high: float
    wilcoxon_stat: float
    wilcoxon_p: float
    mean_false_negatives: float
    mean_false_positives: float


@dataclass
class CrossModelStats:
    friedman_stat: float
    friedman_p: float
    pairwise: list[dict]  # model_a, model_b, wilcoxon_stat, p_raw, p_corrected


@dataclass
class ClusterStats:
    """Per-cluster aggregated metrics across all LLMs."""

    cluster: str
    model_name: str
    n: int
    mean_similarity: float
    mean_matched_similarity: float
    mean_false_negatives: float
    mean_false_positives: float


class StatisticalAnalyser:
    """Computes per-model and cross-model statistical tests."""

    def __init__(self, chance_baseline: float = 0.0) -> None:
        self.chance_baseline = chance_baseline

    def _bootstrap_ci(
        self, values: np.ndarray, n: int = N_BOOTSTRAP, seed: int = RANDOM_SEED
    ) -> tuple[float, float]:
        rng = np.random.default_rng(seed)
        means = [rng.choice(values, len(values), replace=True).mean() for _ in range(n)]
        return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

    def per_model(self, name: str, df: pd.DataFrame) -> ModelStats:
        sims = df["similarity"].dropna().values
        matched = df["matched_similarity"].dropna().values

        ci_low, ci_high = self._bootstrap_ci(sims)
        ci_m_low, ci_m_high = self._bootstrap_ci(matched)

        # Wilcoxon signed-rank: H0 = median equals chance baseline
        # Shift values so H0 median = 0
        shifted = sims - self.chance_baseline
        try:
            w_stat, w_p = wilcoxon(shifted, alternative="greater")
        except ValueError:
            w_stat, w_p = float("nan"), float("nan")

        return ModelStats(
            model_name=name,
            n=len(sims),
            mean_similarity=float(sims.mean()),
            mean_matched_similarity=float(matched.mean()),
            ci_low=ci_low,
            ci_high=ci_high,
            ci_matched_low=ci_m_low,
            ci_matched_high=ci_m_high,
            wilcoxon_stat=float(w_stat),
            wilcoxon_p=float(w_p),
            mean_false_negatives=float(df["false_negatives"].mean()),
            mean_false_positives=float(df["false_positives"].mean()),
        )

    def per_cluster(
        self,
        results: dict[str, pd.DataFrame],
        cluster_lookup: dict[str, str],
    ) -> list[ClusterStats]:
        """Compute per-cluster mean metrics for each LLM."""
        cluster_stats: list[ClusterStats] = []
        for llm_name, df in results.items():
            df = df.copy()
            df["cluster"] = df["lemma"].map(cluster_lookup)
            df = df.dropna(subset=["cluster"])
            for cluster in CLUSTERS:
                sub = df[df["cluster"] == cluster]
                if sub.empty:
                    continue
                cluster_stats.append(
                    ClusterStats(
                        cluster=cluster,
                        model_name=llm_name,
                        n=len(sub),
                        mean_similarity=float(sub["similarity"].mean()),
                        mean_matched_similarity=float(sub["matched_similarity"].mean()),
                        mean_false_negatives=float(sub["false_negatives"].mean()),
                        mean_false_positives=float(sub["false_positives"].mean()),
                    )
                )
        return cluster_stats

    def cross_model(self, results: dict[str, pd.DataFrame]) -> CrossModelStats:
        names = list(results.keys())
        # Align on common lemmas
        common = set.intersection(*[set(df["lemma"]) for df in results.values()])
        aligned = {
            n: results[n].set_index("lemma").loc[list(common)]["similarity"].values
            for n in names
        }

        f_stat, f_p = friedmanchisquare(*aligned.values())

        # Pairwise post-hoc Wilcoxon with Bonferroni correction
        pairs = [
            (names[i], names[j])
            for i in range(len(names))
            for j in range(i + 1, len(names))
        ]
        raw_ps, stats = [], []
        for a, b in pairs:
            try:
                s, p = wilcoxon(aligned[a], aligned[b])
            except ValueError:
                s, p = float("nan"), float("nan")
            stats.append(s)
            raw_ps.append(p)

        _, corrected_ps, _, _ = multipletests(raw_ps, method="bonferroni")
        pairwise = [
            {
                "model_a": a,
                "model_b": b,
                "wilcoxon_stat": float(s),
                "p_raw": float(p),
                "p_bonferroni": float(pc),
            }
            for (a, b), s, p, pc in zip(pairs, stats, raw_ps, corrected_ps)
        ]

        return CrossModelStats(
            friedman_stat=float(f_stat),
            friedman_p=float(f_p),
            pairwise=pairwise,
        )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
PALETTE = {
    "mistral-large-2512": "#4C72B0",
    "llama-4-scout": "#DD8452",
    "qwen3.6-plus": "#55A868",
    "gemini-2.5-flash": "#83111e",
    "deepseek-v3.2": "#2200aa",
    "claude-sonnet-4.6": "#aaeeee",
}


class Plotter:
    """Produces publication-quality figures for a CLiC-it paper."""

    def __init__(self, output_dir: Path, embedding_model_name: str = "") -> None:
        self.output_dir = output_dir
        self.embedding_model_name = embedding_model_name
        self._emb_label = f"[{embedding_model_name}]" if embedding_model_name else ""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)
        plt.rcParams.update({"figure.dpi": 200, "savefig.bbox": "tight"})

    # --- Figure 1: similarity distributions (KDE + rug) ---
    def plot_similarity_distributions(self, results: dict[str, pd.DataFrame]) -> None:
        items = list(results.items())
        row1, row2 = items[:3], items[3:]

        fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=False)

        for col, (name, df) in enumerate(row1):
            ax = axes[0, col]
            color = PALETTE.get(name, "#888888")
            sns.kdeplot(
                df["similarity"], ax=ax, color=color, fill=True, alpha=0.35, linewidth=2
            )
            sns.rugplot(df["similarity"], ax=ax, color=color, alpha=0.3, height=0.04)
            ax.axvline(
                df["similarity"].mean(),
                color=color,
                linestyle="--",
                linewidth=1.5,
                label=f"mean={df['similarity'].mean():.3f}",
            )
            ax.set_xlim(0, 1)
            ax.set_xlabel("Similarity score")
            ax.set_ylabel("Density")
            ax.set_title(name, fontsize=11, fontweight="bold")
            ax.legend(fontsize=9)

        for col, (name, df) in enumerate(row2):
            ax = axes[1, col]
            color = PALETTE.get(name, "#888888")
            sns.kdeplot(
                df["similarity"], ax=ax, color=color, fill=True, alpha=0.35, linewidth=2
            )
            sns.rugplot(df["similarity"], ax=ax, color=color, alpha=0.3, height=0.04)
            ax.axvline(
                df["similarity"].mean(),
                color=color,
                linestyle="--",
                linewidth=1.5,
                label=f"mean={df['similarity'].mean():.3f}",
            )
            ax.set_xlim(0, 1)
            ax.set_xlabel("Similarity score")
            ax.set_ylabel("Density")
            ax.set_title(name, fontsize=11, fontweight="bold")
            ax.legend(fontsize=9)

        # Hide the unused third subplot in the second row
        # axes[1, 2].set_visible(False)

        fig.suptitle(
            f"Similarity Score Distributions per LLM {self._emb_label}",
            fontsize=13,
            fontweight="bold",
        )
        plt.tight_layout()
        fig.savefig(self.output_dir / "fig1_similarity_distributions.png")
        plt.close(fig)
        log.info("Saved fig1_similarity_distributions")

    # --- Figure 2: mean similarity with bootstrapped CI (grouped bar) ---
    def plot_mean_similarity_comparison(self, stats: list[ModelStats]) -> None:
        names = [s.model_name for s in stats]
        means = [s.mean_similarity for s in stats]
        means_m = [s.mean_matched_similarity for s in stats]
        errs_low = [s.mean_similarity - s.ci_low for s in stats]
        errs_high = [s.ci_high - s.mean_similarity for s in stats]
        errs_m_low = [s.mean_matched_similarity - s.ci_matched_low for s in stats]
        errs_m_high = [s.ci_matched_high - s.mean_matched_similarity for s in stats]

        x = np.arange(len(names))
        width = 0.35
        colors = [PALETTE.get(n, "#888") for n in names]

        fig, ax = plt.subplots(figsize=(7, 4.5))
        bars1 = ax.bar(
            x - width / 2,
            means,
            width,
            color=colors,
            alpha=0.85,
            yerr=[errs_low, errs_high],
            capsize=5,
            label="All records",
        )
        bars2 = ax.bar(
            x + width / 2,
            means_m,
            width,
            color=colors,
            alpha=0.45,
            yerr=[errs_m_low, errs_m_high],
            capsize=5,
            label="Matched only",
            hatch="//",
            edgecolor="grey",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha="right")
        ax.set_ylabel("Mean similarity ± 95% CI")
        ax.set_ylim(0, 1)
        ax.set_title(
            f"Mean Similarity by LLM (bootstrapped 95% CI) {self._emb_label}",
            fontweight="bold",
        )
        ax.legend()
        plt.tight_layout()
        fig.savefig(self.output_dir / "fig2_mean_similarity_comparison.png")
        plt.close(fig)
        log.info("Saved fig2_mean_similarity_comparison")

    # --- Figure 3: FP / FN stacked bar ---
    def plot_fp_fn(self, stats: list[ModelStats]) -> None:
        names = [s.model_name for s in stats]
        fp = [s.mean_false_positives for s in stats]
        fn = [s.mean_false_negatives for s in stats]
        x = np.arange(len(names))
        width = 0.5
        colors = [PALETTE.get(n, "#888") for n in names]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(
            x,
            fn,
            width,
            label="False Negatives (missed senses)",
            color=colors,
            alpha=0.85,
        )
        ax.bar(
            x,
            fp,
            width,
            bottom=fn,
            label="False Positives (extra senses)",
            color=colors,
            alpha=0.4,
            hatch="xx",
            edgecolor="grey",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha="right")
        ax.set_ylabel("Mean count per lemma")
        ax.set_title(
            f"Mean False Positives / Negatives per LLM {self._emb_label}",
            fontweight="bold",
        )
        ax.legend()
        plt.tight_layout()
        fig.savefig(self.output_dir / "fig3_fp_fn.png")
        plt.close(fig)
        log.info("Saved fig3_fp_fn")

    # --- Figure 4: violin plot — all vs matched similarity ---
    def plot_violin(self, results: dict[str, pd.DataFrame]) -> None:
        records = []
        for name, df in results.items():
            for _, row in df.iterrows():
                records.append(
                    {
                        "model": name,
                        "type": "All records",
                        "similarity": row["similarity"],
                    }
                )
                records.append(
                    {
                        "model": name,
                        "type": "Matched only",
                        "similarity": row["matched_similarity"],
                    }
                )
        plot_df = pd.DataFrame(records)

        fig, ax = plt.subplots(figsize=(9, 5))
        sns.violinplot(
            data=plot_df,
            x="model",
            y="similarity",
            hue="type",
            split=True,
            inner="quart",
            palette=["#4C72B0", "#DD8452"],
            ax=ax,
            cut=0,
        )
        ax.set_xlabel("")
        ax.set_ylabel("Similarity")
        ax.set_title(
            f"Distribution of Similarities: All vs. Matched Records {self._emb_label}",
            fontweight="bold",
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
        plt.tight_layout()
        fig.savefig(self.output_dir / "fig4_violin_all_vs_matched.png")
        plt.close(fig)
        log.info("Saved fig4_violin_all_vs_matched")

    # --- Figure 5: cross-model pairwise p-value heatmap ---
    def plot_pairwise_heatmap(
        self, cross: CrossModelStats, model_names: list[str]
    ) -> None:
        n = len(model_names)
        matrix = np.ones((n, n))
        idx = {name: i for i, name in enumerate(model_names)}
        for pw in cross.pairwise:
            i, j = idx[pw["model_a"]], idx[pw["model_b"]]
            matrix[i, j] = pw["p_bonferroni"]
            matrix[j, i] = pw["p_bonferroni"]

        fig, ax = plt.subplots(figsize=(5, 4))
        mask = np.eye(n, dtype=bool)
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".3f",
            mask=mask,
            xticklabels=model_names,
            yticklabels=model_names,
            cmap="coolwarm_r",
            vmin=0,
            vmax=0.1,
            ax=ax,
            linewidths=0.5,
            cbar_kws={"label": "Bonferroni-corrected p"},
        )
        ax.set_title(
            f"Pairwise Wilcoxon p-values (Bonferroni) {self._emb_label}\nFriedman: χ²={cross.friedman_stat:.2f}, p={cross.friedman_p:.4f}",
            fontweight="bold",
            fontsize=10,
        )
        plt.tight_layout()
        fig.savefig(self.output_dir / "fig5_pairwise_heatmap.png")
        plt.close(fig)
        log.info("Saved fig5_pairwise_heatmap")

    # --- Figure 6: per-cluster mean similarity grouped bar ---
    def plot_cluster_metrics(self, cluster_stats: list[ClusterStats]) -> None:
        if not cluster_stats:
            log.warning("No cluster stats to plot — skipping fig6.")
            return

        df = pd.DataFrame(
            [
                {
                    "cluster": cs.cluster,
                    "model": cs.model_name,
                    "mean_similarity": cs.mean_similarity,
                    "mean_matched_similarity": cs.mean_matched_similarity,
                }
                for cs in cluster_stats
            ]
        )

        clusters = [c for c in CLUSTERS if c in df["cluster"].unique()]
        models = list(df["model"].unique())
        x = np.arange(len(clusters))
        n_models = len(models)
        width = 0.8 / n_models

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        for ax, metric, label in [
            (axes[0], "mean_similarity", "All records"),
            (axes[1], "mean_matched_similarity", "Matched only"),
        ]:
            for k, model in enumerate(models):
                sub = df[df["model"] == model].set_index("cluster")
                vals = [sub.loc[c, metric] if c in sub.index else 0.0 for c in clusters]
                offset = (k - n_models / 2 + 0.5) * width
                ax.bar(
                    x + offset,
                    vals,
                    width,
                    label=model,
                    color=PALETTE.get(model, "#888888"),
                    alpha=0.85,
                )
            ax.set_xticks(x)
            ax.set_xticklabels(clusters, rotation=15, ha="right")
            ax.set_ylabel("Mean similarity")
            ax.set_ylim(0, 1)
            ax.set_title(f"{label} {self._emb_label}", fontweight="bold")
            ax.legend(fontsize=8)

        fig.suptitle(
            f"Mean Similarity by Cluster & LLM {self._emb_label}",
            fontsize=13,
            fontweight="bold",
        )
        plt.tight_layout()
        fig.savefig(self.output_dir / "fig6_cluster_similarity.png")
        plt.close(fig)
        log.info("Saved fig6_cluster_similarity")

    # --- Figure 7: per-cluster FP/FN grouped bar ---
    def plot_cluster_fp_fn(self, cluster_stats: list[ClusterStats]) -> None:
        if not cluster_stats:
            log.warning("No cluster stats to plot — skipping fig7.")
            return

        df = pd.DataFrame(
            [
                {
                    "cluster": cs.cluster,
                    "model": cs.model_name,
                    "fn": cs.mean_false_negatives,
                    "fp": cs.mean_false_positives,
                }
                for cs in cluster_stats
            ]
        )

        clusters = [c for c in CLUSTERS if c in df["cluster"].unique()]
        models = list(df["model"].unique())
        x = np.arange(len(clusters))
        n_models = len(models)
        width = 0.8 / n_models

        fig, ax = plt.subplots(figsize=(10, 5))
        for k, model in enumerate(models):
            sub = df[df["model"] == model].set_index("cluster")
            fn_vals = [sub.loc[c, "fn"] if c in sub.index else 0.0 for c in clusters]
            fp_vals = [sub.loc[c, "fp"] if c in sub.index else 0.0 for c in clusters]
            offset = (k - n_models / 2 + 0.5) * width
            color = PALETTE.get(model, "#888888")
            ax.bar(
                x + offset, fn_vals, width, color=color, alpha=0.85, label=f"{model} FN"
            )
            ax.bar(
                x + offset,
                fp_vals,
                width,
                bottom=fn_vals,
                color=color,
                alpha=0.4,
                hatch="xx",
                edgecolor="grey",
                label=f"{model} FP",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(clusters, rotation=15, ha="right")
        ax.set_ylabel("Mean count per lemma")
        ax.set_title(
            f"Mean FP / FN by Cluster & LLM {self._emb_label}", fontweight="bold"
        )
        ax.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        fig.savefig(self.output_dir / "fig7_cluster_fp_fn.png")
        plt.close(fig)
        log.info("Saved fig7_cluster_fp_fn")


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------
class ReportWriter:
    """Serialises metrics and stats to the evaluation-report directory."""

    def __init__(
        self, base_output: Path, embedding_model: EmbeddingModel, batch_type: str
    ) -> None:
        self.root = base_output / batch_type / embedding_model.short_name
        self.root.mkdir(parents=True, exist_ok=True)

    def write_model_results(
        self, llm_name: str, eval_df: pd.DataFrame, stats: ModelStats
    ) -> None:
        out = {
            "model": llm_name,
            "aggregate_metrics": {
                "n": stats.n,
                "mean_similarity": stats.mean_similarity,
                "mean_matched_similarity": stats.mean_matched_similarity,
                "ci_95_low": stats.ci_low,
                "ci_95_high": stats.ci_high,
                "ci_matched_95_low": stats.ci_matched_low,
                "ci_matched_95_high": stats.ci_matched_high,
                "mean_false_negatives": stats.mean_false_negatives,
                "mean_false_positives": stats.mean_false_positives,
            },
            "statistical_tests": {
                "wilcoxon_signed_rank": {
                    "description": "H0: median similarity = 0 (chance baseline). One-sided (greater).",
                    "statistic": stats.wilcoxon_stat,
                    "p_value": stats.wilcoxon_p,
                }
            },
            "lemmas": sorted(eval_df["lemma"].tolist()),
        }
        path = self.root / f"{llm_name.replace('/', '_')}_results.json"
        path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
        log.info("Written: %s", path)

    def write_cross_model_stats(self, cross: CrossModelStats) -> None:
        out = {
            "friedman_test": {
                "description": "Non-parametric test for differences across all three LLMs on matched lemma similarities.",
                "statistic": cross.friedman_stat,
                "p_value": cross.friedman_p,
            },
            "pairwise_wilcoxon_bonferroni": cross.pairwise,
        }
        path = self.root / "cross_model_stats.json"
        path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
        log.info("Written: %s", path)

    def write_cluster_stats(self, cluster_stats: list[ClusterStats]) -> None:
        if not cluster_stats:
            return
        records: dict[str, dict[str, dict]] = {}
        for cs in cluster_stats:
            records.setdefault(cs.cluster, {})[cs.model_name] = {
                "n": cs.n,
                "mean_similarity": cs.mean_similarity,
                "mean_matched_similarity": cs.mean_matched_similarity,
                "mean_false_negatives": cs.mean_false_negatives,
                "mean_false_positives": cs.mean_false_positives,
            }
        path = self.root / "cluster_stats.json"
        path.write_text(json.dumps(records, ensure_ascii=False, indent=2))
        log.info("Written: %s", path)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_pipeline(
    embedding_model: EmbeddingModel, batch_type: str = "definition-generator"
) -> None:
    log.info(
        "=== Starting evaluation with embedding model: %s ===", embedding_model.value
    )

    # Load ground truth
    gt_df = GroundTruthLoader(GT_BASE_PATH).load()

    # Cluster lookup (may be empty if tdm.sampled.json is absent)
    cluster_lookup = ClusterLoader(TDM_SAMPLED_PATH).load()

    writer = ReportWriter(OUTPUT_BASE, embedding_model, batch_type)
    cache = EvaluationCache(writer.root)
    plotter = Plotter(writer.root, embedding_model_name=embedding_model.short_name)

    pred_base = (
        PREDICTIONS_V2_BASE_PATH
        if batch_type == "definition-generator-v2"
        else PREDICTIONS_BASE_PATH
    )
    pred_loader = PredictionLoader(pred_base)

    # Lazy-load embedder only when at least one cache miss is present
    embedder: HuggingFaceEmbedder | None = None

    all_results: dict[str, pd.DataFrame] = {}
    all_stats: list[ModelStats] = []

    for llm_name, subdir in LLM_MODELS.items():
        log.info("--- Evaluating LLM: %s ---", llm_name)

        # --- Cache lookup ---
        cached = cache.get(llm_name)
        if cached is not None:
            eval_df = cached
            log.info("Skipping embedding computation for %s (cache hit)", llm_name)
        else:
            # Load embedder on first miss
            if embedder is None:
                embedder = HuggingFaceEmbedder(embedding_model)
            evaluator = SimilarityEvaluator(embedder)
            pred_df = pred_loader.load(llm_name, subdir)
            eval_df = evaluator.evaluate_dataframe(pred_df, gt_df)
            cache.set(llm_name, eval_df)
        eval_df.to_parquet(
            writer.root / f"{llm_name.replace('/', '_')}_eval.parquet", index=False
        )

        analyser = StatisticalAnalyser(chance_baseline=0.0)
        stats = analyser.per_model(llm_name, eval_df)
        writer.write_model_results(llm_name, eval_df, stats)
        all_results[llm_name] = eval_df
        all_stats.append(stats)

    # Per-model plots (distributions, FP/FN, violin)
    plotter.plot_similarity_distributions(all_results)
    plotter.plot_mean_similarity_comparison(all_stats)
    plotter.plot_fp_fn(all_stats)
    plotter.plot_violin(all_results)

    # Cluster metrics
    analyser = StatisticalAnalyser(chance_baseline=0.0)
    cluster_stats = analyser.per_cluster(all_results, cluster_lookup)
    writer.write_cluster_stats(cluster_stats)
    plotter.plot_cluster_metrics(cluster_stats)
    plotter.plot_cluster_fp_fn(cluster_stats)

    # Cross-model stats
    cross = analyser.cross_model(all_results)
    writer.write_cross_model_stats(cross)
    plotter.plot_pairwise_heatmap(cross, list(LLM_MODELS.keys()))

    log.info("=== Done. Results written to: %s ===", writer.root)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate LLM definition generation against Senso Comune ground truth.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        choices=["multilingual-e5-base", "all-MiniLM-L6-v2", "Qwen3-Embedding-4B"],
        default=None,
        help=(
            "Embedding model to use for similarity computation. "
            "If omitted, all models are run sequentially."
        ),
    )
    parser.add_argument(
        "--gt-path",
        type=Path,
        default=GT_BASE_PATH,
        help="Path to the Senso Comune ground truth directory.",
    )
    parser.add_argument(
        "--batch-type",
        type=str,
        choices=["definition-generator", "definition-generator-v2"],
        default="definition-generator",
        help="Which batch run to evaluate: 'definition-generator' for DefinitionGeneratorBatch, 'definition-generator-v2' for DefinitionGeneratorV2Batch.",
    )
    parser.add_argument(
        "--predictions-path",
        type=Path,
        default=PREDICTIONS_BASE_PATH,
        help="Path to the DefinitionGeneratorBatch output directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_BASE,
        help="Root output directory for evaluation reports.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Allow CLI overrides of global paths
    global GT_BASE_PATH, PREDICTIONS_BASE_PATH, OUTPUT_BASE
    GT_BASE_PATH = args.gt_path
    OUTPUT_BASE = args.output
    if args.batch_type == "definition-generator-v2":
        PREDICTIONS_BASE_PATH = (
            _RESOURCES_PATH / "generated/DefinitionGeneratorV2Batch"
        ).resolve()
    elif args.predictions_path != PREDICTIONS_BASE_PATH:
        PREDICTIONS_BASE_PATH = args.predictions_path

    models_to_run: list[EmbeddingModel]
    if args.embedding_model is None:
        models_to_run = list(EmbeddingModel)
        log.info(
            "No embedding model specified — running all %d models.", len(models_to_run)
        )
    else:
        models_to_run = [EmbeddingModel.from_cli(args.embedding_model)]

    for em in models_to_run:
        run_pipeline(em, args.batch_type)


if __name__ == "__main__":
    main()
