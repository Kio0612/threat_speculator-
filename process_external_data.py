import os
import gc
import time
import numpy as np
import polars as pl
import torch
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import StandardScaler

print("=" * 70)
print("STARTING EXTERNAL DATA PIPELINE (PHASES 1, 2 & 3)")
print("=" * 70)

print("\n[Phase 1] Streaming and filtering raw ClinVar data...")
t0 = time.time()
raw_path = "variant_summary.txt"
if not os.path.exists(raw_path):
    raise FileNotFoundError(f"Missing {raw_path} in current directory.")

lazy_df = (
    pl.scan_csv(raw_path, separator="\t", null_values=["-", "na", ""], infer_schema_length=10000, ignore_errors=True)
    .filter(pl.col("Assembly") == "GRCh38")
    .filter(pl.col("Chromosome").is_not_null())
    .filter(pl.col("Start").is_not_null() & (pl.col("Start") > 0))
    .filter(pl.col("Stop").is_not_null() & (pl.col("Stop") >= pl.col("Start")))
    .filter(pl.col("ReviewStatus").is_not_null())
    .filter(~pl.col("ReviewStatus").str.contains("no assertion criteria provided"))
    .filter(pl.col("ClinicalSignificance").str.contains("(?i)pathogenic") | pl.col("ClinicalSignificance").str.contains("(?i)benign"))
    .filter(~pl.col("ClinicalSignificance").str.contains("(?i)uncertain"))
    .filter(~pl.col("ClinicalSignificance").str.contains("(?i)conflicting"))
)

lazy_df = lazy_df.with_columns(
    pl.when(pl.col("ClinicalSignificance").str.contains("(?i)pathogenic")).then(1).otherwise(0).alias("Label")
)

clean_df = lazy_df.select(["Chromosome", "Start", "Stop", "ReferenceAllele", "AlternateAllele", "Type", "GeneSymbol", "Label"]).collect()
clean_df = clean_df.with_columns(pl.col("Chromosome").str.replace(r"(?i)^chr", ""))
valid_chroms = [str(i) for i in range(1, 23)] + ["X", "Y"]
clean_df = clean_df.filter(pl.col("Chromosome").is_in(valid_chroms))
print(f"[Phase 1 Completed] Retained {len(clean_df):,} verified variants in {time.time() - t0:.2f}s")

print("\n[Phase 2] Computing 78-dimensional feature vectors...")
t1 = time.time()
clean_df = clean_df.with_columns([
    (pl.col("Stop") - pl.col("Start") + 1).alias("Span"),
    pl.col("ReferenceAllele").fill_null("").str.len_bytes().alias("RefLen"),
    pl.col("AlternateAllele").fill_null("").str.len_bytes().alias("AltLen"),
    pl.col("GeneSymbol").fill_null("UNKNOWN").alias("GeneSymbol"),
])
clean_df = clean_df.with_columns([
    (pl.col("AltLen") - pl.col("RefLen")).alias("DeltaLen"),
    ((pl.col("AltLen") - pl.col("RefLen")).abs() % 3 != 0).cast(pl.Float32).alias("FrameshiftSignal"),
    (pl.col("Start") / 250_000_000.0).alias("NormPos")
])

chrom_dummies = clean_df.select([pl.col("Chromosome") == c for c in valid_chroms]).to_numpy().astype(np.float32)
variant_types = ["single nucleotide variant", "deletion", "duplication", "insertion", "indel"]
type_dummies = clean_df.select([pl.col("Type") == t for t in variant_types]).to_numpy().astype(np.float32)

bases = ["A", "C", "G", "T"]
base_transitions = [f"{b1}>{b2}" for b1 in bases for b2 in bases if b1 != b2] + ["Indel"]
clean_df = clean_df.with_columns(
    pl.when((pl.col("RefLen") == 1) & (pl.col("AltLen") == 1))
    .then(pl.col("ReferenceAllele") + ">" + pl.col("AlternateAllele"))
    .otherwise(pl.lit("Indel")).alias("Transition")
)
transition_dummies = clean_df.select([pl.col("Transition") == tr for tr in base_transitions]).to_numpy().astype(np.float32)

gene_strings = clean_df["GeneSymbol"].to_list()
hasher = FeatureHasher(n_features=32, input_type="string")
gene_hashed = hasher.transform([[g] for g in gene_strings]).toarray().astype(np.float32)

cont_features = clean_df.select(["Span", "RefLen", "AltLen", "DeltaLen", "NormPos"]).to_numpy().astype(np.float32)
scaler = StandardScaler()
cont_scaled = scaler.fit_transform(cont_features)
frameshift_col = clean_df.select("FrameshiftSignal").to_numpy().astype(np.float32)

x_matrix = np.hstack([cont_scaled, frameshift_col, chrom_dummies, type_dummies, transition_dummies, gene_hashed]).astype(np.float32)
y_labels = clean_df["Label"].to_numpy().astype(np.int64)

torch.save({"x": torch.from_numpy(x_matrix), "y": torch.from_numpy(y_labels)}, "external_node_features.pt")
print(f"[Phase 2 Completed] Extracted Node Tensor Shape: {x_matrix.shape} in {time.time() - t1:.2f}s")

print("\n[Phase 3] Building local genomic graph topology (Window: 2,000 bp)...")
t2 = time.time()
sorted_df = clean_df.with_columns(pl.arange(0, clean_df.height).alias("original_idx")).sort(["Chromosome", "GeneSymbol", "Start"])
orig_indices = sorted_df["original_idx"].to_numpy()
starts = sorted_df["Start"].to_numpy()
genes = sorted_df["GeneSymbol"].to_numpy()
chroms = sorted_df["Chromosome"].to_numpy()

N = len(sorted_df)
edge_src_list, edge_dst_list = [], []
for k in range(1, 11):
    if k >= N: break
    valid = ((chroms[:-k] == chroms[k:]) & (genes[:-k] == genes[k:]) & ((starts[k:] - starts[:-k]) <= 2000))
    edge_src_list.append(orig_indices[:-k][valid])
    edge_dst_list.append(orig_indices[k:][valid])

if len(edge_src_list) > 0:
    forward_src = np.concatenate(edge_src_list)
    forward_dst = np.concatenate(edge_dst_list)
    full_src = np.concatenate([forward_src, forward_dst])
    full_dst = np.concatenate([forward_dst, forward_src])
    keys = (full_src.astype(np.int64) << 32) | (full_dst.astype(np.int64) & 0xFFFFFFFF)
    _, unique_indices = np.unique(keys, return_index=True)
    edge_index = np.vstack([full_src[unique_indices], full_dst[unique_indices]]).astype(np.int64)
else:
    edge_index = np.empty((2, 0), dtype=np.int64)

torch.save({"edge_index": torch.from_numpy(edge_index)}, "external_edge_index.pt")
print(f"[Phase 3 Completed] Constructed {edge_index.shape[1]:,} undirected edges in {time.time() - t2:.2f}s")
print("\n======================================================================")
print("PIPELINE COMPLETE. READY FOR EVALUATION.")
print("======================================================================")
