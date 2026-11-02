import typing
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

_orig_eval_type = getattr(typing, "_eval_type", None)
if _orig_eval_type is not None:
    def _eval_type_compat(t, globalns=None, localns=None, *args, **kwargs):
        try:
            return _orig_eval_type(t, globalns, localns, (), *args, **kwargs)
        except TypeError:
            try:
                return _orig_eval_type(t, globalns, localns, None, *args, **kwargs)
            except TypeError:
                return _orig_eval_type(t, globalns, localns, *args, **kwargs)
    typing._eval_type = _eval_type_compat


class VariantGraphSAGE(nn.Module):
    def __init__(self, in_channels: int = 78, hidden_dim: int = 128, out_dim: int = 64, dropout_rate: float = 0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_dim, aggr="mean")
        self.conv2 = SAGEConv(hidden_dim, out_dim, aggr="mean")
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(out_dim, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        return self.classifier(x)


device = torch.device("cpu")
model = VariantGraphSAGE(in_channels=78, hidden_dim=128, out_dim=64, dropout_rate=0.3)

weights_path = "variant_gnn_weights.pt"
if not os.path.exists(weights_path):
    print(f"\n[ERROR] '{weights_path}' was not found in the current folder.")
    exit(1)

model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
model.eval()
print("[1/3] Trained GraphSAGE weights successfully loaded into CPU memory.")

print("[2/3] Loading real-world external dataset tensors...")
t0 = time.time()

for req_file in ["external_node_features.pt", "external_edge_index.pt"]:
    if not os.path.exists(req_file):
        print(f"\n[ERROR] Missing required data artifact: '{req_file}'")
        exit(1)

ext_nodes = torch.load("external_node_features.pt", map_location=device, weights_only=False)
x_ext = ext_nodes["x"][:, :78].float()
y_ext = ext_nodes["y"].numpy()

ext_edges = torch.load("external_edge_index.pt", map_location=device, weights_only=False)
edge_index_ext = ext_edges["edge_index"].long()

print(f"      Variants (Nodes) : {x_ext.shape[0]:,} (shape: {list(x_ext.shape)})")
print(f"      Graph Edges      : {edge_index_ext.shape[1]:,}")
print(f"      Data loaded in   : {time.time() - t0:.2f}s")

print("[3/3] Executing model inference across external biological variants...")
t1 = time.time()

with torch.no_grad():
    logits = model(x_ext, edge_index_ext).squeeze(-1)
    probabilities = torch.sigmoid(logits).numpy()
    
    OPTIMAL_THRESHOLD = 0.21
    predictions = (probabilities >= OPTIMAL_THRESHOLD).astype(int)

print(f"      Inference completed in {time.time() - t1:.2f}s")

roc_auc = roc_auc_score(y_ext, probabilities)
pr_auc = average_precision_score(y_ext, probabilities)
f1 = f1_score(y_ext, predictions, zero_division=0)
precision = precision_score(y_ext, predictions, zero_division=0)
recall = recall_score(y_ext, predictions, zero_division=0)

print("\n" + "=" * 65)
print(f"REAL-WORLD EXTERNAL CLINVAR VALIDATION (Threshold: {OPTIMAL_THRESHOLD})")
print("=" * 65)
print(f"  Total Mutations Evaluated : {len(y_ext):,}")
print(f"  ROC-AUC Score             : {roc_auc:.4f}")
print(f"  PR-AUC Score              : {pr_auc:.4f}")
print(f"  F1-Score                  : {f1:.4f}")
print(f"  Precision                 : {precision:.4f}")
print(f"  Recall (Sensitivity)      : {recall:.4f}")
print("=" * 65)
