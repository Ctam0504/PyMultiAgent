"""
train_classifier_v3.py
======================
Pipeline huấn luyện, đánh giá & giải thích mô hình phân loại (True/False)
tuân thủ chuẩn Nghiên cứu Khoa học (NCKH) 7 bước:

  Bước 1: Thiết lập cố định môi trường & Seed (Reproducibility)
  Bước 2: Phân chia dữ liệu chống rò rỉ (Stratified Splitting: 85% Train/Val - 15% Hold-out Test)
  Bước 3: Tiền xử lý & Trích xuất đặc trưng (Feature Engineering, Scaling & Selection)
  Bước 4: Xử lý mất cân bằng lớp (Cost-Sensitive Learning: w_True = 1.92)
  Bước 5: Huấn luyện, Tối ưu & Tinh chỉnh (Baselines, Tree Ensembles, PyTorch MLP, Optuna)
  Bước 6: Đánh giá Hold-out Test & Kiểm chứng thống kê (Paired t-test / Wilcoxon, p-value < 0.05)
  Bước 7: Giải thích mô hình (SHAP Summary Plot & Error Analysis)

LƯU Ý: Chỉ lưu duy nhất 1 artifact là file PyTorch '.pt' (best_model.pt).
"""

import ast
import json
import os
import random
import re
import sys
import time
import warnings

# Đảm bảo UTF-8 console output
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")  # Chạy headless không cần màn hình GUI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import xgboost as xgb

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


# ══════════════════════════════════════════════════════════════════════
# BƯỚC 1: THIẾT LẬP CỐ ĐỊNH MÔI TRƯỜNG & SEED (REPRODUCIBILITY)
# ══════════════════════════════════════════════════════════════════════
SEED = 42
WORKSPACE = os.path.dirname(os.path.abspath(__file__))

def set_all_seeds(seed: int = 42):
    """Cố định seed trên toàn bộ thư viện để đảm bảo khả năng tái lập (Reproducibility)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[Bước 1] Đã cố định toàn bộ ngẫu nhiên (Reproducibility Seed = {seed})")


# ══════════════════════════════════════════════════════════════════════
# TRÍCH XUẤT ĐẶC TRƯNG MÃ NGUỒN (AST & LOGPROB)
# ══════════════════════════════════════════════════════════════════════
def _nesting_depth(tree) -> int:
    BLOCK = (
        ast.If, ast.For, ast.While, ast.With,
        ast.FunctionDef, ast.AsyncFunctionDef,
        ast.ClassDef, ast.Try, ast.AsyncWith,
    )
    max_depth = [0]
    def _walk(node, depth):
        if isinstance(node, BLOCK):
            max_depth[0] = max(max_depth[0], depth)
            depth += 1
        for child in ast.iter_child_nodes(node):
            _walk(child, depth)
    _walk(tree, 0)
    return max_depth[0]


def extract_features_from_code(code: str) -> dict:
    """Trích xuất 18 đặc trưng cú pháp và độ phức tạp từ đoạn code Python."""
    feats = {
        "code_len": 0, "code_lines": 0, "blank_ratio": 0.0,
        "Ops": 0, "Branch": 0, "Loops": 0, "Structs": 0,
        "cyclomatic": 1, "max_depth": 0,
        "num_returns": 0, "has_try_except": 0,
        "num_calls": 0, "num_imports": 0,
        "num_comparisons": 0, "avg_id_len": 0.0,
        "is_syntax_valid": 0, "has_docstring": 0,
    }
    if not code:
        return feats

    lines = code.splitlines()
    total_lines = len(lines)
    blank_lines = sum(1 for l in lines if not l.strip())
    feats["code_len"] = len(code)
    feats["code_lines"] = total_lines
    feats["blank_ratio"] = blank_lines / total_lines if total_lines else 0.0

    try:
        tree = ast.parse(code)
    except SyntaxError:
        feats["is_syntax_valid"] = 0
        return feats

    feats["is_syntax_valid"] = 1
    OP_NODES = (
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
        ast.FloorDiv, ast.LShift, ast.RShift, ast.BitOr,
        ast.BitXor, ast.BitAnd, ast.Eq, ast.NotEq, ast.Lt,
        ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot,
        ast.In, ast.NotIn, ast.And, ast.Or, ast.Not,
        ast.Invert, ast.UAdd, ast.USub,
        ast.Assign, ast.AugAssign, ast.AnnAssign,
    )
    BRANCH_NODES = (ast.If, ast.IfExp, getattr(ast, "Match", type(None)))
    LOOP_NODES = (
        ast.For, ast.While, ast.AsyncFor,
        ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp,
    )
    STRUCT_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    IMPORT_NODES = (ast.Import, ast.ImportFrom)
    CC_NODES = (ast.If, ast.For, ast.While, ast.ExceptHandler, ast.BoolOp, ast.IfExp)

    ops = branch = loops = structs = returns = calls = imports_ = comparisons = 0
    cc = 1
    identifiers = []

    for node in ast.walk(tree):
        if isinstance(node, OP_NODES):
            ops += 1
        elif isinstance(node, BRANCH_NODES):
            branch += 1
        elif isinstance(node, LOOP_NODES):
            loops += 1
        elif isinstance(node, STRUCT_NODES):
            structs += 1

        if isinstance(node, ast.Return):
            returns += 1
        if isinstance(node, ast.Try):
            feats["has_try_except"] = 1
        if isinstance(node, ast.Call):
            calls += 1
        if isinstance(node, IMPORT_NODES):
            imports_ += 1
        if isinstance(node, ast.Compare):
            comparisons += 1
        if isinstance(node, CC_NODES):
            cc += 1
        if isinstance(node, ast.Name):
            identifiers.append(node.id)
        elif isinstance(node, ast.arg):
            identifiers.append(node.arg)

    feats["Ops"] = ops
    feats["Branch"] = branch
    feats["Loops"] = loops
    feats["Structs"] = structs
    feats["cyclomatic"] = cc
    feats["max_depth"] = _nesting_depth(tree)
    feats["num_returns"] = returns
    feats["num_calls"] = calls
    feats["num_imports"] = imports_
    feats["num_comparisons"] = comparisons
    feats["avg_id_len"] = float(np.mean([len(i) for i in identifiers])) if identifiers else 0.0

    if (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    ):
        feats["has_docstring"] = 1

    return feats


def load_raw_dataset(filename="leetcode2.json"):
    """Đọc dữ liệu thô, ưu tiên leetcode2.json (3,078 mẫu) hoặc fallback leetcode.json."""
    path = os.path.join(WORKSPACE, filename)
    if not os.path.exists(path):
        fallback = os.path.join(WORKSPACE, "leetcode.json")
        if os.path.exists(fallback):
            path = fallback
        else:
            raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {path}")

    print(f"[*] Đang tải dữ liệu từ: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    rows = []
    for item in raw:
        if item.get("result") is None:
            continue
        code = item.get("generated_code", "")
        feats = extract_features_from_code(code)
        feats["logprob"] = item.get("logprob", np.nan)
        feats["result"] = 1 if item.get("result") is True else 0
        rows.append(feats)

    df = pd.DataFrame(rows)
    print(f"[*] Tổng số mẫu hợp lệ: {len(df)} | Lớp 0 (False): {sum(df['result']==0)} | Lớp 1 (True): {sum(df['result']==1)}")
    return df


# ══════════════════════════════════════════════════════════════════════
# BƯỚC 3: TIỀN XỬ LÝ & LỰA CHỌN ĐẶC TRƯNG CHỐNG RÒ RỈ (LEAK-FREE PREPROCESSING)
# ══════════════════════════════════════════════════════════════════════
class LeakFreePreprocessor:
    """
    Tiền xử lý độc lập, chỉ fit trên tập Train, transform sang Val/Test:
      1. Median Imputation cho logprob
      2. Feature engineering: logprob_per_line
      3. Lọc Low Variance
      4. Lọc đa cộng tuyến (|r| > 0.85)
      5. StandardScaler
    """
    def __init__(self, corr_threshold=0.85):
        self.corr_threshold = corr_threshold
        self.logprob_median = 0.0
        self.selected_features = []
        self.scaler = StandardScaler()

    def fit(self, X_df: pd.DataFrame, y: np.ndarray):
        df = X_df.copy()
        
        # 1. Median Imputation
        self.logprob_median = float(df["logprob"].median())
        df["logprob"] = df["logprob"].fillna(self.logprob_median)
        df["logprob_per_line"] = df["logprob"] / df["code_lines"].replace(0, 1)

        # 2. Loại bỏ đặc trưng phương sai gần bằng 0
        selector = VarianceThreshold(threshold=1e-4)
        selector.fit(df)
        survived_cols = list(df.columns[selector.get_support()])
        df_survived = df[survived_cols]

        # 3. Loại bỏ đa cộng tuyến (|r| > corr_threshold)
        corr_matrix = df_survived.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > self.corr_threshold)]
        self.selected_features = [col for col in survived_cols if col not in to_drop]

        # 4. Fit Scaler
        X_selected = df[self.selected_features].values
        self.scaler.fit(X_selected)
        return self

    def transform(self, X_df: pd.DataFrame) -> np.ndarray:
        df = X_df.copy()
        df["logprob"] = df["logprob"].fillna(self.logprob_median)
        df["logprob_per_line"] = df["logprob"] / df["code_lines"].replace(0, 1)
        X_selected = df[self.selected_features].values
        return self.scaler.transform(X_selected)

    def fit_transform(self, X_df: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        return self.fit(X_df, y).transform(X_df)


# ══════════════════════════════════════════════════════════════════════
# MÔ HÌNH PYTORCH MLP (DEEP LEARNING TABULAR CLASSIFIER)
# ══════════════════════════════════════════════════════════════════════
class PyTorchMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout_rate: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class PyTorchClassifierWrapper(BaseEstimator, ClassifierMixin):
    """Bọc PyTorch model để tương thích với scikit-learn API."""
    def __init__(self, input_dim: int = 18, hidden_dim: int = 64, lr: float = 1e-3, 
                 epochs: int = 40, batch_size: int = 64, weight_decay: float = 1e-4, 
                 pos_weight: float = 1.92, device: str = None):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.pos_weight = pos_weight
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def fit(self, X, y):
        self.input_dim = X.shape[1]
        self.model = PyTorchMLP(self.input_dim, self.hidden_dim).to(self.device)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.pos_weight], device=self.device))

        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for _ in range(self.epochs):
            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                logits = self.model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            tx = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.model(tx)
            probs = torch.sigmoid(logits).cpu().numpy()
        return np.column_stack([1.0 - probs, probs])

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)


# ══════════════════════════════════════════════════════════════════════
# BƯỚC 5: ĐÁNH GIÁ CHỈ SỐ VÀ OPTUNA TUNING
# ══════════════════════════════════════════════════════════════════════
def compute_all_metrics(y_true, y_pred, y_prob):
    """Tính toàn bộ bộ chỉ số theo chuẩn NCKH."""
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Macro_F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "F1_True": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "Precision_True": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "Recall_True": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "ROC_AUC": float(roc_auc_score(y_true, y_prob)),
        "PR_AUC": float(average_precision_score(y_true, y_prob)),
    }


def tune_xgboost_optuna(X_train, y_train, pos_weight, n_trials=25):
    """Optuna TPE tối ưu siêu tham số XGBoost trên 5-Fold CV."""
    if not HAS_OPTUNA:
        return xgb.XGBClassifier(n_estimators=100, scale_pos_weight=pos_weight, random_state=SEED, eval_metric="logloss")

    print("\n[*] Đang tối ưu siêu tham số XGBoost bằng Optuna (5-Fold CV)...")
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 60, 200),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
            "scale_pos_weight": pos_weight,
            "random_state": SEED,
            "eval_metric": "logloss",
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            clf = xgb.XGBClassifier(**params)
            clf.fit(X_train[train_idx], y_train[train_idx])
            preds = clf.predict(X_train[val_idx])
            scores.append(f1_score(y_train[val_idx], preds, average="macro"))
        return np.mean(scores)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials)
    print(f"    Best Optuna Macro F1: {study.best_value:.4f}")
    best_params = study.best_params
    best_params.update({"scale_pos_weight": pos_weight, "random_state": SEED, "eval_metric": "logloss"})
    return xgb.XGBClassifier(**best_params)


# ══════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════
def main():
    t_start = time.time()
    set_all_seeds(SEED)

    # ------------------------------------------------------------------
    # BƯỚC 2: PHÂN CHIA DỮ LIỆU CHỐNG RÒ RỈ (STRATIFIED SPLITTING)
    # ------------------------------------------------------------------
    df_raw = load_raw_dataset("leetcode2.json")
    X_full = df_raw.drop(columns=["result"])
    y_full = df_raw["result"].values

    # Tính trọng số Cost-Sensitive
    n_false = np.sum(y_full == 0)
    n_true = np.sum(y_full == 1)
    pos_weight = float(n_false / n_true)  # ~ 1.9226
    print(f"\n[Bước 4] Cost-Sensitive Weight (w_True): {n_false} / {n_true} = {pos_weight:.4f}")

    # Tách Hold-out Test 15% (462 mẫu: 304 False, 158 True)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    train_val_idx, test_idx = next(sss.split(X_full, y_full))

    X_train_val = X_full.iloc[train_val_idx].reset_index(drop=True)
    y_train_val = y_full[train_val_idx]

    X_holdout_test = X_full.iloc[test_idx].reset_index(drop=True)
    y_holdout_test = y_full[test_idx]

    print(f"\n[Bước 2] Phân chia dữ liệu:")
    print(f"  • Tập Train/Validation (85%): {len(y_train_val):,} mẫu (False: {sum(y_train_val==0)}, True: {sum(y_train_val==1)})")
    print(f"  • Tập Hold-out Test (15%):     {len(y_holdout_test):,} mẫu (False: {sum(y_holdout_test==0)}, True: {sum(y_holdout_test==1)}) -> Cất đi hoàn toàn!")

    # ------------------------------------------------------------------
    # BƯỚC 3 & 5: CROSS-VALIDATION ĐÁNH GIÁ CÁC MÔ HÌNH TRÊN TRAIN/VAL
    # ------------------------------------------------------------------
    print(f"\n[Bước 3 & 5] Huấn luyện & Đánh giá 5-Fold Cross-Validation trên Train/Val...")
    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # Danh sách các ứng viên mô hình
    model_candidates = {
        "Baseline (Dummy)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(class_weight={0: 1.0, 1: pos_weight}, max_iter=500, random_state=SEED),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight={0: 1.0, 1: pos_weight}, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight={0: 1.0, 1: pos_weight}, random_state=SEED),
        "XGBoost (Default)": xgb.XGBClassifier(scale_pos_weight=pos_weight, random_state=SEED, eval_metric="logloss"),
        "PyTorch MLP": PyTorchClassifierWrapper(pos_weight=pos_weight),
    }
    if HAS_LGB:
        model_candidates["LightGBM"] = lgb.LGBMClassifier(scale_pos_weight=pos_weight, random_state=SEED, verbose=-1)

    cv_macro_f1_scores = {name: [] for name in model_candidates}

    # Chạy 5-Fold CV (Mỗi fold fit Preprocessor riêng biệt để chống rò rỉ)
    for fold, (train_f_idx, val_f_idx) in enumerate(cv5.split(X_train_val, y_train_val), 1):
        X_tr_fold = X_train_val.iloc[train_f_idx]
        y_tr_fold = y_train_val[train_f_idx]
        X_va_fold = X_train_val.iloc[val_f_idx]
        y_va_fold = y_train_val[val_f_idx]

        # Tiền xử lý chống rò rỉ trong fold
        prep_fold = LeakFreePreprocessor()
        X_tr_proc = prep_fold.fit_transform(X_tr_fold, y_tr_fold)
        X_va_proc = prep_fold.transform(X_va_fold)

        for name, clf in model_candidates.items():
            clf.fit(X_tr_proc, y_tr_fold)
            preds = clf.predict(X_va_proc)
            score = f1_score(y_va_fold, preds, average="macro", zero_division=0)
            cv_macro_f1_scores[name].append(score)

    print("\n" + "─"*55)
    print(f"| {'Mô hình (Model)':<25} | {'CV Macro F1 (Mean ± Std)':<23} |")
    print("─"*55)
    for name, scores in cv_macro_f1_scores.items():
        print(f"| {name:<25} | {np.mean(scores):.4f} ± {np.std(scores):.4f}            |")
    print("─"*55)

    # ------------------------------------------------------------------
    # OPTUNA TUNING TRÊN MÔ HÌNH CHỦ ĐẠO
    # ------------------------------------------------------------------
    # Fit preprocessor toàn bộ tập Train/Val để chuẩn bị huấn luyện mô hình tối ưu
    global_preprocessor = LeakFreePreprocessor()
    X_train_val_proc = global_preprocessor.fit_transform(X_train_val, y_train_val)
    selected_cols = global_preprocessor.selected_features
    print(f"\n[*] Số đặc trưng sau khi lọc Low-variance & Đa cộng tuyến: {len(selected_cols)} / 19")
    print(f"    Danh sách: {selected_cols}")

    best_tuned_xgb = tune_xgboost_optuna(X_train_val_proc, y_train_val, pos_weight, n_trials=20)
    best_tuned_xgb.fit(X_train_val_proc, y_train_val)

    # Huấn luyện lại PyTorch MLP trên toàn bộ Train/Val
    final_pytorch_mlp = PyTorchClassifierWrapper(input_dim=len(selected_cols), pos_weight=pos_weight, epochs=45)
    final_pytorch_mlp.fit(X_train_val_proc, y_train_val)

    # ------------------------------------------------------------------
    # BƯỚC 6: ĐÁNH GIÁ TRÊN HOLD-OUT TEST SET (15% = 462 MẪU)
    # ------------------------------------------------------------------
    print("\n" + "═"*70)
    print("🎯 BƯỚC 6: ĐÁNH GIÁ TRÊN HOLD-OUT TEST SET (MỞ RA LẦN ĐẦU TIÊN)")
    print("═"*70)
    X_test_proc = global_preprocessor.transform(X_holdout_test)

    # Tìm threshold tối ưu cho model tốt nhất trên validation split
    val_probs = best_tuned_xgb.predict_proba(X_train_val_proc)[:, 1]
    best_thresh, best_thresh_score = 0.5, 0.0
    for t in np.linspace(0.2, 0.8, 61):
        sc = f1_score(y_train_val, (val_probs >= t).astype(int), average="macro")
        if sc > best_thresh_score:
            best_thresh_score, best_thresh = sc, t

    # Dự đoán trên Hold-out Test
    y_test_probs = best_tuned_xgb.predict_proba(X_test_proc)[:, 1]
    y_test_preds = (y_test_probs >= best_thresh).astype(int)

    test_metrics = compute_all_metrics(y_holdout_test, y_test_preds, y_test_probs)

    print(f"  • Ngưỡng phân loại tối ưu (Optimal Threshold): {best_thresh:.3f}")
    print(f"  • Accuracy:        {test_metrics['Accuracy']:.4f}")
    print(f"  • Macro F1-Score:  {test_metrics['Macro_F1']:.4f}")
    print(f"  • F1 (Lớp True):   {test_metrics['F1_True']:.4f}")
    print(f"  • Precision (True): {test_metrics['Precision_True']:.4f}")
    print(f"  • Recall (True):    {test_metrics['Recall_True']:.4f}")
    print(f"  • ROC-AUC:         {test_metrics['ROC_AUC']:.4f}")
    print(f"  • PR-AUC:          {test_metrics['PR_AUC']:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_holdout_test, y_test_preds)
    print(f"\n  Confusion Matrix:")
    print(f"    TN={cm[0,0]:<4}  FP={cm[0,1]:<4}")
    print(f"    FN={cm[1,0]:<4}  TP={cm[1,1]:<4}")

    # Vẽ và lưu Confusion Matrix & PR Curve
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Confusion Matrix Plot
    cax = axes[0].matshow(cm, cmap=plt.cm.Blues, alpha=0.8)
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14, fontweight="bold")
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(["Pred False", "Pred True"])
    axes[0].set_yticklabels(["True False", "True True"])
    axes[0].set_title(f"Confusion Matrix (Thresh={best_thresh:.2f})")

    # PR Curve Plot
    precisions, recalls, _ = precision_recall_curve(y_holdout_test, y_test_probs)
    axes[1].plot(recalls, precisions, color="darkorange", lw=2, label=f"PR curve (AUC = {test_metrics['PR_AUC']:.3f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend(loc="lower left")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    eval_fig_path = os.path.join(WORKSPACE, "evaluation_metrics.png")
    plt.savefig(eval_fig_path, dpi=300)
    plt.close()
    print(f"[*] Đã lưu đồ thị đánh giá vào: {eval_fig_path}")

    # ------------------------------------------------------------------
    # KIỂM ĐỊNH Ý NGHĨA THỐNG KÊ (STATISTICAL SIGNIFICANCE TESTING)
    # ------------------------------------------------------------------
    print("\n" + "═"*70)
    print("🔬 KIỂM ĐỊNH Ý NGHĨA THỐNG KÊ (PAIRED T-TEST & WILCOXON TEST)")
    print("═"*70)
    rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
    baseline_f1s = []
    proposed_f1s = []

    baseline_model = LogisticRegression(class_weight={0: 1.0, 1: pos_weight}, max_iter=500, random_state=SEED)
    for tr_idx, va_idx in rkf.split(X_train_val_proc, y_train_val):
        # Baseline
        baseline_model.fit(X_train_val_proc[tr_idx], y_train_val[tr_idx])
        base_preds = baseline_model.predict(X_train_val_proc[va_idx])
        baseline_f1s.append(f1_score(y_train_val[va_idx], base_preds, average="macro"))

        # Proposed (Tuned XGB)
        best_tuned_xgb.fit(X_train_val_proc[tr_idx], y_train_val[tr_idx])
        prop_preds = best_tuned_xgb.predict(X_train_val_proc[va_idx])
        proposed_f1s.append(f1_score(y_train_val[va_idx], prop_preds, average="macro"))

    t_stat, p_val_ttest = stats.ttest_rel(proposed_f1s, baseline_f1s)
    wilc_stat, p_val_wilcoxon = stats.wilcoxon(np.array(proposed_f1s) - np.array(baseline_f1s), alternative="greater")

    print(f"  • Proposed Model Macro F1 Mean: {np.mean(proposed_f1s):.4f} | Baseline Macro F1 Mean: {np.mean(baseline_f1s):.4f}")
    print(f"  • Paired t-test:        t-statistic = {t_stat:.4f}, p-value = {p_val_ttest:.6f}")
    print(f"  • Wilcoxon Signed-Rank: statistic   = {wilc_stat:.4f}, p-value = {p_val_wilcoxon:.6f}")
    if p_val_ttest < 0.05:
        print(f"  ==> KẾT LUẬN: Sự cải thiện của mô hình đề xuất có ý nghĩa thống kê (p < 0.05)!")
    else:
        print(f"  ==> KẾT LUẬN: p-value >= 0.05 (chưa đạt ngưỡng ý nghĩa thống kê 95%).")

    # ------------------------------------------------------------------
    # BƯỚC 7: GIẢI THÍCH MÔ HÌNH BẰNG SHAP (MODEL INTERPRETABILITY)
    # ------------------------------------------------------------------
    if HAS_SHAP:
        print("\n" + "═"*70)
        print("🧠 BƯỚC 7: GIẢI THÍCH MÔ HÌNH VỚI SHAP (SHAPLEY ADDITIVE EXPLANATIONS)")
        print("═"*70)
        explainer = shap.TreeExplainer(best_tuned_xgb)
        shap_values = explainer.shap_values(X_test_proc)

        # 1. SHAP Summary Plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test_proc, feature_names=selected_cols, show=False)
        shap_summary_path = os.path.join(WORKSPACE, "shap_summary.png")
        plt.tight_layout()
        plt.savefig(shap_summary_path, dpi=300)
        plt.close()
        print(f"[*] Đã lưu SHAP Summary Plot vào: {shap_summary_path}")

        # 2. Phân tích lỗi (False Positives & False Negatives)
        fp_indices = np.where((y_holdout_test == 0) & (y_test_preds == 1))[0]
        fn_indices = np.where((y_holdout_test == 1) & (y_test_preds == 0))[0]
        print(f"  • Số ca False Positive (thực tế Failed nhưng đoán Passed): {len(fp_indices)}")
        print(f"  • Số ca False Negative (thực tế Passed nhưng đoán Failed): {len(fn_indices)}")

    # ------------------------------------------------------------------
    # LƯU DUY NHẤT ARTIFACT .PT (THEO YÊU CẦU NGHIÊM NGẶT)
    # ------------------------------------------------------------------
    print("\n" + "═"*70)
    print("💾 ĐÓNG GÓI VÀ LƯU DUY NHẤT ARTIFACT PYTORCH .PT")
    print("═"*70)

    # Chuyển đổi trọng số và cấu trúc mô hình sang định dạng lưu trữ PyTorch checkpoint
    checkpoint = {
        "format_version": "v3_nckh",
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": SEED,
        "class_imbalance": {
            "n_false": int(n_false),
            "n_true": int(n_true),
            "scale_pos_weight": float(pos_weight),
        },
        "preprocessor": {
            "logprob_median": float(global_preprocessor.logprob_median),
            "selected_features": list(global_preprocessor.selected_features),
            "scaler_mean": global_preprocessor.scaler.mean_.tolist(),
            "scaler_scale": global_preprocessor.scaler.scale_.tolist(),
        },
        "best_model_name": "XGBoost_Tuned_Optuna",
        "best_threshold": float(best_thresh),
        "test_metrics": test_metrics,
        "statistical_testing": {
            "p_val_ttest": float(p_val_ttest),
            "p_val_wilcoxon": float(p_val_wilcoxon),
        },
        "pytorch_mlp_state_dict": final_pytorch_mlp.model.state_dict(),
        "xgboost_dump": best_tuned_xgb.get_booster().save_raw(raw_format="json").decode("utf-8"),
    }

    pt_save_path = os.path.join(WORKSPACE, "best_model.pt")
    torch.save(checkpoint, pt_save_path)
    print(f"[✓] Đã lưu thành công duy nhất file mô hình: {pt_save_path}")

    # Xóa các file .joblib cũ nếu còn để bảo đảm chỉ giữ file .pt
    for old_file in ["best_model.joblib", "scaler.joblib", "model_metadata.joblib"]:
        p = os.path.join(WORKSPACE, old_file)
        if os.path.exists(p):
            try:
                os.remove(p)
                print(f"[x] Đã dọn dẹp file cũ không dùng: {old_file}")
            except Exception:
                pass

    print(f"\n⏱ Tổng thời gian chạy pipeline: {(time.time() - t_start):.2f} giây")
    print("=" * 70)


if __name__ == "__main__":
    main()
