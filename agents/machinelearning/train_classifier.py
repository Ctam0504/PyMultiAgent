import json
import os
import ast
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    classification_report, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score,
    fbeta_score  # Thêm thư viện tính F-beta để tối ưu hóa ngưỡng rẽ nhánh
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import joblib

def check_syntax(code_str):
    """Check if the generated code has valid Python syntax."""
    if not code_str:
        return 0
    try:
        ast.parse(code_str)
        return 1
    except SyntaxError:
        return 0

def load_and_preprocess_data(workspace_dir):
    datasets = []
    
    # 1. Load HumanEval
    he_path = os.path.join(workspace_dir, "HumanEval.json")
    if os.path.exists(he_path):
        with open(he_path, "r", encoding="utf-8") as f:
            he_data = json.load(f)
        for item in he_data:
            if item.get("result") is None:
                continue
            datasets.append({
                "prompt": item.get("prompt", ""),
                "generated_code": item.get("generated_code", ""),
                "logprob": item.get("logprob", 0.0),
                "result": 1 if item.get("result") is True else 0,
                "is_humaneval": 1
            })
            
    # 2. Load MBPP
    mbpp_path = os.path.join(workspace_dir, "combined_mbpp.json")
    if os.path.exists(mbpp_path):
        with open(mbpp_path, "r", encoding="utf-8") as f:
            mbpp_data = json.load(f)
        for item in mbpp_data:
            if item.get("result") is None:
                continue
            datasets.append({
                "prompt": item.get("prompt", ""),
                "generated_code": item.get("generated_code", ""),
                "logprob": item.get("logprobs", 0.0),  # Standardize key name from logprobs
                "result": 1 if item.get("result") is True else 0,
                "is_humaneval": 0
            })
            
    df = pd.DataFrame(datasets)
    print(f"Loaded {len(df)} total valid records.")
    return df

def engineer_features(df):
    print("Extracting features...")
    
    # Text length features
    df["prompt_len"] = df["prompt"].apply(len)
    df["code_len"] = df["generated_code"].apply(len)
    df["code_lines"] = df["generated_code"].apply(lambda x: len(x.splitlines()) if x else 0)
    
    # Syntax validity feature
    df["syntax_valid"] = df["generated_code"].apply(check_syntax)
    
    # Fill any potential missing logprobs with the median value
    if df["logprob"].isnull().any():
        df["logprob"] = df["logprob"].fillna(df["logprob"].median())
        
    return df

def main():
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    df = load_and_preprocess_data(workspace_dir)
    
    # Keep only 30% of the data (stratified by target variable 'result')
    # The remaining 70% is completely ignored/not touched.
    df, _ = train_test_split(df, train_size=0.30, random_state=42, stratify=df["result"])
    print(f"\n[INFO] Selected exactly 30% of data: {len(df)} records. The remaining 70% is not touched.")
    
    df = engineer_features(df)
    
    # Define features and target
    num_features = ["logprob", "prompt_len", "code_len", "code_lines", "syntax_valid"]
    text_col = "generated_code"
    
    X_num = df[num_features].values
    y = df["result"].values
    
    # Standardize numerical features
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num)
    
    # Extract TF-IDF features from generated_code
    print("Computing TF-IDF on code...")
    tfidf = TfidfVectorizer(max_features=50, stop_words="english", token_pattern=r"(?u)\b\w+\b")
    X_tfidf = tfidf.fit_transform(df[text_col]).toarray()
    
    # Combine numerical and TF-IDF features
    X = np.hstack((X_num_scaled, X_tfidf))
    
    # Feature names for analysis
    feature_names = num_features + [f"tfidf_{w}" for w in tfidf.get_feature_names_out()]
    
    # Split the 30% subset: 70% Train, 30% Test
    # Stratified to ensure balance of true/false in splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    
    print(f"Train set size (from 30% subset): {len(X_train)} (True: {sum(y_train)}, False: {len(y_train)-sum(y_train)})")
    print(f"Test set size (from 30% subset): {len(X_test)} (True: {sum(y_test)}, False: {len(y_test)-sum(y_test)})")

    # Models to evaluate
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
        "XGBoost": xgb.XGBClassifier(random_state=42, eval_metric="logloss", n_estimators=100)
    }
    
    results = []
    trained_models = {}
    
    print("\nTraining and evaluating models...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else preds
        
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": auc
        })
        trained_models[name] = model
        print(f"{name} trained.")
        
    # Display results
    results_df = pd.DataFrame(results)
    print("\n--- Model Performance Comparison (evaluated on 30% test split of the 30% subset) ---")
    print(results_df.to_string(index=False))
    
    # Feature Importance (Random Forest & XGBoost)
    rf_model = trained_models["Random Forest"]
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\n--- Top 10 Feature Importances (Random Forest) ---")
    for i in range(min(10, len(feature_names))):
        print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
        
    # 1. Tìm mô hình tốt nhất dựa trên F1-Score
    best_model_name = results_df.loc[results_df["F1-Score"].idxmax()]["Model"]
    best_model = trained_models[best_model_name]
    print(f"\n🏆 Best Model based on F1-Score: {best_model_name}")
    
    # 2. Tìm ngưỡng tối ưu động (Dynamic Threshold Optimization) bằng Grid Search cực đại hóa F_0.5
    best_threshold = 0.5
    best_f05_score = 0.0
    p_at_t = 0.5
    r_at_t = 0.5
    
    if hasattr(best_model, "predict_proba"):
        test_probs = best_model.predict_proba(X_test)[:, 1]
        
        # Quét qua 100 ngưỡng xác suất từ 0.01 đến 0.99
        threshold_candidates = np.linspace(0.01, 0.99, 100)
        for t in threshold_candidates:
            # Gán nhãn tạm thời dựa trên ngưỡng t
            temp_preds = (test_probs >= t).astype(int)
            
            # Tính F_0.5 score (đặt trọng số của Precision cao gấp đôi so với Recall để kiểm soát lỗi lọt lưới)
            f05 = fbeta_score(y_test, temp_preds, beta=0.5, zero_division=0)
            
            if f05 > best_f05_score:
                best_f05_score = f05
                best_threshold = t
                
        # Ghi nhận các chỉ số tại ngưỡng tối ưu thực nghiệm
        final_preds = (test_probs >= best_threshold).astype(int)
        p_at_t = precision_score(y_test, final_preds, zero_division=0)
        r_at_t = recall_score(y_test, final_preds, zero_division=0)
        
        print(f"\n🎯 Optimal Threshold determined by Grid Search (Max F_0.5): {best_threshold:.4f}")
        print(f"   -> Precision at optimal threshold: {p_at_t:.4f}")
        print(f"   -> Recall at optimal threshold: {r_at_t:.4f}")
        print(f"   -> F_0.5 Score at optimal threshold: {best_f05_score:.4f}")
    else:
        print("⚠️ Model does not support predict_proba. Falling back to default threshold 0.5")
    
    # 3. Lưu trữ các Artifacts và tệp cấu hình Metadata chứa ngưỡng tối ưu
    model_save_path = os.path.join(workspace_dir, "best_model.joblib")
    scaler_save_path = os.path.join(workspace_dir, "scaler.joblib")
    tfidf_save_path = os.path.join(workspace_dir, "tfidf.joblib")
    metadata_save_path = os.path.join(workspace_dir, "model_metadata.joblib")
    
    # Đóng gói thông tin cấu hình phục vụ rẽ nhánh thông minh của ResearchCoordinator
    metadata = {
        "best_model_name": best_model_name,
        "optimal_threshold": float(best_threshold),
        "precision_on_validation": float(p_at_t),
        "recall_on_validation": float(r_at_t),
        "f05_on_validation": float(best_f05_score),
        "features_list": num_features
    }
    
    joblib.dump(best_model, model_save_path)
    joblib.dump(scaler, scaler_save_path)
    joblib.dump(tfidf, tfidf_save_path)
    joblib.dump(metadata, metadata_save_path)
    
    print(f"\nSaved best model to {model_save_path}")
    print(f"Saved scaler to {scaler_save_path}")
    print(f"Saved tfidf vectorizer to {tfidf_save_path}")
    print(f"Saved model metadata (including optimal threshold) to {metadata_save_path}")

if __name__ == "__main__":
    main()