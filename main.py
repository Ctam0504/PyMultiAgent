import os
import sys

# Đảm bảo in ký tự tiếng Việt / emoji trên Windows console không bị lỗi cp1252
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

os.environ["OLLAMA_FLASH_ATTENTION"] = "1"

import multiprocessing
# Ép buộc sử dụng 'spawn' để tránh xung đột CUDA/Torch và Ollama Sockets
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

import re
import datetime
import subprocess
import time
import hashlib
import joblib
import numpy as np
import ast
import argparse
import gc
import torch

from agents.coder import Coder
from agents.tester import Tester
from agents.reviewer import Reviewer
from utils.sandbox import DockerSandbox

class ResearchCoordinator:
    def __init__(self, mode="coder_tester_reviewer_classifier"):
        """
        mode:
          - 'coder'                            : Chỉ Coder, không lặp, không tester/reviewer
          - 'coder_tester'                     : Coder + Tester, không lặp, không reviewer
          - 'coder_tester_reviewer'            : Coder + Tester + Reviewer, có lặp, không classifier
          - 'coder_tester_reviewer_classifier' : Coder + Tester + Reviewer + Classifier (alias: 'full')
        """
        raw_mode = mode.lower().strip()
        # 'full' và 'code_tester_reviewer_classifier' chính là 'coder_tester_reviewer_classifier'
        if raw_mode in ["full", "code_tester_reviewer_classifier"]:
            self.mode = "coder_tester_reviewer_classifier"
        else:
            self.mode = raw_mode

        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

        # Khởi tạo các Agent tùy theo mode
        self.coder = Coder(model_name="qwen2.5-coder:7b", num_predict=4096, num_ctx=16384, base_url=base_url)

        if self.mode in ["coder_tester", "coder_tester_reviewer", "coder_tester_reviewer_classifier"]:
            self.adversary = Tester(model_name="qwen2.5-coder:7b", num_predict=4096, num_ctx=16384, base_url=base_url)
        else:
            self.adversary = None

        if self.mode in ["coder_tester_reviewer", "coder_tester_reviewer_classifier"]:
            self.judge = Reviewer(model_name="qwen2.5-coder:7b", num_predict=4096, num_ctx=16384, base_url=base_url)
        else:
            self.judge = None

        self.best_code = ""
        self.golden_memory = []
        self.coder_history = ""
        self.latest_code = ""
        self.event_log = []
        
        self._load_classifier()

    def _load_classifier(self):
        try:
            ml_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agents", "machinelearning")
            pt_path = os.path.join(ml_dir, "best_model.pt")
            
            # 1. Ưu tiên nạp định dạng PyTorch checkpoint mới (.pt)
            if os.path.exists(pt_path):
                checkpoint = torch.load(pt_path, map_location="cpu", weights_only=False)
                prep = checkpoint.get("preprocessor", {})
                self.selected_features = prep.get("selected_features", [])
                self.scaler_mean = np.array(prep.get("scaler_mean", []), dtype=np.float32)
                self.scaler_scale = np.array(prep.get("scaler_scale", []), dtype=np.float32)
                self.classifier_logprob_median = float(prep.get("logprob_median", -0.075))
                self.optimal_threshold = float(checkpoint.get("best_threshold", 0.56))
                model_name = checkpoint.get("best_model_name", "XGBoost_Tuned_Optuna")

                if "xgboost_dump" in checkpoint:
                    import xgboost as xgb
                    self.classifier_booster = xgb.Booster()
                    self.classifier_booster.load_model(bytearray(checkpoint["xgboost_dump"], "utf-8"))
                    self.classifier_type = "pt_xgboost"
                    self.has_classifier = True
                    print(f"🔮 Đã nạp Classifier (.pt - {model_name}) thành công. Ngưỡng tối ưu: {self.optimal_threshold:.4f} | Số đặc trưng: {len(self.selected_features)}")
                    return

            # 2. Fallback sang định dạng joblib cũ nếu có
            model_path = os.path.join(ml_dir, "best_model.joblib")
            scaler_path = os.path.join(ml_dir, "scaler.joblib")
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.classifier_model = joblib.load(model_path)
                self.classifier_scaler = joblib.load(scaler_path)

                metadata_path = os.path.join(ml_dir, "model_metadata.joblib")
                if os.path.exists(metadata_path):
                    metadata = joblib.load(metadata_path)
                    self.optimal_threshold = float(metadata.get("optimal_threshold", 0.80))
                    self.classifier_logprob_median = float(metadata.get("logprob_median", 0.0))
                    version = metadata.get("version", "v1")
                    print(f"🔮 Đã nạp Classifier {version} (.joblib) thành công. Ngưỡng tối ưu: {self.optimal_threshold:.4f}")
                else:
                    self.optimal_threshold = 0.80
                    self.classifier_logprob_median = 0.0
                self.classifier_type = "joblib"
                self.has_classifier = True
                return

            self.has_classifier = False
            self.optimal_threshold = 0.56
            self.classifier_logprob_median = 0.0
        except Exception as e:
            print(f"⚠️ Không thể nạp mô hình học máy: {e}. Hệ thống sử dụng luồng Multi-Agent mặc định.")
            self.has_classifier = False
            self.optimal_threshold = 0.56
            self.classifier_logprob_median = 0.0

    @staticmethod
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

    @classmethod
    def _extract_all_code_features(cls, code: str) -> dict:
        """Trích xuất toàn diện 18 đặc trưng cú pháp và độ phức tạp từ mã nguồn."""
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
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow, ast.FloorDiv,
            ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd,
            ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
            ast.Is, ast.IsNot, ast.In, ast.NotIn,
            ast.And, ast.Or, ast.Not, ast.Invert, ast.UAdd, ast.USub,
            ast.Assign, ast.AugAssign, ast.AnnAssign
        )
        BRANCH_NODES = (ast.If, ast.IfExp, getattr(ast, 'Match', type(None)))
        LOOP_NODES = (ast.For, ast.While, ast.AsyncFor,
                      ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)
        STRUCT_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        IMPORT_NODES = (ast.Import, ast.ImportFrom)
        CC_NODES = (ast.If, ast.For, ast.While, ast.ExceptHandler, ast.BoolOp, ast.IfExp)

        ops = branch = loops = structs = returns = calls = imports_ = comparisons = 0
        cc = 1
        identifiers = []

        for node in ast.walk(tree):
            if isinstance(node, OP_NODES):       ops += 1
            elif isinstance(node, BRANCH_NODES): branch += 1
            elif isinstance(node, LOOP_NODES):   loops += 1
            elif isinstance(node, STRUCT_NODES): structs += 1

            if isinstance(node, ast.Return):     returns += 1
            if isinstance(node, ast.Try):        feats["has_try_except"] = 1
            if isinstance(node, ast.Call):       calls += 1
            if isinstance(node, IMPORT_NODES):   imports_ += 1
            if isinstance(node, ast.Compare):    comparisons += 1
            if isinstance(node, CC_NODES):       cc += 1
            if isinstance(node, ast.Name):       identifiers.append(node.id)
            elif isinstance(node, ast.arg):      identifiers.append(node.arg)

        feats["Ops"] = ops
        feats["Branch"] = branch
        feats["Loops"] = loops
        feats["Structs"] = structs
        feats["cyclomatic"] = cc
        feats["max_depth"] = cls._nesting_depth(tree)
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

    @staticmethod
    def _extract_ast_features(code: str) -> dict:
        """Hàm trích xuất AST cũ (giữ tương thích ngược)."""
        zero = {"Ops": 0, "Branch": 0, "Loops": 0, "Structs": 0}
        if not code:
            return zero
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return zero

        ops = branch = loops = structs = 0
        OP_NODES = (
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow, ast.FloorDiv,
            ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd,
            ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
            ast.Is, ast.IsNot, ast.In, ast.NotIn,
            ast.And, ast.Or, ast.Not, ast.Invert, ast.UAdd, ast.USub,
            ast.Assign, ast.AugAssign, ast.AnnAssign
        )
        BRANCH_NODES = (ast.If, ast.IfExp, getattr(ast, 'Match', type(None)))
        LOOP_NODES = (ast.For, ast.While, ast.AsyncFor,
                      ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)
        STRUCT_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        for node in ast.walk(tree):
            if isinstance(node, OP_NODES):      ops += 1
            elif isinstance(node, BRANCH_NODES): branch += 1
            elif isinstance(node, LOOP_NODES):   loops += 1
            elif isinstance(node, STRUCT_NODES): structs += 1
        return {"Ops": ops, "Branch": branch, "Loops": loops, "Structs": structs}

    def predict_single_agent(self, prompt, code, logprob=0.0):
        if not getattr(self, 'has_classifier', False):
            return False, 0.0

        lp = logprob if (logprob is not None and logprob != 0.0) else self.classifier_logprob_median

        # 1. Sử dụng mô hình mới từ file PyTorch (.pt)
        if getattr(self, "classifier_type", None) == "pt_xgboost" and hasattr(self, "classifier_booster"):
            feats = self._extract_all_code_features(code)
            feats["logprob"] = lp
            feats["logprob_per_line"] = lp / max(feats.get("code_lines", 1), 1)

            # Lấy vector đặc trưng theo đúng các cột đã qua tuyển chọn ở Bước 3
            feat_vector = np.array([[feats.get(c, 0.0) for c in self.selected_features]], dtype=np.float32)

            # Chuẩn hóa StandardScaler
            if hasattr(self, "scaler_mean") and hasattr(self, "scaler_scale"):
                scale_denom = np.where(self.scaler_scale == 0, 1.0, self.scaler_scale)
                feat_vector = (feat_vector - self.scaler_mean) / scale_denom

            import xgboost as xgb
            dmat = xgb.DMatrix(feat_vector)
            prob = float(self.classifier_booster.predict(dmat)[0])
            is_single = (prob >= self.optimal_threshold)
            return is_single, prob

        # 2. Fallback cho mô hình legacy (joblib) nếu có
        code_len = len(code) if code else 0
        ast_feats = self._extract_ast_features(code)
        X = np.array([[lp, code_len,
                       ast_feats["Ops"], ast_feats["Branch"],
                       ast_feats["Loops"], ast_feats["Structs"]]])
        prob = 0.0
        if hasattr(self, "classifier_scaler") and hasattr(self, "classifier_model"):
            X_scaled = self.classifier_scaler.transform(X)
            if hasattr(self.classifier_model, "predict_proba"):
                prob = float(self.classifier_model.predict_proba(X_scaled)[0, 1])

        is_single = (prob >= self.optimal_threshold)
        return is_single, prob

    def clear_vram(self):
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def extract_function_name(self, task_goal):
        match = re.search(r"(?:hàm|function|def)\s+([a-zA-Z_][a-zA-Z0-9_]*)", task_goal)
        return match.group(1) if match else "solution_func"

    def log_event(self, attempt, cycle, mode, status, note):
        print(f"   [{mode.upper()}] {status}: {note}")
        self.event_log.append({
            "attempt": attempt, "cycle": cycle, "mode": mode,
            "status": status, "note": note
        })

    def save_research_report(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"research_report_{timestamp}.md"
        table_rows = ""
        for e in self.event_log:
            safe_note = str(e['note']).replace('\n', ' ').replace('|', '-')
            table_rows += f"| {e['attempt']} | {e['cycle']} | {e['mode']} | {e['status']} | {safe_note} |\n"

        report = f"""# 🔬 BÁO CÁO NGHIÊN CỨU BIẾN HÌNH (METAMORPHIC)
- **Thời gian:** {datetime.datetime.now().strftime("%H:%M:%S %d/%m/%Y")}
- **Chế độ chạy:** {self.mode.upper()}
- **Kết quả:** {"THÀNH CÔNG" if self.best_code else "THẤT BẠI"}

| Thử nghiệm | Vòng MR | Chế độ | Trạng thái | Ghi chú chi tiết |
| :--- | :--- | :--- | :--- | :--- |
{table_rows}

## 🏆 Mã nguồn cuối cùng
```python
{self.best_code if self.best_code else "# N/A"}
"""
        with open(filename, "w", encoding="utf-8") as f: f.write(report)
        print(f"📊 Đã lưu báo cáo tại: {filename}")
    def solve_task(self, task_goal, entry_point=None, max_attempts=5, stability_threshold=3):
        task_start_time = time.time()
        target_fn = entry_point if entry_point else self.extract_function_name(task_goal)
        
        last_code_hash = ""
        self.coder_history = "" 
        self.latest_code = ""
        self.best_code = ""
        self.golden_memory = []
        self.event_log = []

        # "coder" is intentionally single-pass. The other workflows need
        # retries because they use tester/reviewer feedback to improve code.
        attempt_limit = 1 if self.mode == "coder" else max_attempts
        last_classifier_prob = None
        last_classifier_decision = None

        for attempt in range(1, attempt_limit + 1):
            self.clear_vram()
            print(f"\n{'━'*15} THỬ NGHIỆM THUẬT TOÁN {attempt}/{attempt_limit} (MODE: {self.mode.upper()}) {'━'*15}")
            
            # 1. CODER GENERATION
            c_prompt = f"### [NHIỆM VỤ]: {task_goal}\n### [TÊN HÀM]: {target_fn}\n"
            c_prompt += "LƯU Ý: Code sẽ được kiểm tra bằng Metamorphic Testing.\n"
            if self.coder_history:
                if self.latest_code:
                    c_prompt += f"\n### [MÃ NGUỒN CŨ CẦN SỬA LỖI]:\n```python\n{self.latest_code}\n```\n"
                c_prompt += f"\n### [PHÂN TÍCH LỖI VÀ CHỈ THỊ SỬA TỪ REVIEWER]:\n{self.coder_history}\n"
                c_prompt += "\nCHỈ THỊ: Đọc kỹ phân tích lỗi và sửa dứt điểm lỗi logic trong mã nguồn cũ."
            
            new_code, real_logprob = self.coder.generate_with_logprob(c_prompt)
            if not new_code: 
                continue
                
            self.latest_code = new_code
            print(f"\n📝 [MÃ NGUỒN SINH BỞI CODER]:\n{'-'*40}\n{new_code}\n{'-'*40}")
            
            current_hash = hashlib.md5(new_code.encode()).hexdigest()
            if self.mode != "coder" and current_hash == last_code_hash:
                print("⚠️ Phát hiện trùng mã nguồn cũ. Ép buộc Coder thay đổi hướng tiếp cận...")
                new_code = self.coder.generate(c_prompt + "\nCHỈ THỊ: Thử một hướng tiếp cận logic hoàn toàn khác biệt.")
                if new_code:
                    self.latest_code = new_code
                real_logprob = 0.0
            last_code_hash = current_hash

            if self.mode == "coder":
                print("\n⚙️ [MODE: CODER] Thực chạy Sanity Check trực tiếp...")
                success, result = DockerSandbox.run(new_code, "")
                if success:
                    self.log_event(attempt, 0, "single_coder", "✅ PASS SANITY", "Mã nguồn thực thi thành công.")
                    self.best_code = new_code
                    elapsed = time.time() - task_start_time
                    return {
                        "best_code": self.best_code, "success": True, 
                        "mode": "coder", "attempts": attempt, "execution_time": elapsed,
                        "classifier_prob": None, "classifier_decision": None
                    }
                else:
                    self.log_event(attempt, 0, "single_coder", "❌ FAIL SANITY", f"Lỗi: {result.strip()}")
                    self.coder_history = f"Lỗi runtime: {result.strip()}"
                    elapsed = time.time() - task_start_time
                    return {
                        "best_code": self.latest_code,
                        "success": False,
                        "mode": "coder",
                        "attempts": attempt,
                        "execution_time": elapsed,
                        "classifier_prob": None, "classifier_decision": None
                    }

            # 2. CLASSIFICATION DECISION (chế độ 'coder_tester_reviewer_classifier')
            is_single, prob = False, 0.0
            if self.mode == "coder_tester_reviewer_classifier":
                is_single, prob = self.predict_single_agent(task_goal, new_code, logprob=real_logprob)
                prob_str = f"{prob:.4f}" if prob is not None else 'N/A'
                last_classifier_prob = prob

                if is_single:
                    last_classifier_decision = "single_agent"
                    print(f"\n🔮 [CLASSIFIER] Đánh giá: THÀNH CÔNG (Xác suất: {prob_str} >= Ngưỡng: {self.optimal_threshold:.4f}) -> Chuyển sang Single Agent (kết thúc ngay, không lặp Multi-Agent).")
                    success, result = DockerSandbox.run(new_code, "")
                    elapsed = time.time() - task_start_time
                    if success:
                        self.log_event(attempt, 0, "single_agent", "✅ PASS SANITY", "Vượt qua kiểm tra thực thi cơ bản.")
                        self.best_code = new_code
                        return {
                            "best_code": self.best_code, "success": True,
                            "mode": "single_agent", "attempts": attempt, "execution_time": elapsed,
                            "classifier_prob": last_classifier_prob, "classifier_decision": "single_agent"
                        }
                    else:
                        self.log_event(attempt, 0, "single_agent", "❌ FAIL SANITY", f"Thực thi lỗi: {result.strip()}")
                        self.coder_history = f"Lỗi runtime: {result.strip()}"
                        return {
                            "best_code": self.latest_code, "success": False,
                            "mode": "single_agent", "attempts": attempt, "execution_time": elapsed,
                            "classifier_prob": last_classifier_prob, "classifier_decision": "single_agent"
                        }
                else:
                    last_classifier_decision = "multi_agent"
                    print(f"\n🔮 [CLASSIFIER] Đánh giá: CHƯA ĐẠT ĐỘ TIN CẬY (Xác suất: {prob_str} < Ngưỡng: {self.optimal_threshold:.4f}) -> Kích hoạt Multi-Agent (Tester & Reviewer).")

            # 3. VERIFICATION LOOP
            current_cycle_success = True
            temp_mr_list = []
            i = 1
            
            while i <= stability_threshold:
                self.clear_vram()
                mode_str = "balanced" if i == 1 else "extreme"
                print(f"\n🧪 [VÒNG MR {i}] Tạo kịch bản kiểm thử...")
                
                already_tested = "\n---\n".join(self.golden_memory) if self.golden_memory else ""
                tester_ctx = f"\n[CÁC KỊCH BẢN ĐÃ PASS VÒNG TRƯỚC]:\n{already_tested}" if already_tested else ""
                
                mr_cases = self.adversary.create_tests(
                    task_goal, code_to_attack=new_code, mode=mode_str, history_feedback=tester_ctx
                )
                
                if not mr_cases or "assert " not in mr_cases:
                    self.log_event(attempt, i, mode_str, "🚨 TESTER FAULTY", "Thiếu assertion.")
                    i += 1
                    continue

                mr_cases = mr_cases.strip()
                print(f"\n🔍 [MR TEST CASE VÒNG {i}]:\n{mr_cases}")
                
                all_mr_tests = "\n\n".join(self.golden_memory + [mr_cases])
                success, result = DockerSandbox.run(new_code, all_mr_tests)
                
                if success:
                    self.log_event(attempt, i, mode_str, "✅ PASS MR", "Mã nguồn vượt qua MR.")
                    if mr_cases not in temp_mr_list: 
                        temp_mr_list.append(mr_cases)
                    i += 1
                else:
                    if self.mode == "coder_tester" or not self.judge:
                        self.log_event(attempt, i, mode_str, "❌ FAIL MR", "Trượt test case (Không có Reviewer thẩm định).")
                        self.coder_history = f"Fail tại test case:\n{mr_cases}\nLỗi:\n{result}"
                        current_cycle_success = False
                        break
                    
                    print(f"⚖️ [JUDGE] Reviewer tiến hành thẩm định lỗi...")
                    judge_feedback = self.judge.review(
                        task_goal=task_goal, code=new_code, test_cases=mr_cases, error_msg=result if result else "N/A"
                    )
                    print(f"\n📝 [FEEDBACK TỪ REVIEWER]:\n{'-'*40}\n{judge_feedback}\n{'-'*40}\n")
                    if "TARGET: TESTER" in judge_feedback.upper():
                        self.log_event(attempt, i, mode_str, "🚨 TESTER FAULTY", "Reviewer xác định Tester viết sai test case.")
                        i += 1
                    else:
                        self.log_event(attempt, i, mode_str, "❌ REJECT CODER", "Reviewer xác nhận lỗi do Coder.")
                        self.coder_history = f"--- THẤT BẠI TẠI MR VÒNG {i} ---\n{judge_feedback}\n"
                        current_cycle_success = False 
                        break

            if current_cycle_success:
                elapsed = time.time() - task_start_time
                self.best_code = new_code
                for mr in temp_mr_list:
                    if mr not in self.golden_memory: self.golden_memory.append(mr)
                return {
                    "best_code": self.best_code, "success": True, 
                    "mode": self.mode, "attempts": attempt, "execution_time": elapsed,
                    "classifier_prob": last_classifier_prob, "classifier_decision": last_classifier_decision
                }

        elapsed = time.time() - task_start_time
        final_code = self.best_code if self.best_code else self.latest_code
        return {
            "best_code": final_code, "success": False, 
            "mode": self.mode, "attempts": attempt_limit, "execution_time": elapsed,
            "classifier_prob": last_classifier_prob, "classifier_decision": last_classifier_decision
        }

    def run_research_cycle(self, task_goal, max_attempts=5, stability_threshold=3):
        print("\n" + "═"*60 + f"\n🚀 METAMORPHIC TESTING RUNNER (MODE: {self.mode.upper()})\n" + "═"*60)
        res = self.solve_task(task_goal, max_attempts=max_attempts, stability_threshold=stability_threshold)
        self.save_research_report()
        if res.get("best_code"):
            with open("solution.py", "w", encoding="utf-8") as f: 
                f.write(res["best_code"])
        return res
def restart_ollama_service():
    """Hàm riêng để khởi động Ollama 1 lần duy nhất từ Terminal CLI."""
    print("⚡ Khởi động lại dịch vụ Ollama với Flash Attention...")
    try:
        subprocess.run(["ollama", "stop"], capture_output=True, timeout=10)
        time.sleep(1)
        env = os.environ.copy()
        env.update(
            {
                "OLLAMA_FLASH_ATTENTION": "1",
                "OLLAMA_KV_CACHE_TYPE": "f16",
                "OLLAMA_MAX_LOADED_MODELS": "1",
                "OLLAMA_NUM_PARALLEL": "2",
            }
        )
        subprocess.Popen(
        ["ollama", "serve"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
        )
        time.sleep(3)
        print("✅ Đã tối ưu cấu hình Ollama thành công.")
    except Exception as e:
        print(f"⚠️ Không thể khởi động lại Ollama: {e}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Agent Metamorphic Testing Coordinator")
    parser.add_argument("--mode", "-m", type=str,
                        choices=["coder", "coder_tester", "coder_tester_reviewer", 
                                   "full"],
                        default="full",
                        help="Chế độ chạy (full là tên viết tắt của coder_tester_reviewer_classifier)")
    parser.add_argument("--task", "-t", type=str, default="", help="Yêu cầu bài toán (Prompt)")
    parser.add_argument("--attempts", type=int, default=5, help="Số lần thử tối đa")
    parser.add_argument("--stability", type=int, default=3, help="Ngưỡng vòng lặp MR thành công")

    args = parser.parse_args()

    # Chỉ restart Ollama khi chạy trực tiếp file main.py từ CLI
    restart_ollama_service()

    task_goal = args.task
    if not task_goal:
        task_goal = input("👉 NHẬP YÊU CẦU BÀI TOÁN: ")

    coordinator = ResearchCoordinator(mode=args.mode)
    coordinator.run_research_cycle(
        task_goal=task_goal, 
        max_attempts=args.attempts, 
        stability_threshold=args.stability
    )