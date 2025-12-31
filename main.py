import os
import re
import datetime
import subprocess
import gc
import time
from agents.coder import Coder
from agents.tester import Tester
from agents.reviewer import Reviewer
from utils.sandbox import Sandbox

class ResearchCoordinator:
    def __init__(self):
        self.coder = Coder(model_name="deepseek-coder-v2")
        self.adversary = Tester(model_name="gemma3:12b")
        self.judge = Reviewer(model_name="Qwen2.5-Coder")
        
        self.best_code = ""
        self.golden_memory = []  
        self.history_logs = ""
        self.event_log = []

    def clear_vram(self):
        try:
            subprocess.run(["ollama", "stop", "deepseek-coder-v2"], capture_output=True)
            subprocess.run(["ollama", "stop", "gemma3:12b"], capture_output=True)
            subprocess.run(["ollama", "stop", "Qwen2.5-Coder"], capture_output=True)
            gc.collect()
            time.sleep(0.5) 
        except:
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

        report = f"""# 🔬 BÁO CÁO NGHIÊN CỨU ĐỐI ĐẦU
- **Thời gian:** {datetime.datetime.now().strftime("%H:%M:%S %d/%m/%Y")}
- **Kết quả:** {"THÀNH CÔNG" if self.best_code else "THẤT BẠI"}

| Thuật toán | Vòng | Chế độ | Trạng thái | Ghi chú chi tiết |
| :--- | :--- | :--- | :--- | :--- |
{table_rows}

## 🏆 Mã nguồn tốt nhất
```python
{self.best_code if self.best_code else "# N/A"}
""" 
        with open(filename, "w", encoding="utf-8") as f: f.write(report) 
        print(f"📊 Đã lưu báo cáo tại: {filename}")
    def run_research_cycle(self, max_attempts=10, stability_threshold=7):
        print("\n" + "═"*60 + "\n🚀 ADVERSARIAL RESEARCH: LOGIC-FIRST MODE\n" + "═"*60)
        task_goal = input("👉 NHẬP YÊU CẦU BÀI TOÁN: ")
        target_fn = self.extract_function_name(task_goal)
        
        last_code_hash = ""
        repeat_count = 0
        self.history_logs = "" 

        for attempt in range(1, max_attempts + 1):
            self.clear_vram() 
            print(f"\n{'━'*15} THỬ NGHIỆM THUẬT TOÁN {attempt}/{max_attempts} {'━'*15}")
            
            # 1. CODER GENERATION
            c_prompt = f"### [NHIỆM VỤ]: {task_goal}\n### [TÊN HÀM]: {target_fn}\n"
            if self.history_logs:
                c_prompt += f"\n### [BẰNG CHỨNG THẤT BẠI VÀ CHỈ THỊ TRƯỚC ĐÓ]:\n{self.history_logs}"
            
            new_code = self.coder.generate(c_prompt)
            self.best_code = new_code
            if not new_code: continue
            
            print(f"\n📝 [MÃ NGUỒN ATTEMPT {attempt}]:\n{'-' * 40}\n{new_code}\n{'-' * 40}")

            # Kiểm tra trùng lặp code
            import hashlib
            current_hash = hashlib.md5(new_code.encode()).hexdigest()
            repeat_count = (repeat_count + 1) if current_hash == last_code_hash else 0
            last_code_hash = current_hash

            current_cycle_success = True
            temp_tests = []
            
            i = 1
            max_tester_retries = 3 # Giới hạn số lần Tester được phép sai logic trong 1 vòng
            tester_retry_count = 0

            while i <= stability_threshold:
                self.clear_vram() 
                mode = "valid" if i == 1 else ("balanced" if i < 4 else "extreme")
                
                print(f"\n🧪 [VÒNG {i}] Chế độ: {mode} (Tester retry: {tester_retry_count})")
                test_cases = self.adversary.create_tests(task_goal, code_to_attack=new_code, mode=mode,history_feedback=self.history_logs)
                print(f"\n🧪 [BỘ TEST - VÒNG {i}]:")
                print("-" * 40)
                print(test_cases)
                print("-" * 40)
                if not test_cases: 
                    i += 1
                    continue

                # 2. SANDBOX EXECUTION
                all_tests = "\n".join(self.golden_memory + [test_cases])
                success, result = Sandbox.run(new_code, all_tests)
                
                if success:
                    self.log_event(attempt, i, mode, "✅ PASS", f"Vượt qua bộ test vòng {i}")
                    if test_cases not in temp_tests: temp_tests.append(test_cases)
                    i += 1
                    tester_retry_count = 0 # Reset retry khi thành công
                else:
                    full_runtime_error = result.strip()
                    
                    # 3. JUDGE REVIEW (TRỌNG TÀI PHÂN XỬ)
                    print(f"⚖️ [JUDGE] Đang thẩm định mâu thuẫn giữa Coder và Tester...")
                    judge_feedback = self.judge.review(
                        task_goal=task_goal, 
                        code=new_code, 
                        test_cases=test_cases, 
                        history_feedback=self.history_logs,
                        error_msg=full_runtime_error
                    )
                    
                    # PHÂN LUỒNG DỰA TRÊN QUYẾT ĐỊNH CỦA JUDGE
                    if "TARGET: TESTER" in judge_feedback.upper():
                        self.log_event(attempt, i, mode, "⚠️ TESTER WRONG", "Judge bác bỏ bộ test vì sai logic đề bài.")
                        tester_retry_count += 1
                        self.history_logs = f"""
⚠️ [CẢNH BÁO CHO TESTER - VÒNG {i}]: 
Bộ test trước của bạn đã bị bác bỏ bởi Trọng tài vì SAI LOGIC.
[LÝ DO BÁC BỎ]: {judge_feedback}
=> Yêu cầu: Tính toán lại cẩn thận và tạo bộ test mới chính xác hơn.
"""
                        if tester_retry_count > max_tester_retries:
                            self.log_event(attempt, i, mode, "🏆 FORCED SUCCESS", "Tester quá lì lợm. Chốt mã nguồn này là bản tốt nhất!")
                            print(f"\n🔥 [QUYẾT ĐỊNH CUỐI CÙNG] Tester sai quá 3 lần. Dừng quy trình và lấy mã nguồn hiện tại.")
                            self.best_code = new_code
                            current_cycle_success = True
                            for t in temp_tests:
                                if t not in self.golden_memory: self.golden_memory.append(t)
                            break    
                        else:
                            self.log_event(attempt, i, mode, "⚠️ TESTER WRONG", f"Bác bỏ bộ test (Lần {tester_retry_count})")
                            continue
                    elif "TARGET: SYSTEM" in judge_feedback.upper():
                        self.log_event(attempt, i, mode, "🚨 SYSTEM", "Lỗi môi trường/hệ thống.")
                        i += 1; continue
                    
                    else: # TARGET: CODER
                        short_error = full_runtime_error.splitlines()[-1] if full_runtime_error else "AssertionError"
                        self.log_event(attempt, i, mode, "❌ REJECT CODER", f"Phát hiện lỗi logic tại Coder.")
                        
                        # Lưu log để Coder sửa ở Attempt sau
                        self.history_logs = f"""
--- BẰNG CHỨNG THẤT BẠI TẠI ATTEMPT {attempt} ---
[TEST CASE GÂY LỖI]:
{test_cases}
[PHÂN TÍCH CỦA TRỌNG TÀI]:
{judge_feedback}
---------------------------------------------
"""
                        current_cycle_success = False 
                        break # Coder thất bại hoàn toàn vòng này, sang Attempt mới

            if current_cycle_success:
                print(f"\n🏆 [HỘI TỤ] Thuật toán đã vượt qua mọi thử thách tại vòng {attempt}!")
                self.best_code = new_code
                for t in temp_tests:
                    if t not in self.golden_memory: self.golden_memory.append(t)
                
                self.save_research_report()
                with open("solution.py", "w", encoding="utf-8") as f: f.write(self.best_code)
                return

        print(f"\n⚠️ KẾT THÚC: Không tìm thấy lời giải tối ưu sau {max_attempts} lần thử.")
        with open("solution.py", "w", encoding="utf-8") as f: 
            f.write(self.best_code)
            
        print("💾 Đã ép buộc ghi mã nguồn cuối cùng vào solution.py")
        self.save_research_report()
if __name__ == "__main__": 
    coordinator = ResearchCoordinator() 
    coordinator.run_research_cycle(max_attempts=10, stability_threshold=7)