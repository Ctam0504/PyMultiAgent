from .base_agent import BaseAgent
import re

class Tester(BaseAgent):
    # Cập nhật: Thêm tham số history_feedback để nhận "lời mắng" từ Judge
    def create_tests(self, task_goal, code_to_attack=None, mode="balanced", history_feedback=""):
        target_fn = re.search(r"(?:hàm|function|def)\s+([a-zA-Z_][a-zA-Z0-9_]*)", task_goal)
        fn_name = target_fn.group(1) if target_fn else "solution_func"

        # CẬP NHẬT PROMPT: Thêm phần xử lý lịch sử thất bại
        history_section = ""
        if history_feedback:
            history_section = f"""
### [BÀI HỌC TỪ THẤT BẠI TRƯỚC ĐÓ]
Trọng tài (Judge) đã bác bỏ bộ test trước của bạn vì sai logic. 
Chi tiết lỗi: 
{history_feedback}
=> BẠN PHẢI: Đọc kỹ lý do sai, tính toán lại từng bước và KHÔNG lặp lại sai lầm này.
"""

        prompt = f"""VAI TRÒ: Chuyên gia Kiểm thử Hệ thống.
NHIỆM VỤ: Tạo bộ test case cho hàm `{fn_name}`.

{history_section}

### [NGUYÊN TẮC VÀNG ĐỂ KHÔNG BAO GIỜ SAI LOGIC]
LLM thường tính toán sai các con số cụ thể. Để khắc phục, bạn PHẢI tuân thủ:
1. KHÔNG GHI KẾT QUẢ TÍNH NHẨM: Thay vì ghi `== 45.0`, hãy ghi biểu thức `== round(50 * 0.9, 2)`.
2. SỬ DỤNG BIỂU THỨC PYTHON: Vế phải của `assert` nên là biểu thức máy tính tự tính được.
3. TẬP TRUNG VÀO LOGIC: Nếu đề bài yêu cầu lọc mã >= 500, hãy đảm bảo các giá trị nhỏ hơn 500 phải trả về kết quả rỗng.

### [YÊU CẦU CỤ THỂ]
Đề bài: {task_goal}
Chiến thuật ({mode}): {self.get_distribution(mode)}

TRẢ VỀ CODE TRONG THẺ <test>...</test>. 
Mỗi dòng là một `assert` độc lập.
""" 
        raw_output = self.llm.invoke(prompt) 
        return self.post_process_tests(raw_output, fn_name)

    def get_distribution(self, mode):
        return {
            "valid": "Input thông thường.",
            "noise": "Input cực đoan (None, rỗng, số âm, kiểu dữ liệu lạ).",
            "balanced": "Kết hợp cả hai."
        }.get(mode, "balanced")

    def post_process_tests(self, text, target_fn): 
        test_block = re.search(r'<test>(.*?)</test>', text, re.DOTALL)
        content = test_block.group(1) if test_block else text
        
        lines = content.split('\n')
        final_asserts = []
        for line in lines:
            line = line.strip()
            # Bảo vệ cú pháp: Đảm bảo assert không bị lỗi thụt đầu dòng hoặc rác
            if line.startswith('assert') and f'{target_fn}(' in line:
                # Tự động thêm làm tròn cho các biểu thức số thực để tránh lỗi 0.0000000001
                if '/' in line or '.' in line:
                    # Nếu AI quên round, ta có thể gợi ý hoặc để Judge soi, 
                    # nhưng tốt nhất là prompt đã ép nó ghi biểu thức.
                    pass
                final_asserts.append(line)

        return '\n'.join(list(dict.fromkeys(final_asserts))[:7]).strip()