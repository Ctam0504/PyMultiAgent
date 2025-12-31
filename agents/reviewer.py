from .base_agent import BaseAgent
import re

class Reviewer(BaseAgent):
    def review(self, task_goal, code, test_cases="", history_feedback="", error_msg=""):
        # Chuẩn hóa thông tin lỗi từ Sandbox
        runtime_info = f"\n[KẾT QUẢ THỰC THI]:\n{error_msg}" if error_msg else "\n[TRẠNG THÁI]: Code chạy thành công nhưng cần kiểm tra tính đúng đắn logic."

        prompt = f"""VAI TRÒ: Bạn là Chuyên gia Thẩm định Logic (Logic Auditor) tối cao. 
THÁI ĐỘ: Tuyệt đối khách quan và nghiêm khắc. Nếu Tester đưa ra ví dụ sai toán học, bạn phải CHỈ TRÍCH trực diện lỗi đếm hoặc lỗi logic của Tester.

[YÊU CẦU GỐC]: {task_goal}
[MÃ NGUỒN CỦA CODER]: 
{code}
[BỘ KIỂM THỬ ĐANG CHẠY]:
{test_cases}
{runtime_info}

### QUY TRÌNH KIỂM TOÁN (CHIẾN THUẬT PHẢN BIỆN):
1. ĐỘC LẬP TÍNH TOÁN: Quên kết quả của Tester đi. Hãy tự đếm số ký tự, tự tính toán số liệu dựa trên từng Input. 
2. PHÂN TÍCH SAI LẦM: 
   - Nếu TARGET: TESTER, bạn phải chỉ rõ: "Tester đã sai khi khẳng định [Kết quả A] là đúng, trong khi thực tế toán học/logic yêu cầu [Kết quả B]".
   - Chỉ trích cụ thể hành vi "ngáo" (ảo giác) của Tester nếu nó vi phạm các quy tắc cơ bản (như đếm độ dài chuỗi).
3. ĐỐI CHIẾU 3 BÊN: (Logic bạn đã có rất tốt).

### ĐỊNH DẠNG ĐẦU RA BẮT BUỘC:
STATUS: REJECTED | TARGET: [CODER/TESTER/SYSTEM]
REASON: [Phải nêu rõ bằng chứng sai. Ví dụ: 'Tester sai logic đếm: Chuỗi "cd" có độ dài 2, theo đề bài 2 < 3 nên phải bị loại. Tester bắt giữ lại là sai hoàn toàn.']
CHỈ THỊ: [Lệnh cho Tester hoặc Coder phải làm gì tiếp theo]

BẢNG KIỂM TOÁN LOGIC (TỰ TÍNH TOÁN CẨN THẬN):
| Input | Bước thực hiện | Kết quả Đúng (Bắt buộc) | Kết quả Tester (Assert) | Sai lệch nằm ở đâu? |
| :--- | :--- | :--- | :--- | :--- |
"""
        raw_output = self.llm.invoke(prompt)
        return self.post_process_review(raw_output, error_msg)

    def post_process_review(self, text, error_msg):
        if not text: 
            return "STATUS: REJECTED | TARGET: SYSTEM | REASON: Model không phản hồi."
        
        # Làm sạch văn bản
        clean_text = re.sub(r'```.*?```', '', text, flags=re.DOTALL).strip()
        
        # 1. Xác định TARGET bằng logic ưu tiên phản biện
        target = "CODER" # Mặc định
        if "TARGET: TESTER" in clean_text.upper():
            target = "TESTER"
        elif "TARGET: SYSTEM" in clean_text.upper():
            target = "SYSTEM"
        
        # 2. Trích xuất REASON và Bảng kiểm toán
        reason = "Lỗi logic chưa xác định"
        r_match = re.search(r"REASON:?\s*(.*?)(?=CHỈ THỊ:|STATUS:|BẢNG|$)", clean_text, re.IGNORECASE | re.DOTALL)
        
        audit_table = ""
        if "BẢNG KIỂM TOÁN" in clean_text:
            audit_table = "\nBẢNG KIỂM TOÁN:" + clean_text.split("BẢNG KIỂM TOÁN")[1].strip()

        if r_match:
            reason = r_match.group(1).strip()
        
        # 3. Trích xuất CHỈ THỊ
        instruction = "Hãy kiểm tra lại logic dựa trên yêu cầu gốc."
        i_match = re.search(r"CHỈ THỊ:?\s*(.*?)(?=BẢNG KIỂM TOÁN:|STATUS:|$)", clean_text, re.IGNORECASE | re.DOTALL)
        if i_match:
            instruction = i_match.group(1).strip()

        # 4. Quyết định cuối cùng (PASSED chỉ khi không có lỗi thực thi và Judge duyệt)
        if "TARGET: TESTER" in target:
            return f"STATUS: REJECTED | TARGET: TESTER | REASON: {reason} {audit_table} | CHỈ THỊ: {instruction}"
        if "PASSED" in clean_text.upper() and not error_msg:
            return "PASSED"

        return f"STATUS: REJECTED | TARGET: {target} | REASON: {reason} {audit_table} | CHỈ THỊ: {instruction}"