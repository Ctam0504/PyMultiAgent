from .base_agent import BaseAgent
import re

class Coder(BaseAgent):
    def generate(self, prompt_context):
        full_prompt = f"""VAI TRÒ: Bạn là một Chuyên gia Lập trình Python CẤP CAO.
Bạn vừa bị Trọng tài (Reviewer) phê bình nghiêm khắc. 

### [1. DỮ LIỆU ĐỐI CHIẾU & BẢNG KIỂM TOÁN]
{prompt_context}

### [2. CHỈ THỊ CƯỠNG BÁCH]
1. ĐỐI CHIẾU BẢNG: Nếu trong [1] có 'BẢNG KIỂM TOÁN', hãy nhìn vào cột 'Kết luận' để thấy bước nào bạn làm 'Sai'.
2. TRÍCH XUẤT LỆNH: Tìm 'CHỈ THỊ' của Trọng tài. Đây là mệnh lệnh phải tuân thủ 100%.
3. TƯ DUY LẠI: Nếu mã nguồn cũ lặp lại lỗi cũ, bạn sẽ bị đào thải. Hãy thử một cách tiếp cận khác (ví dụ: dùng 2 vòng lặp thay vì 1, hoặc dùng `dict.get()`).

### [3. ĐỊNH DẠNG ĐẦU RA]
Trả về mã nguồn trong block ```python ... ```.
Comment 2 dòng đầu bắt buộc:
Tuyệt đối không sử dụng hàm input(). Dữ liệu đầu vào phải được nhận thông qua tham số của hàm.
# Sửa lỗi: [Tóm tắt ngắn gọn lỗi logic/thực thi đã xảy ra]
# Giải pháp: [Mô tả ngắn gọn thuật toán/cấu trúc mới để khắc phục]
"""
        response = self.llm.invoke(full_prompt)
        code = self.extract_code(response)
        return code.strip()

    def extract_code(self, text):
        """Trích xuất code từ Markdown linh hoạt hơn."""
        match = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            return match.group(1)
        # Fallback nếu model quên bọc tag
        return text.strip()