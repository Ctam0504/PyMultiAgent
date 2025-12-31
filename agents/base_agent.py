from langchain_ollama import OllamaLLM
import re

class BaseAgent:
    def __init__(self, model_name="deepseek-coder-v2"):
        self.llm = OllamaLLM(
            model=model_name, 
            temperature=0.1,
            num_predict=1024,
            num_ctx=3072,     # Giới hạn context để ổn định VRAM
            keep_alive=0,     # Giải phóng bộ nhớ ngay lập tức
            base_url="http://localhost:11434",
            additional_kwargs={
                "num_gpu": 20, 
                "mirostat": 0,
            },
            stop=[
                "<｜begin of sentence｜>", "<｜end of sentence｜>", 
                "User:", "Assistant:", "### [1.", "### [NHIỆM VỤ GỐC]"
            ]
        )

    def extract_code(self, text):
        if not text: return ""
        
        # 1. Làm sạch ký tự rác MoE
        bad_chars = ['｜', '▁', '◁', '▷', '█', '░', '▒', '▓']
        for char in bad_chars:
            text = text.replace(char, '')
        
        # 2. Xóa System Tokens
        text = re.sub(r'<\|.*?\|>', '', text)
        
        # 3. Trích xuất Block Code (Ưu tiên block cuối cùng)
        code_blocks = re.findall(r'```(?:python|py)?\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
        
        if code_blocks:
            clean_code = code_blocks[-1].strip()
        else:
            # Fallback: Tìm từ 'def ' cuối cùng đến hết
            match = list(re.finditer(r'(def\s+.*)', text, re.DOTALL))
            clean_code = match[-1].group(1).strip() if match else text.strip()

        # 4. TỰ ĐỘNG SỬA LỖI REGEX (Sửa lỗi \d, \w, \s gây SyntaxWarning)
        # Tìm các pattern Regex không có tiền tố 'r' và thêm nó vào
        # Ví dụ: '\d+' -> r'\d+'
        clean_code = re.sub(r"(?<!r)(['\"])(?:\\[dwsDWS][+*?]?)+(['\"])", r"r\1\g<0>\1", clean_code)
        # Sửa lỗi trực tiếp cho trường hợp phổ biến bạn gặp:
        clean_code = clean_code.replace("'\\d'", "r'\\d'").replace('"\\d"', 'r"\\d"')

        # 5. CẮT RÁC SAU RETURN (Đảm bảo code sạch cho Sandbox)
        if "return " in clean_code:
            lines = clean_code.split('\n')
            final_lines = []
            found_final_return = False
            for line in lines:
                final_lines.append(line)
                # Nếu gặp dòng return không có thụt lề (thường là kết thúc hàm chính)
                if line.strip().startswith('return ') and not line.startswith((' ', '\t')):
                    # Kiểm tra xem đây có phải là return của hàm chính không
                    found_final_return = True
                # Nếu đã qua dòng return chính và gặp 1 dòng text không thụt lề -> Cắt bỏ từ đây
                if found_final_return and line == lines[-1] == line.strip() and not line.startswith(('#', 'def', 'import')):
                    final_lines.pop()
                    break
            clean_code = '\n'.join(final_lines).strip()

        # 6. TỰ ĐỘNG ĐÓNG NGOẶC
        for open_b, close_b in [('(', ')'), ('[', ']'), ('{', '}')]:
            diff = clean_code.count(open_b) - clean_code.count(close_b)
            if diff > 0:
                clean_code += close_b * diff
            
        return clean_code