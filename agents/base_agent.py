from langchain_ollama import OllamaLLM
import re
import math
import requests
import json
import os

class BaseAgent:
    def __init__(self, model_name="deepseek-coder-v2", num_predict=4096,seed = 42, num_ctx=16384, base_url=None, temperature=0.1):
        self.base_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self.llm = OllamaLLM(
            model=model_name, 
            temperature=temperature,
            num_predict=num_predict,
            seed=seed,
            num_ctx=num_ctx,
            timeout=300,
            base_url=self.base_url,
            additional_kwargs={
                "mirostat": 0,
                "top_p": 0.95,
                "top_k": 40,
                "repeat_penalty": 1.05
            },
            stop=[
                "<｜begin of sentence｜>", "<｜end of sentence｜>", 
                "User:", "Assistant:", "### [1.", "### [NHIỆM VỤ GỐC]"
            ]
        )

    def generate_with_logprob(self, prompt: str, early_stop_marker: str = "```") -> tuple:
        """
        Gọi Ollama streaming API để lấy cả response text và log-probability trung bình,
        trong khi vẫn giữ cơ chế early-stopping.
        """
        payload = {
            "model": self.llm.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": self.llm.temperature,
                "num_predict": self.llm.num_predict,
                "num_ctx": self.llm.num_ctx,
                "top_p": 0.95,
                "top_k": 40,
                "repeat_penalty": 1.05
            },
            "logprobs": True,
            "keep_alive": -1
        }
        response_text = ""
        lps = []

        try:
            with requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=300,
            ) as resp:
                resp.raise_for_status()
                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    try:
                        chunk = json.loads(raw_line)
                    except (ValueError, KeyError):
                        continue

                    token_text = chunk.get("response", "")
                    response_text += token_text

                    lp_field = chunk.get("logprobs", None)
                    if lp_field is not None:
                        if isinstance(lp_field, list):
                            for item in lp_field:
                                if isinstance(item, dict):
                                    lp_val = item.get("logprob", None)
                                    if (lp_val is not None
                                            and isinstance(lp_val, (int, float))
                                            and not math.isinf(lp_val)
                                            and not math.isnan(lp_val)):
                                        lps.append(lp_val)

                        elif isinstance(lp_field, dict):
                            lp_val = lp_field.get("logprob", None)
                            if (lp_val is not None
                                    and isinstance(lp_val, (int, float))
                                    and not math.isinf(lp_val)
                                    and not math.isnan(lp_val)):
                                lps.append(lp_val)

                        elif isinstance(lp_field, (int, float)):
                            if not math.isinf(lp_field) and not math.isnan(lp_field):
                                lps.append(float(lp_field))

                    # Early stop
                    if len(response_text.split(early_stop_marker)) >= 3:
                        break

                    if chunk.get("done", False):
                        break

        except Exception:
            return "", 0.0

        mean_logprob = float(sum(lps) / len(lps)) if lps else 0.0
        return response_text, mean_logprob

    def extract_code(self, text):
        if not text: 
            return ""
        
        # 1. Làm sạch ký tự rác MoE và các ký tự điều khiển
        bad_chars = ['｜', '▁', '◁', '▷', '█', '░', '▒', '▓']
        for char in bad_chars:
            text = text.replace(char, '')
        
        # 2. Xóa System Tokens
        text = re.sub(r'<\|.*?\|>', '', text)
        
        # 3. Trích xuất Block Code (Ưu tiên lấy block ```python ... ```)
        code_blocks = re.findall(r'```(?:python|py)?\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
        
        clean_code = ""
        if code_blocks:
            for block in reversed(code_blocks):
                if 'def ' in block or 'import ' in block:
                    clean_code = block.strip()
                    break
            if not clean_code:
                clean_code = code_blocks[-1].strip()
        else:
            # Nếu LLM không trả về ``` ```, tìm từ def tới hết
            match = list(re.finditer(r'(def\s+.*)', text, re.DOTALL))
            if match:
                clean_code = match[-1].group(1).strip()
            else:
                clean_code = text.strip()

        # 4. TỰ ĐỘNG SỬA LỖI REGEX AN TOÀN
        clean_code = re.sub(r"(?<!r)(['\"])(?:\\[dwsDWS][+*?]?)+\1", r"r\g<0>", clean_code)
        clean_code = clean_code.replace("'\\d'", "r'\\d'").replace('"\\d"', 'r"\\d"')

        # 5. TỰ ĐỘNG ĐÓNG NGOẶC (Nếu thiếu)
        for open_b, close_b in [('(', ')'), ('[', ']'), ('{', '}')]:
            diff = clean_code.count(open_b) - clean_code.count(close_b)
            if diff > 0:
                clean_code += close_b * diff
            
        return clean_code