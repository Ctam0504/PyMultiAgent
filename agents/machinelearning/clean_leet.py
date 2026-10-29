import json
import re
import ollama
from tqdm import tqdm

# Cấu hình
INPUT_FILE = "leetcode.jsonl"
OUTPUT_FILE = "leetcode_rephrased.jsonl"
MODEL_NAME = "gemma3:4b"

# System Prompt nghiêm ngặt để ép LLM không thay đổi cấu trúc/signature
SYSTEM_PROMPT = """You are an expert computer science teacher and technical writer.
Your task is to rephrase the problem description inside the given docstring of a Python function.

CRITICAL RULES:
1. Rephrase the story/description, wording, and explanation in a fresh, new creative style or different technical perspective.
2. DO NOT change the function signature, argument names, argument types, or return type.
3. DO NOT change the constraints, example values, or input/output formats.
4. Keep the original docstring format (triplicate double quotes \"\"\").
5. Output ONLY the rephrased code snippet/docstring. Do not include markdown code block formatting like ```python, and do not add any intro/outro explanations."""


def extract_docstring_and_code(prompt_text: str):
    """Tách phần docstring (mô tả đề bài) và phần signature của hàm."""
    # Tìm docstring nằm trong triple quotes """..."""
    match = re.search(r'("""[\s\S]*?""")', prompt_text)
    if not match:
        return None, prompt_text

    docstring = match.group(1)
    prefix = prompt_text[: match.start()]
    suffix = prompt_text[match.end() :]
    return docstring, prefix, suffix


def rephrase_docstring(docstring: str) -> str:
    """Gọi Ollama gemma3:4b để rephrase đoạn docstring."""
    user_prompt = f"""Rephrase the explanation and wording in the following docstring while keeping all constraints, examples, and formatting completely intact:

{docstring}"""

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            options={
                "temperature": 0.4,  # Giữ nhiệt độ thấp để tránh sáng tạo quá đà làm hỏng format
            },
        )
        rephrased = response["message"]["content"].strip()

        # Làm sạch nếu model lỡ bọc trong ```python ... ```
        if rephrased.startswith("```"):
            lines = rephrased.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            rephrased = "\n".join(lines).strip()

        # Đảm bảo vẫn được bao bọc bởi triple quotes
        if not rephrased.startswith('"""'):
            rephrased = '"""\n' + rephrased
        if not rephrased.endswith('"""'):
            rephrased = rephrased + '\n"""'

        return rephrased
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return docstring  # Nếu lỗi thì trả lại docstring gốc


def process_jsonl():
    print(f"Processing {INPUT_FILE} using {MODEL_NAME}...")

    # Đếm tổng số dòng để hiện progress bar
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    with open(INPUT_FILE, "r", encoding="utf-8") as infile, open(
        OUTPUT_FILE, "w", encoding="utf-8"
    ) as outfile:

        for line in tqdm(infile, total=total_lines, desc="Rephrasing LeetCode Prompts"):
            if not line.strip():
                continue

            data = json.loads(line)
            original_prompt = data.get("prompt", "")

            # Tách docstring ra khỏi phần import và signature
            docstring, prefix, suffix = extract_docstring_and_code(original_prompt)

            if docstring:
                # Rephrase phần docstring
                new_docstring = rephrase_docstring(docstring)
                # Ghép lại prompt mới
                data["prompt"] = f"{prefix}{new_docstring}{suffix}"

            # Ghi lại vào file jsonl mới (giữ nguyên task_id, entry_point, canonical_solution, test)
            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"\nDone! Output saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    process_jsonl()