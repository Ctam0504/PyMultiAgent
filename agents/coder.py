from .base_agent import BaseAgent

class Coder(BaseAgent):
    def __init__(self, model_name="deepseek-coder-v2", num_predict=4096, num_ctx=16384, base_url=None):
        super().__init__(model_name=model_name, num_predict=num_predict, num_ctx=num_ctx, base_url=base_url, temperature=0.1)
    
    def generate(self, prompt_context):
        full_prompt = f"""ROLE: Python Programming Expert (Senior Software Engineer).
TASK: Fix logic errors or implement algorithms based on requirements.

### [1. PROBLEM CONTEXT]:
{prompt_context}

### [2. EXECUTION PROCESS]:
1. ANALYSIS: If there is an 'Audit Table' (LOGIC AUDIT TABLE), read it carefully to understand which rule is being violated.
2. IMPLEMENTATION: Write complete Python code. Ensure the function name matches the requirement.
3. OPTIMIZATION: Ensure the algorithm is correct for all edge cases.

### [3. STRICT OUTPUT REQUIREMENTS]:
- ONLY return source code inside a ```python ... ``` block.
- CRITICAL: You MUST write the complete function body (implementation logic, loops/mappings, and return statement) inside the function! NEVER stop inside the docstring or leave the function body empty!
- ABSOLUTELY DO NOT include unrelated libraries, network code (socket), or system code unless requested.
- Do not provide lengthy explanations outside the code block.
- Follow name of the function strictly.
- Include a test example at the end of the file.
"""
        response = ""
        for chunk in self.llm.stream(full_prompt):
            response += chunk
            if len(response.split("```")) >= 3:
                break
                
        return self.extract_code(response).strip()

    def generate_with_logprob(self, prompt_context: str) -> tuple:
        full_prompt = f"""ROLE: Python Programming Expert (Senior Software Engineer).
TASK: Fix logic errors or implement algorithms based on requirements.

### [1. PROBLEM CONTEXT]:
{prompt_context}

### [2. EXECUTION PROCESS]:
1. ANALYSIS: If there is an 'Audit Table' (LOGIC AUDIT TABLE), read it carefully to understand which rule is being violated.
2. IMPLEMENTATION: Write complete Python code. Ensure the function name matches the requirement.
3. OPTIMIZATION: Ensure the algorithm is correct for all edge cases.

### [3. STRICT OUTPUT REQUIREMENTS]:
- ONLY return source code inside a ```python ... ``` block.
- CRITICAL: You MUST write the complete function body (implementation logic, loops/mappings, and return statement) inside the function! NEVER stop inside the docstring or leave the function body empty!
- ABSOLUTELY DO NOT include unrelated libraries, network code (socket), or system code unless requested.
- Do not provide lengthy explanations.
- Follow name of the function strictly.
- Include a test example at the end of the file.
"""
        raw_text, mean_logprob = super().generate_with_logprob(full_prompt)
        code = self.extract_code(raw_text).strip()
        return code, mean_logprob