from .base_agent import BaseAgent
import re
import textwrap

class Tester(BaseAgent):
    def __init__(self, model_name="gemma3:4b", num_predict=4096, num_ctx=16384, base_url=None):
        super().__init__(model_name=model_name, num_predict=num_predict, num_ctx=num_ctx, base_url=base_url, temperature=0.3)

    def create_tests(self, task_goal, code_to_attack=None, mode="balanced", history_feedback=""):
        fn_name = "solution_func"
        if code_to_attack:
            actual_fn = re.search(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)", code_to_attack)
            if actual_fn:
                fn_name = actual_fn.group(1)
        else:
            target_fn = re.search(r"(?:hàm|function|def)\s+([a-zA-Z_][a-zA-Z0-9_]*)", task_goal)
            if target_fn:
                fn_name = target_fn.group(1)
        
        history_section = f"\n2. AUDITOR FEEDBACK:\n{history_feedback}\n" if history_feedback else ""
        code_section = f"\n3. CURRENT SOURCE CODE TO TEST:\n```python\n{code_to_attack}\n```\n" if code_to_attack else ""

        prompt = f"""ROLE: Senior QA Engineer (Metamorphic Testing Expert).
TASK: Test the function `{fn_name}` based on PROBLEM SPECIFICATION.

### [INPUT DATA]:
1. PROBLEM SPECIFICATION & REQUIREMENTS (TASK GOAL):
{task_goal}
{history_section}{code_section}
### [RULE: NO-ORACLE & 3-TIER ASSERTION PRINCIPLE]
- Inputs (I1, I2) MUST strictly satisfy domain preconditions and docstring examples in PROBLEM SPECIFICATION.
- NEVER write tautological assertions or weak comparisons without element-level checking.
- EVERY MR test block MUST contain 3 Tiers of assertions:
  1. Tier 1 (Non-Triviality): Check that output is not None/empty.
  2. Tier 2 (Domain Structural Invariant Check - CRITICAL): Validate structural rules on elements of the output as specified in PROBLEM SPECIFICATION.
  3. Tier 3 (Metamorphic Property Assertion): Compare exact property relations appropriate for the input transformation.

### [STRATEGY: CHOOSE 2 DIFFERENT MR TYPES]
1. Symmetry/Permutation (Preserving multiset/set structure)
2. Value Transformation (Scale, offset, or space/format invariance)
3. Inclusion/Exclusion (Subsets or element filtering invariants)
4. Recursive/Inductive (Inductive relations)
5. Domain-Specific (Docstring rule invariants)

### [OUTPUT FORMAT]
<thought>
- Analyze problem requirements from PROBLEM SPECIFICATION.
- Identify domain structural invariants for Tier 2.
- Select 2 MRs.
</thought>
<test>
# MR 1: <MR_1_NAME>
# Write valid executable Python assertions using 3 Tiers.

# MR 2: <MR_2_NAME>
# Write valid executable Python assertions using 3 Tiers.
</test>

### MODE: {mode} - {self.get_distribution(mode)}
"""

        raw_output = ""
        for chunk in self.llm.stream(prompt):
            raw_output += chunk
            if "</test>" in raw_output:
                break
                
        return self.post_process_tests(raw_output, fn_name)

    def get_distribution(self, mode):
        return {
            "valid": "Focus on stability (Permutation).",
            "noise": "Focus on edge cases (0, None, Empty).",
            "balanced": "Mix Permutation with Value Transformation."
        }.get(mode, "balanced")

    def post_process_tests(self, text, target_fn): 
        def clean_code(block):
            clean = re.sub(r'```(?:python|py)?\s*(.*?)\s*```', r'\1', block, flags=re.DOTALL | re.IGNORECASE).strip()
            return textwrap.dedent(clean)

        test_blocks = re.findall(r'<test>(.*?)</test>', text, re.DOTALL)
        
        final_tests = []
        for block in test_blocks:
            clean_block = clean_code(block)
            if f'{target_fn}' in clean_block and 'assert' in clean_block:
                final_tests.append(clean_block)

        if not final_tests:
            code_blocks = re.findall(r'```(?:python|py)?\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
            for block in code_blocks:
                clean_block = clean_code(block)
                if 'assert' in clean_block:
                    final_tests.append(clean_block)

        return '\n\n'.join(final_tests[:1])