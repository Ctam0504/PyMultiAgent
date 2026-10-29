# PyMultiAgent: Adversarial Multi-Agent Code Generation Framework

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#framework-architecture">Architecture</a> •
  <a href="#repository-structure">Structure</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a>
</p>


---

## 📖 Overview

**PyMultiAgent** is an adversarial multi-agent code generation framework designed to improve the effectiveness, correctness, and robustness of code generation. By orchestrating specialized agents (Orchestrator, Coder, Tester, and Reviewer) alongside a **Classifier-based dynamic routing mechanism**, the system intelligently balances computational overhead and execution accuracy (routing tasks between single-agent and multi-agent workflows based on code complexity).

---

## 🏗️ Framework Architecture

The framework consists of distinct specialized roles collaborating through an iterative feedback loop:

1. **Orchestrator & Classifier:** Analyzes code complexity using AST parsing and a machine learning classifier to predict task difficulty, dynamically routing execution to either a fast single-agent pipeline or a robust multi-agent workflow.
2. **Coder Agent:** Generates initial code implementations based on instructions and prompt contexts.
3. **Tester Agent:** Independently designs and generates rigorous unit test cases for code verification.
4. **Reviewer & Sandbox Executor:** Executes tests in an isolated sandbox environment, validates static analysis, and routes feedback back to the coder or tester for iterative self-optimization.

---

## 📁 Repository Structure

```text
PyMultiAgent/
├── Agents/                 # Core agent modules
│   ├── machinelearning/
|   |   └── classifier.py    # Classifier training & prediction scripts
│   ├── base_agent.py       # Base agent class for initialization and modification
│   ├── coder.py            # Code generation agent
│   ├── tester_.py          # Test generation agent
│   └── reviewer.py         # Code review and feedback agent
├── utils/
│   └── sandbox.py          # Secure code execution environment
├── app.py                  # Web application entry point (Streamlit/API)
├── main.py                 # Main CLI entry point
├── requirements.txt        # Required Python dependencies
└── README.md
```
---
## 📁 Pipeline of the system:

![Pipeline](images/pipeline.png "Pipeline of project")
 
---
### Result in 2 benchmark:
### 1. Benchmark Evaluation (MBPP)

| Metric | Coder | Coder + Tester | Coder + Tester + Reviewer | Coder + Tester + Reviewer + Classifier |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy (Pass@3)** | 98.70% ± 0.26% | 95.41% ± 0.43% | **98.84% ± 0.33%** | 98.56% ± 0.17% |
| **Time Execution (s)** | 7,292.98 | 41,464.02 | 28,429.89 | 28,571.63 |

### 2. Benchmark Evaluation (HumanEval)

| Metric | Coder | Coder + Tester | Coder + Tester + Reviewer | Coder + Tester + Reviewer + Classifier |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy (Pass@3)** | 75.41% ± 0.35% | 80.08% ± 3% | **84.55% ± 0.93%** | 80.89% ± 2% |
| **Time Execution (s)** | 2,847.37 | 17,815.46 | 5,007.51 | **2,233.21** |
---
⚙️ Installation
Clone the repository:

Bash
git clone [https://github.com/Ctam0504/PyMultiAgent.git](https://github.com/Ctam0504/PyMultiAgent.git)
cd PyMultiAgent

Install dependencies:

Bash
pip install -r requirements.txt

🚀 Usage
To execute code generation with dynamic classifier routing:

Bash
python main.py

To launch the interactive application interface:
streamlit run app.py