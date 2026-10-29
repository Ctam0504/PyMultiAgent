import os
import re
import time
import hashlib
import streamlit as st

st.set_page_config(
    page_title="Metamorphic Multi-Agent System - IT Office 2D",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

os.environ["OLLAMA_BASE_URL"] = os.environ.get(
    "OLLAMA_BASE_URL", "http://localhost:11434"
)

try:
    from main import ResearchCoordinator
except ImportError:
    st.error("❌ Không tìm thấy file `main.py`! Hãy đảm bảo file `app.py` nằm cùng thư mục với `main.py`.")
    st.stop()

# ================= CSS MÔ PHỎNG PHÒNG BAN IT 2D & WORKSTATION CARDS =================
STAGE_CSS = """
<style>
.office-room {
    position: relative;
    background: linear-gradient(180deg, #1b2434 0%, #161e2e 55%, #101722 100%);
    border: 2px solid #2d3d57;
    border-radius: 16px;
    height: 250px;
    margin-bottom: 12px;
    overflow: hidden;
    box-shadow: inset 0 0 40px rgba(0,0,0,0.8), 0 8px 24px rgba(0,0,0,0.5);
}
div.stButton > button[kind="primary"] {
    background-color: #1e88e5 !important;
    border-color: #1e88e5 !important;
    color: #ffffff !important;
    font-weight: 600;
}
div.stButton > button[kind="primary"]:hover {
    background-color: #1565c0 !important;
    border-color: #1565c0 !important;
}
.office-wall {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 140px;
    background: linear-gradient(180deg, #182232 0%, #1f2b3e 100%);
    border-bottom: 3px solid #2e3e57;
}
.office-window {
    position: absolute;
    top: 15px;
    right: 35px;
    width: 140px;
    height: 70px;
    background: #090e17;
    border: 3px solid #33445c;
    border-radius: 6px;
    overflow: hidden;
    box-shadow: inset 0 0 15px rgba(0, 242, 254, 0.15);
}
.window-pane {
    position: absolute;
    width: 100%;
    height: 100%;
    background: repeating-linear-gradient(90deg, transparent 0px, transparent 25px, #162438 26px, #162438 30px);
}
.window-stars {
    position: absolute;
    width: 4px;
    height: 4px;
    background: #ffffff;
    box-shadow: 20px 10px #70a1ff, 50px 25px #ffffff, 80px 8px #7bed9f, 110px 30px #ffa502;
    border-radius: 50%;
}
.kanban-board {
    position: absolute;
    top: 18px;
    left: 40px;
    width: 110px;
    height: 65px;
    background: #f1f2f6;
    border: 3px solid #57606f;
    border-radius: 4px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.4);
    padding: 4px;
    display: flex;
    gap: 4px;
}
.kanban-col {
    flex: 1;
    background: #dfe4ea;
    border-radius: 2px;
    display: flex;
    flex-direction: column;
    gap: 3px;
    padding: 2px;
}
.kanban-card { height: 9px; border-radius: 2px; }
.card-yellow { background: #eccc68; }
.card-blue { background: #70a1ff; }
.card-green { background: #2ed573; }
.card-red { background: #ff4757; }
.office-floor {
    position: absolute;
    bottom: 0;
    left: 0;
    width: 100%;
    height: 110px;
    background: repeating-linear-gradient(0deg, #1a1e27 0px, #1a1e27 18px, #151820 19px),
                repeating-linear-gradient(90deg, #222938 0px, #222938 60px, #19202d 61px);
    box-shadow: inset 0 10px 25px rgba(0,0,0,0.6);
}
.paper-plane-path {
    position: absolute;
    bottom: 95px;
    height: 2px;
    background: repeating-linear-gradient(90deg, #00f2fe 0, #00f2fe 8px, transparent 8px, transparent 16px);
    z-index: 6;
    transition: all 0.6s cubic-bezier(0.34, 1.56, 0.64, 1);
}
.paper-plane {
    position: absolute;
    top: -9px;
    right: -4px;
    font-size: 14px;
    animation: flightBob 0.6s infinite alternate;
}
.workstation-zone {
    position: absolute;
    bottom: 12px;
    width: 90px;
    display: flex;
    flex-direction: column;
    align-items: center;
    transition: all 0.4s ease;
    z-index: 10;
}
.desk {
    position: absolute;
    bottom: 0;
    width: 78px;
    height: 38px;
    background: linear-gradient(180deg, #3d4a61 0%, #263143 100%);
    border: 2px solid #546582;
    border-radius: 6px;
    box-shadow: 0 8px 12px rgba(0,0,0,0.7);
    display: flex;
    justify-content: center;
    align-items: flex-start;
    padding-top: 3px;
}
.monitor {
    width: 32px;
    height: 20px;
    background: #0d121c;
    border: 2px solid #4a5c78;
    border-radius: 3px;
    position: relative;
    box-shadow: 0 0 10px rgba(0, 242, 254, 0.3);
}
.monitor-screen {
    width: 100%;
    height: 100%;
    background: #09121f;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    font-size: 9px;
}
.monitor-stand {
    position: absolute;
    bottom: -5px;
    left: 50%;
    transform: translateX(-50%);
    width: 8px;
    height: 4px;
    background: #718096;
}
.coffee-mug {
    position: absolute;
    top: 5px;
    right: 8px;
    width: 7px;
    height: 9px;
    background: #ff6b6b;
    border-radius: 2px;
}
.coffee-steam {
    position: absolute;
    top: -6px;
    right: 10px;
    font-size: 8px;
    animation: steamFloat 1.2s infinite ease-out;
}
.human-character {
    position: absolute;
    bottom: 30px;
    display: flex;
    flex-direction: column;
    align-items: center;
    transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
    z-index: 12;
}
.human-head {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: #ffdfba;
    border: 2px solid #2d3748;
    display: flex;
    justify-content: center;
    align-items: center;
    font-size: 16px;
    position: relative;
    box-shadow: 0 3px 6px rgba(0,0,0,0.3);
    z-index: 14;
}
.human-body {
    width: 32px;
    height: 26px;
    border-radius: 8px 8px 4px 4px;
    margin-top: -3px;
    border: 2px solid #242c3d;
    display: flex;
    justify-content: center;
    align-items: center;
    position: relative;
    box-shadow: 0 4px 8px rgba(0,0,0,0.4);
    z-index: 13;
}
.outfit-coder { background: #2e86de; }
.outfit-classifier { background: #9b59b6; }
.outfit-tester { background: #e67e22; }
.outfit-sandbox { background: #16a085; }
.outfit-reviewer { background: #c0392b; }

.hands-typing {
    position: absolute;
    bottom: -2px;
    width: 22px;
    display: flex;
    justify-content: space-between;
}
.hand {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #ffdfba;
    border: 1px solid #7f8c8d;
}
.chair-shadow {
    width: 44px;
    height: 8px;
    background: rgba(0, 0, 0, 0.6);
    border-radius: 50%;
    margin-top: 1px;
    filter: blur(1.5px);
}
.role-badge {
    position: absolute;
    bottom: -16px;
    font-size: 10px;
    font-weight: 700;
    color: #a4b0be;
    background: #0f1622;
    padding: 1px 7px;
    border-radius: 8px;
    border: 1px solid #2f3542;
    white-space: nowrap;
    text-shadow: 0 1px 2px #000;
}
.active-worker .human-character {
    transform: translateY(-16px) scale(1.15);
    z-index: 25;
}
.active-worker .human-body {
    border-color: #00f2fe !important;
    box-shadow: 0 0 16px rgba(0, 242, 254, 0.8);
    animation: humanWorking 0.5s infinite alternate ease-in-out;
}
.active-worker .monitor {
    border-color: #00f2fe;
    box-shadow: 0 0 16px #00f2fe;
}
.active-worker .role-badge {
    color: #00f2fe;
    border-color: #00f2fe;
    font-weight: 800;
}
.done-worker .human-body {
    border-color: #2ed573;
    box-shadow: 0 0 8px rgba(46, 213, 115, 0.4);
}
.done-worker .role-badge { color: #2ed573; }
.idle-worker { opacity: 0.55; filter: grayscale(40%); }

.office-bubble {
    position: absolute;
    top: -55px;
    background: #ffffff;
    color: #1e272e;
    font-size: 11px;
    font-weight: 700;
    padding: 5px 11px;
    border-radius: 12px;
    white-space: nowrap;
    box-shadow: 0 6px 18px rgba(0,0,0,0.6);
    z-index: 40;
    animation: popBubble 0.25s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}
.office-bubble::after {
    content: '';
    position: absolute;
    bottom: -6px;
    left: 50%;
    transform: translateX(-50%);
    border-width: 6px 6px 0;
    border-style: solid;
    border-color: #ffffff transparent;
}
.mini-terminal {
    background-color: #080b10;
    border: 1px solid #1f2b3e;
    border-radius: 8px;
    padding: 8px 12px;
    height: 110px;
    overflow-y: auto;
    font-family: 'Consolas', monospace;
    font-size: 11px;
    line-height: 1.45;
}
.compact-test-scroll {
    max-height: 250px;
    overflow-y: auto;
    padding-right: 6px;
}
@keyframes humanWorking {
    0% { transform: translateY(0px) rotate(0deg); }
    50% { transform: translateY(-3px) rotate(-2deg); }
    100% { transform: translateY(-5px) rotate(2deg); }
}
@keyframes steamFloat {
    0% { transform: translateY(0px) opacity(0.8); }
    100% { transform: translateY(-7px) opacity(0); }
}
@keyframes flightBob {
    0% { transform: translateY(0px); }
    100% { transform: translateY(-4px); }
}
@keyframes popBubble {
    0% { transform: scale(0.6); opacity: 0; }
    100% { transform: scale(1); opacity: 1; }
}
</style>
"""
st.markdown(STAGE_CSS, unsafe_allow_html=True)

st.title("🏢 Phòng R&D Công Nghệ - Metamorphic Multi-Agent System")
st.caption("Mô phỏng 2D tương tác thời gian thực giữa các kỹ sư lập trình, kiểm thử và kiến trúc sư hệ thống")

AGENTS_DEF = [
    {"key": "CODER", "name": "Coder (Dev)", "face": "👨‍💻", "outfit": "outfit-coder", "icon_screen": "💻", "pos": 6},
    {"key": "CLASSIFIER", "name": "Classifier (Data)", "face": "👩‍🔬", "outfit": "outfit-classifier", "icon_screen": "🧠", "pos": 28},
    {"key": "TESTER", "name": "Tester (QA)", "face": "🕵️‍♂️", "outfit": "outfit-tester", "icon_screen": "🐞", "pos": 50},
    {"key": "SANDBOX", "name": "Sandbox (DevOps)", "face": "🧑‍🔧", "outfit": "outfit-sandbox", "icon_screen": "🖧", "pos": 72},
    {"key": "REVIEWER", "name": "Reviewer (Lead)", "face": "👨‍💼", "outfit": "outfit-reviewer", "icon_screen": "📋", "pos": 91},
]

# State khởi tạo
if "current_code" not in st.session_state:
    st.session_state.current_code = ""
if "new_repaired_code" not in st.session_state:
    st.session_state.new_repaired_code = ""
if "final_code" not in st.session_state:
    st.session_state.final_code = ""
if "test_cases_list" not in st.session_state:
    st.session_state.test_cases_list = []
if "active_agent" not in st.session_state:
    st.session_state.active_agent = "CODER"
if "previous_agent" not in st.session_state:
    st.session_state.previous_agent = "CODER"
if "agent_speech" not in st.session_state:
    st.session_state.agent_speech = "Chào team! Sẵn sàng nhận dự án."
if "agent_history" not in st.session_state:
    st.session_state.agent_history = []
if "log_entries" not in st.session_state:
    st.session_state.log_entries = []

if "classifier_data" not in st.session_state:
    st.session_state.classifier_data = {
        "features": {}, "prob": None, "threshold": 0.56, "decision": None, "logprob": 0.0
    }
if "reviewer_verdicts" not in st.session_state:
    st.session_state.reviewer_verdicts = []

def clean_code_snippet(raw_code: str) -> str:
    if not raw_code:
        return ""
    code = raw_code.strip()
    if code.startswith("```"):
        code = re.sub(r"^```(?:python)?\s*", "", code, flags=re.IGNORECASE)
        code = re.sub(r"\s*```$", "", code)
    return code.strip()

def get_code_core_hash(code: str) -> str:
    """Hash chuẩn hóa bỏ qua chú thích, dòng trống và khoảng trắng thừa để kiểm tra trùng lặp logic"""
    if not code:
        return ""
    cleaned = clean_code_snippet(code)
    lines = [
        re.sub(r"\s+", " ", l.strip()) 
        for l in cleaned.splitlines() 
        if l.strip() and not l.strip().startswith("#")
    ]
    normalized = "\n".join(lines)
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()

# ================= SIDEBAR =================
with st.sidebar:
    st.header("⚙️ Cấu hình phòng ban")
    mode = st.selectbox(
        "Quy trình phối hợp (Mode):",
        options=["full", "coder_tester_reviewer", "coder_tester", "coder"],
        index=0,
        format_func=lambda x: {
            "full": "Full Multi-Agent (+ ML Classifier)",
            "coder_tester_reviewer": "Coder + Tester + Reviewer",
            "coder_tester": "Coder + Tester",
            "coder": "Chỉ Coder (Single Dev)",
        }[x],
    )

    attempts = st.number_input(
        "Số sprint tối đa (Max Attempts):",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
        disabled=(mode == "coder"),
    )

    stability = st.number_input(
        "Số vòng nghiệm thu MR (Stability Threshold):",
        min_value=1,
        max_value=5,
        value=3,
        step=1,
        disabled=(mode == "coder"),
    )

    if st.button("🧹 Dọn dẹp phòng làm việc", use_container_width=True):
        st.session_state.current_code = ""
        st.session_state.new_repaired_code = ""
        st.session_state.final_code = ""
        st.session_state.test_cases_list = []
        st.session_state.active_agent = "CODER"
        st.session_state.previous_agent = "CODER"
        st.session_state.agent_speech = "Phòng ban đã được dọn sạch."
        st.session_state.agent_history = []
        st.session_state.log_entries = []
        st.session_state.classifier_data = {
            "features": {}, "prob": None, "threshold": 0.56, "decision": None, "logprob": 0.0
        }
        st.session_state.reviewer_verdicts = []
        st.rerun()

# ================= NHẬP YÊU CẦU =================
task_goal = st.text_area(
    "👉 Yêu cầu kỹ thuật bài toán (Task Goal):",
    height=80,
    placeholder="Ví dụ: Viết hàm tim_so_nguyen_to(n) trả về danh sách các số nguyên tố nhỏ hơn hoặc bằng n...",
)

start_button = st.button("🚀 Triển khai dự án", type="primary", use_container_width=True)

# ================= BỐ CỤC 2 CỘT =================
col_left, col_right = st.columns([1, 1], gap="medium")

with col_left:
    st.subheader("💻 Mã nguồn cũ (Cần sửa / Đối chiếu)")
    current_code_placeholder = st.empty()
    if st.session_state.current_code:
        current_code_placeholder.code(st.session_state.current_code, language="python")
    else:
        current_code_placeholder.info("Mã nguồn cũ trước khi bị Reviewer bắt lỗi sẽ hiển thị tại đây để đối chiếu...")

    st.markdown("---")
    st.subheader("🏢 Không gian làm việc phòng ban IT")
    stage_placeholder = st.empty()
    terminal_placeholder = st.empty()

    st.markdown("#### 🔍 Chi tiết bàn làm việc (Agent Workstation Insights)")
    agent_tabs_placeholder = st.empty()

with col_right:
    st.subheader("🛠️ Mã mới sửa đổi & Sản phẩm nghiệm thu")
    result_code_placeholder = st.empty()
    if st.session_state.final_code:
        result_code_placeholder.code(st.session_state.final_code, language="python")
    elif st.session_state.new_repaired_code:
        result_code_placeholder.code(st.session_state.new_repaired_code, language="python")
    else:
        result_code_placeholder.info("Mã nguồn mới vừa chỉnh sửa hoặc bản nghiệm thu cuối cùng sẽ xuất hiện tại đây.")

    st.markdown("---")
    st.subheader("🧪 Bộ kiểm thử biến hình (Metamorphic Relations)")
    test_placeholder = st.empty()
    if st.session_state.test_cases_list:
        with test_placeholder.container():
            st.markdown('<div class="compact-test-scroll">', unsafe_allow_html=True)
            for idx, tc in enumerate(st.session_state.test_cases_list, 1):
                with st.expander(f"📌 Test Case Đối Kháng MR - Vòng {idx} ({len(tc.splitlines())} dòng)", expanded=False):
                    st.code(tc, language="python")
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        test_placeholder.info("Các bẫy kiểm thử do QA Tester tạo ra sẽ liệt kê gọn gàng tại đây.")

# ================= HÀM VẼ GIAO DIỆN PHÒNG BAN =================
def render_stage(active_key, prev_key, speech):
    agent_dict = {a["key"]: a for a in AGENTS_DEF}
    cur_pos = agent_dict.get(active_key, {}).get("pos", 6)
    prev_pos = agent_dict.get(prev_key, {}).get("pos", cur_pos)

    left_x = min(prev_pos, cur_pos) + 3
    beam_w = abs(cur_pos - prev_pos)
    if beam_w == 0:
        beam_w = 4

    flight_html = f'<div class="paper-plane-path" style="left: {left_x}%; width: {beam_w}%;"><div class="paper-plane">📄</div></div>'

    workstations_html = []
    for agent in AGENTS_DEF:
        k = agent["key"]
        pos_x = agent["pos"]
        is_active = (k == active_key)
        is_done = (k in st.session_state.agent_history and not is_active)

        if is_active:
            status_cls = "active-worker"
            bubble_html = f'<div class="office-bubble">{speech}</div>'
        elif is_done:
            status_cls = "done-worker"
            bubble_html = ''
        else:
            status_cls = "idle-worker"
            bubble_html = ''

        workstations_html.append(
            f'<div class="workstation-zone {status_cls}" style="left: {pos_x}%;">'
            f'{bubble_html}'
            f'<div class="human-character">'
            f'  <div class="human-head">{agent["face"]}</div>'
            f'  <div class="human-body {agent["outfit"]}">'
            f'    <div class="hands-typing"><div class="hand"></div><div class="hand"></div></div>'
            f'  </div>'
            f'  <div class="chair-shadow"></div>'
            f'</div>'
            f'<div class="desk">'
            f'  <div class="coffee-mug"><div class="coffee-steam">~</div></div>'
            f'  <div class="monitor">'
            f'    <div class="monitor-screen">{agent["icon_screen"]}</div>'
            f'    <div class="monitor-stand"></div>'
            f'  </div>'
            f'</div>'
            f'<div class="role-badge">{agent["name"]}</div>'
            f'</div>'
        )

    full_room_html = (
        '<div class="office-room">'
        '<div class="office-wall">'
        '  <div class="kanban-board">'
        '    <div class="kanban-col"><div class="kanban-card card-yellow"></div><div class="kanban-card card-blue"></div></div>'
        '    <div class="kanban-col"><div class="kanban-card card-red"></div></div>'
        '    <div class="kanban-col"><div class="kanban-card card-green"></div><div class="kanban-card card-green"></div></div>'
        '  </div>'
        '  <div class="office-window"><div class="window-stars"></div><div class="window-pane"></div></div>'
        '</div>'
        '<div class="office-floor"></div>'
        + flight_html
        + ''.join(workstations_html) +
        '</div>'
    )
    stage_placeholder.markdown(full_room_html, unsafe_allow_html=True)

def render_terminal():
    lines = []
    for item in st.session_state.log_entries[-10:]:
        lvl = item.get("level", "info")
        color = "#58a6ff"
        if lvl == "success": color = "#3fb950"
        elif lvl == "error": color = "#f85149"
        elif lvl == "warning": color = "#d29922"

        lines.append(
            f"<div style='margin-bottom: 2px;'>"
            f"<span style='color:{color};'>▶ [{item['agent']}] {item['status']}</span>: {item['note']}"
            f"</div>"
        )
    terminal_html = f"<div class='mini-terminal'>{''.join(lines)}</div>"
    terminal_placeholder.markdown(terminal_html, unsafe_allow_html=True)

def render_workstation_details():
    with agent_tabs_placeholder.container():
        tab_classifier, tab_reviewer, tab_tester, tab_sandbox = st.tabs([
            "👩‍🔬 Classifier (Data)",
            "👨‍💼 Reviewer (Lead)",
            "🕵️‍♂️ Tester (QA)",
            "🧑‍🔧 Sandbox (DevOps)"
        ])

        with tab_classifier:
            c_data = st.session_state.classifier_data
            feats = c_data.get("features", {})
            prob = c_data.get("prob")
            th = c_data.get("threshold", 0.56)
            decision = c_data.get("decision")

            if prob is not None:
                c1, c2, c3 = st.columns(3)
                c1.metric("Xác suất Tin cậy (Prob)", f"{prob:.4f}")
                c2.metric("Ngưỡng tối ưu (Threshold)", f"{th:.4f}")
                dec_color = "🟢 Single-Agent" if "single" in str(decision).lower() else "🟠 Multi-Agent"
                c3.metric("Phán định mô hình", dec_color)

                st.caption("📊 **Các chỉ số đặc trưng mã nguồn trích xuất (AST Features):**")
                cols_feat = st.columns(4)
                cols_feat[0].markdown(f"**Lines:** `{feats.get('code_lines', 0)}`")
                cols_feat[0].markdown(f"**Chars:** `{feats.get('code_len', 0)}`")
                cols_feat[1].markdown(f"**Cyclomatic:** `{feats.get('cyclomatic', 1)}`")
                cols_feat[1].markdown(f"**Max Depth:** `{feats.get('max_depth', 0)}`")
                cols_feat[2].markdown(f"**Branch / If:** `{feats.get('Branch', 0)}`")
                cols_feat[2].markdown(f"**Loops:** `{feats.get('Loops', 0)}`")
                cols_feat[3].markdown(f"**Operations:** `{feats.get('Ops', 0)}`")
                cols_feat[3].markdown(f"**Logprob:** `{c_data.get('logprob', 0.0):.4f}`")
            else:
                st.info("Chưa có phiên phân loại ML nào được kích hoạt. Hãy khởi chạy dự án.")

        with tab_reviewer:
            if st.session_state.reviewer_verdicts:
                for idx, rev in enumerate(st.session_state.reviewer_verdicts, 1):
                    target = rev.get("target", "CODER")
                    badge_style = "color:#ff4757; font-weight:bold;" if target == "CODER" else "color:#ffa502; font-weight:bold;"
                    st.markdown(f"**Lần thẩm định #{idx}** — Phán quyết lỗi: <span style='{badge_style}'>[TARGET: {target}]</span>", unsafe_allow_html=True)
                    st.code(rev.get("feedback", "").strip(), language="markdown")
            else:
                st.info("Chưa có tranh chấp lỗi nào cần Reviewer phân xử.")

        with tab_tester:
            if st.session_state.test_cases_list:
                st.caption(f"QA Tester đã thiết kế **{len(st.session_state.test_cases_list)}** kịch bản đối kháng:")
                for i, tc in enumerate(st.session_state.test_cases_list, 1):
                    st.markdown(f"• **MR Vòng {i}**: `{len(tc.splitlines())}` dòng assertions kiểm thử.")
            else:
                st.info("Tester đang ở trạng thái chờ nhiệm vụ.")

        with tab_sandbox:
            sandbox_logs = [e for e in st.session_state.log_entries if e["agent"] == "SANDBOX"]
            if sandbox_logs:
                for log in sandbox_logs[-4:]:
                    st.markdown(f"**{log['status']}**: {log['note']}")
            else:
                st.info("Môi trường Docker Sandbox sẵn sàng.")

def interact_agent(new_agent_key, speech, log_status="", log_note="", level="info"):
    st.session_state.previous_agent = st.session_state.active_agent
    st.session_state.active_agent = new_agent_key
    st.session_state.agent_speech = speech

    if new_agent_key not in st.session_state.agent_history:
        st.session_state.agent_history.append(new_agent_key)

    if log_status:
        st.session_state.log_entries.append({
            "agent": new_agent_key, "status": log_status, "note": log_note, "level": level
        })
    render_stage(new_agent_key, st.session_state.previous_agent, speech)
    render_terminal()
    render_workstation_details()

render_stage(st.session_state.active_agent, st.session_state.previous_agent, st.session_state.agent_speech)
render_terminal()
render_workstation_details()

# ================= XỬ LÝ CHU TRÌNH KHI BẤM NÚT =================
if start_button:
    if not task_goal.strip():
        st.warning("⚠️ Vui lòng nhập yêu cầu bài toán trước!")
    else:
        st.session_state.current_code = ""
        st.session_state.new_repaired_code = ""
        st.session_state.final_code = ""
        st.session_state.test_cases_list = []
        st.session_state.agent_history = []
        st.session_state.log_entries = []
        st.session_state.reviewer_verdicts = []
        st.session_state.classifier_data = {
            "features": {}, "prob": None, "threshold": 0.56, "decision": None, "logprob": 0.0
        }
        test_placeholder.empty()
        result_code_placeholder.empty()
        current_code_placeholder.empty()

        coordinator = ResearchCoordinator(mode=mode)
        interact_agent("CODER", "Tôi đang phân tích yêu cầu để code!", "INIT", f"Chế độ: {mode.upper()}", "info")

        # Lưu trữ các hash mã nguồn đã từng xuất hiện để đối chiếu hiển thị
        seen_hashes = set()

        def sync_ui_code(code_str: str):
            """
            Chỉ cập nhật UI khi mã nguồn mới THỰC SỰ KHÁC BIỆT so với mã đang hiển thị.
            Nếu mã trùng lặp (ví dụ lần 3 giống lần 2), giữ nguyên hoàn toàn Lần 1 và Lần 2 trên UI.
            """
            cleaned = clean_code_snippet(code_str)
            if not cleaned:
                return

            h_new = get_code_core_hash(cleaned)
            right_code = st.session_state.new_repaired_code
            h_right = get_code_core_hash(right_code) if right_code else ""

            # NẾU MÃ BỊ TRÙNG VỚI MÃ Ở CỘT PHẢI HOẶC ĐÃ XUẤT HIỆN TRƯỚC ĐÓ:
            if right_code and (h_new == h_right or h_new in seen_hashes):
                interact_agent(
                    "CODER",
                    "⚠️ Mã sinh ra trùng lặp! Giữ nguyên hiển thị đối chiếu lần trước trên màn hình...",
                    "DUPLICATE",
                    "Mã nguồn trùng lặp - Giữ nguyên 2 cột đối chiếu trên giao diện",
                    "warning"
                )
                return

            # NẾU LÀ MÃ MỚI THỰC SỰ KHÁC BIỆT:
            if right_code:
                # Đưa mã của lần trước sang Cột Trái làm mã cũ đối chiếu
                st.session_state.current_code = right_code
                current_code_placeholder.code(right_code, language="python")

            seen_hashes.add(h_new)
            st.session_state.new_repaired_code = cleaned
            result_code_placeholder.code(cleaned, language="python")

        # ================= 1. HOOK CODER ĐỒNG BỘ NGUYÊN BẢN VỚI TERMINAL =================
        if hasattr(coordinator, "coder") and coordinator.coder:
            orig_gen_with_logprob = getattr(coordinator.coder, "generate_with_logprob", None)
            orig_gen = getattr(coordinator.coder, "generate", None)

            # A. Hook generate_with_logprob ở đầu mỗi attempt
            if orig_gen_with_logprob:
                def patched_gen_with_logprob(*args, **kwargs):
                    if st.session_state.new_repaired_code:
                        interact_agent("CODER", "Đang viết bản vá theo feedback...", "REPAIR", "Coder đang viết mã mới...", "warning")
                    else:
                        interact_agent("CODER", "Đang gõ mã nguồn giải thuật...", "CODING", "Coder đang viết code...", "info")

                    # Chạy nguyên bản logic của main.py
                    code, lp = orig_gen_with_logprob(*args, **kwargs)
                    
                    # Cập nhật UI (nếu trùng thì tự động giữ nguyên hiển thị 2 cột)
                    sync_ui_code(code)
                    return code, lp

                coordinator.coder.generate_with_logprob = patched_gen_with_logprob

            # B. Hook generate khi main.py phát hiện trùng và yêu cầu thử hướng khác
            if orig_gen:
                def patched_gen(*args, **kwargs):
                    interact_agent(
                        "CODER",
                        "Phát hiện trùng lặp từ Terminal! Đang thử hướng tiếp cận khác...",
                        "RETRY",
                        "Đổi hướng tiếp cận theo điều phối của main.py...",
                        "warning"
                    )

                    # Chạy đúng 1 lần theo đúng tham số và prompt mà main.py truyền vào
                    code = orig_gen(*args, **kwargs)

                    # Cập nhật UI (nếu vẫn trùng thì tiếp tục giữ nguyên 2 cột)
                    sync_ui_code(code)
                    return code

                coordinator.coder.generate = patched_gen

        # ================= 2. CLASSIFIER HOOK =================
        orig_predict_single = getattr(coordinator, "predict_single_agent", None)
        if orig_predict_single:
            def patched_predict_single(prompt, code, logprob=0.0):
                interact_agent("CLASSIFIER", "Nhận mã! Đang tính toán độ phức tạp...", "SCAN", "Classifier trích xuất đặc trưng AST...", "info")
                features = coordinator._extract_all_code_features(code)
                is_single, prob = orig_predict_single(prompt, code, logprob)
                
                p_val = f"{prob:.4f}" if prob is not None else "N/A"
                th = getattr(coordinator, "optimal_threshold", 0.56)
                decision_str = "Single Agent" if is_single else "Multi-Agent"

                st.session_state.classifier_data = {
                    "features": features,
                    "prob": prob,
                    "threshold": th,
                    "decision": decision_str,
                    "logprob": logprob if logprob is not None else 0.0,
                }

                if is_single:
                    interact_agent("CLASSIFIER", f"Độ tin cậy cao (P={p_val} >= {th:.2f})! Đẩy thẳng Sandbox.", "DECISION", f"P={p_val} -> Single Agent", "success")
                else:
                    interact_agent("CLASSIFIER", f"Độ phức tạp cao (P={p_val} < {th:.2f})! Kích hoạt QA Tester!", "DECISION", f"P={p_val} -> Multi-Agent", "warning")
                return is_single, prob
            coordinator.predict_single_agent = patched_predict_single

        # ================= 3. TESTER HOOK =================
        if hasattr(coordinator, "adversary") and coordinator.adversary:
            orig_create_tests = getattr(coordinator.adversary, "create_tests", None)
            if orig_create_tests:
                def patched_create_tests(*args, **kwargs):
                    interact_agent("TESTER", "Đang chế tạo các ca kiểm thử biên...", "ATTACK", "QA Tester tìm kiếm lỗi tiềm ẩn...", "info")
                    mr_cases = orig_create_tests(*args, **kwargs)
                    cleaned_mr = clean_code_snippet(mr_cases)
                    if cleaned_mr and "assert" in cleaned_mr:
                        st.session_state.test_cases_list.append(cleaned_mr)
                        rnd = len(st.session_state.test_cases_list)
                        interact_agent("TESTER", f"Đã viết xong test case Vòng {rnd}! Đẩy vào Sandbox!", "MR_READY", f"Kịch bản MR Vòng {rnd} hoàn tất.", "success")
                        
                        with test_placeholder.container():
                            st.markdown('<div class="compact-test-scroll">', unsafe_allow_html=True)
                            for idx, tc in enumerate(st.session_state.test_cases_list, 1):
                                with st.expander(f"📌 Test Case Đối Kháng MR - Vòng {idx} ({len(tc.splitlines())} dòng)", expanded=False):
                                    st.code(tc, language="python")
                            st.markdown('</div>', unsafe_allow_html=True)
                    return mr_cases
                coordinator.adversary.create_tests = patched_create_tests

        # ================= 4. REVIEWER HOOK =================
        if hasattr(coordinator, "judge") and coordinator.judge:
            orig_review = getattr(coordinator.judge, "review", None)
            if orig_review:
                def patched_review(*args, **kwargs):
                    interact_agent("REVIEWER", "Có xung đột kết quả! Đang thẩm định...", "JUDGING", "Tech Lead thẩm định nguyên nhân lỗi...", "info")
                    feedback = orig_review(*args, **kwargs)
                    target = "TESTER" if "TARGET: TESTER" in str(feedback).upper() else "CODER"
                    
                    st.session_state.reviewer_verdicts.append({
                        "target": target,
                        "feedback": feedback
                    })

                    if target == "TESTER":
                        interact_agent("REVIEWER", "Phán quyết: Tester bắt test sai logic!", "VERDICT", "Tester viết sai kịch bản.", "warning")
                    else:
                        interact_agent("REVIEWER", "Phán quyết: Coder viết code sai!", "VERDICT", "Yêu cầu Coder refactor lại.", "error")
                    return feedback
                coordinator.judge.review = patched_review

        # ================= 5. SANDBOX HOOK =================
        def patched_log_event(attempt, cycle, mode_str, status, note):
            if "PASS" in status or "SUCCESS" in status:
                interact_agent("SANDBOX", f"Thực thi Vòng {cycle}: TOÀN BỘ PASS! ✅", f"[#{attempt}] {status}", note, "success")
            elif "FAIL" in status or "REJECT" in status:
                interact_agent("SANDBOX", f"Thực thi Vòng {cycle}: PHÁT HIỆN LỖI! ❌", f"[#{attempt}] {status}", note, "error")
            else:
                interact_agent("SANDBOX", f"Đang build Docker Vòng {cycle}...", f"[#{attempt}] {status}", note, "info")

        coordinator.log_event = patched_log_event

        # ================= THỰC THI QUY TRÌNH =================
        start_time = time.time()
        try:
            res = coordinator.solve_task(
                task_goal=task_goal,
                max_attempts=attempts,
                stability_threshold=stability,
            )
        except Exception as e:
            st.error(f"❌ Có lỗi trong quá trình thực thi: {str(e)}")
            interact_agent("SANDBOX", "Hệ thống gặp sự cố!", "CRASH", str(e), "error")
            res = {"success": False, "best_code": ""}

        elapsed_time = time.time() - start_time
        final_code = clean_code_snippet(res.get("best_code", ""))

        # Phân định kết quả nghiệm thu cuối cùng
        if res.get("success") and final_code:
            st.session_state.final_code = final_code
            interact_agent("CODER", f"Tuyệt vời! Mã nguồn vượt qua kiểm thử ({elapsed_time:.1f}s) 🎉", "VICTORY", "Hoàn thành và bàn giao dự án!", "success")
            result_code_placeholder.code(final_code, language="python")
            st.balloons()
        else:
            interact_agent("REVIEWER", f"Kết thúc: Chưa đạt tiêu chuẩn ({elapsed_time:.1f}s)", "FAILED", "Chưa tìm ra mã nguồn tối ưu.", "error")
            if st.session_state.new_repaired_code:
                result_code_placeholder.warning("⚠️ Bản vá mới nhất (Chưa vượt qua toàn bộ Test):")
                result_code_placeholder.code(st.session_state.new_repaired_code, language="python")
            else:
                result_code_placeholder.error("❌ Không có mã nguồn nào vượt qua nghiệm thu.")