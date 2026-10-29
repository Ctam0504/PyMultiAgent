import argparse
import json
import os
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

# Đảm bảo in tiếng Việt / Unicode không bị lỗi trên Windows cmd/powershell
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def load_dataset(file_path: str):
    """Đọc dữ liệu từ file JSON."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path.resolve()}")
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError(f"Dữ liệu mong đợi là list, nhưng nhận được {type(data)}")
    
    return data


def extract_error_type(err_str: str) -> str:
    """Trích xuất tên loại ngoại lệ (Exception/Error) từ traceback hoặc thông báo lỗi."""
    if not err_str:
        return "None"
    
    lines = [line.strip() for line in str(err_str).strip().splitlines() if line.strip()]
    if not lines:
        return "Unknown"
    
    last_line = lines[-1]
    
    # Kiểm tra TIMEOUT trước
    if "TIMEOUT" in last_line or "timed out" in last_line.lower():
        return "TimeoutError"
        
    match = re.match(r"^([A-Za-z0-9_]+Error|[A-Za-z0-9_]+Exception|Exception)", last_line)
    if match:
        return match.group(1)
    
    # Nếu không khớp regex chuẩn, lấy token đầu tiên trước dấu hai chấm
    token = last_line.split(":")[0].strip()
    return token[:40] if token else "Unknown"


def compute_stats(values):
    """Tính các giá trị thống kê cơ bản: count, mean, std, min, median, max."""
    if not values:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "median": 0.0, "max": 0.0}
    
    n = len(values)
    mean_val = statistics.mean(values)
    std_val = statistics.stdev(values) if n > 1 else 0.0
    median_val = statistics.median(values)
    min_val = min(values)
    max_val = max(values)
    
    return {
        "count": n,
        "mean": mean_val,
        "std": std_val,
        "min": min_val,
        "median": median_val,
        "max": max_val,
    }


def analyze_data(data: list) -> dict:
    """Thực hiện toàn bộ phân tích dữ liệu."""
    total_records = len(data)
    
    # 1. Phân bố nhãn result (True / False / khác)
    results = [item.get("result") for item in data]
    result_counter = Counter(results)
    
    # 2. Phân tích lỗi (Errors)
    error_counter = Counter()
    has_error_count = 0
    for item in data:
        err = item.get("error")
        if err:
            has_error_count += 1
            err_type = extract_error_type(err)
            error_counter[err_type] += 1
        else:
            error_counter["No Error"] += 1
            
    # 3. Phân tích Logprob theo từng nhóm
    all_logprobs = [item["logprob"] for item in data if item.get("logprob") is not None]
    true_logprobs = [item["logprob"] for item in data if item.get("result") is True and item.get("logprob") is not None]
    false_logprobs = [item["logprob"] for item in data if item.get("result") is False and item.get("logprob") is not None]
    
    # 4. Phân tích độ dài code và prompt
    code_lengths_true = [len(item.get("generated_code", "")) for item in data if item.get("result") is True]
    code_lengths_false = [len(item.get("generated_code", "")) for item in data if item.get("result") is False]
    code_lines_true = [len(item.get("generated_code", "").splitlines()) for item in data if item.get("result") is True]
    code_lines_false = [len(item.get("generated_code", "").splitlines()) for item in data if item.get("result") is False]
    
    # 5. Phân tích theo bài toán (task)
    tasks = [item.get("task") for item in data if item.get("task")]
    unique_tasks = len(set(tasks))
    
    task_stats = {}
    for item in data:
        t = item.get("task")
        if not t:
            continue
        if t not in task_stats:
            task_stats[t] = {"total": 0, "passed": 0}
        task_stats[t]["total"] += 1
        if item.get("result") is True:
            task_stats[t]["passed"] += 1
            
    return {
        "total_records": total_records,
        "result_distribution": {
            k: {"count": v, "percentage": (v / total_records) * 100}
            for k, v in result_counter.items()
        },
        "has_error_count": has_error_count,
        "error_distribution": dict(error_counter.most_common(15)),
        "logprob_stats": {
            "all": compute_stats(all_logprobs),
            "true": compute_stats(true_logprobs),
            "false": compute_stats(false_logprobs),
        },
        "code_length_stats": {
            "char_len_true": compute_stats(code_lengths_true),
            "char_len_false": compute_stats(code_lengths_false),
            "lines_true": compute_stats(code_lines_true),
            "lines_false": compute_stats(code_lines_false),
        },
        "unique_tasks": unique_tasks,
        "task_stats": task_stats,
    }


def print_report(analysis: dict, file_path: str):
    """In bảng báo cáo trực quan ra màn hình."""
    print("=" * 70)
    print(f"📊 BÁO CÁO PHÂN TÍCH DỮ LIỆU: {os.path.basename(file_path)}")
    print(f"   Đường dẫn: {os.path.abspath(file_path)}")
    print("=" * 70)
    
    total = analysis["total_records"]
    print(f"\n[1] TỔNG QUAN")
    print(f"  • Tổng số bản ghi (records): {total:,}")
    print(f"  • Số lượng bài toán (unique tasks): {analysis['unique_tasks']:,}")
    
    print(f"\n[2] PHÂN BỐ CÁC LỚP KẾT QUẢ (RESULT CLASSES)")
    print(f"  {'-'*55}")
    print(f"  | {'Lớp (Class)':<15} | {'Số lượng':<12} | {'Tỷ lệ (%)':<15} |")
    print(f"  {'-'*55}")
    for res_class, info in sorted(analysis["result_distribution"].items(), key=lambda x: str(x[0])):
        name = "True (Passed)" if res_class is True else ("False (Failed)" if res_class is False else str(res_class))
        bar = "█" * int(info["percentage"] / 4)
        print(f"  | {name:<15} | {info['count']:<12,} | {info['percentage']:>6.2f}% {bar:<10} |")
    print(f"  {'-'*55}")
    
    print(f"\n[3] PHÂN BỐ CÁC LOẠI LỖI (TOP ERRORS TRONG TẬP FALSE)")
    print(f"  • Số bản ghi phát sinh lỗi: {analysis['has_error_count']:,} / {total:,} ({analysis['has_error_count']/total*100:.2f}%)")
    print(f"  {'-'*55}")
    print(f"  | {'Loại lỗi (Error Type)':<25} | {'Số lượng':<10} | {'Tỷ lệ':<10} |")
    print(f"  {'-'*55}")
    for err_type, count in analysis["error_distribution"].items():
        pct = (count / total) * 100
        print(f"  | {err_type:<25} | {count:<10,} | {pct:>6.2f}%   |")
    print(f"  {'-'*55}")
    
    print(f"\n[4] SO SÁNH LOGPROB (ĐỘ TỰ TIN CỦA MODEL)")
    print(f"  {'-'*65}")
    print(f"  | {'Nhóm':<15} | {'Mean':<10} | {'Std':<10} | {'Median':<10} | {'[Min, Max]':<12} |")
    print(f"  {'-'*65}")
    for group, label in [("all", "Tất cả"), ("true", "True (Passed)"), ("false", "False (Failed)")]:
        st = analysis["logprob_stats"][group]
        min_max_str = f"[{st['min']:.3f}, {st['max']:.3f}]"
        print(f"  | {label:<15} | {st['mean']:<10.4f} | {st['std']:<10.4f} | {st['median']:<10.4f} | {min_max_str:<12} |")
    print(f"  {'-'*65}")
    
    print(f"\n[5] SO SÁNH ĐỘ DÀI MÃ NGUỒN SINH RA (GENERATED CODE)")
    cl = analysis["code_length_stats"]
    print(f"  • Ký tự TB: Lớp True = {cl['char_len_true']['mean']:.1f} ký tự | Lớp False = {cl['char_len_false']['mean']:.1f} ký tự")
    print(f"  • Số dòng TB: Lớp True = {cl['lines_true']['mean']:.1f} dòng   | Lớp False = {cl['lines_false']['mean']:.1f} dòng")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Đọc và phân tích tập dữ liệu leetcode2.json")
    default_path = Path(__file__).parent / "leetcode2.json"
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        default=str(default_path),
        help=f"Đường dẫn tới file leetcode2.json (mặc định: {default_path})",
    )
    parser.add_argument(
        "--save-json",
        "-s",
        type=str,
        default=None,
        help="Đường dẫn file để xuất kết quả phân tích dưới dạng JSON (nếu muốn)",
    )

    args = parser.parse_args()
    
    try:
        data = load_dataset(args.file)
        analysis = analyze_data(data)
        print_report(analysis, args.file)
        
        if args.save_json:
            out_path = Path(args.save_json)
            # Chuyển key bool thành string trong result_distribution trước khi dump json
            analysis_serializable = dict(analysis)
            analysis_serializable["result_distribution"] = {
                str(k): v for k, v in analysis["result_distribution"].items()
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(analysis_serializable, f, indent=2, ensure_ascii=False)
            print(f"\n[+] Đã lưu báo cáo JSON vào: {out_path.resolve()}")
            
    except Exception as e:
        print(f"Lỗi: {e}")


if __name__ == "__main__":
    main()
