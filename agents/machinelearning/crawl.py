import json
import re
import time
from bs4 import BeautifulSoup
from curl_cffi import requests

def camel_to_snake(name: str) -> str:
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def clean_html_text(html_content: str) -> str:
    if not html_content:
        return ""
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)

def parse_examples_from_html(html_content: str) -> list:
    if not html_content:
        return []
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text()
    
    pattern = r"Input:\s*(.*?)\n\s*Output:\s*(.*?)(?=\n\s*Explanation|\n\s*Example|\n\s*Constraints|\n\s*Follow-up|$)"
    matches = re.findall(pattern, text, re.DOTALL)
    
    assert_statements = []
    for inp, out in matches:
        inp_clean = inp.strip().replace('\n', ' ')
        out_clean = out.strip().replace('\n', ' ')
        
        args = []
        for part in inp_clean.split(','):
            if '=' in part:
                val = part.split('=', 1)[1].strip()
                args.append(val)
            else:
                if part.strip():
                    args.append(part.strip())
                
        args_str = ", ".join(args)
        if args_str and out_clean:
            assert_statements.append(f"    assert candidate({args_str}) == {out_clean}")
            
    return assert_statements

def get_all_leetcode_slugs(total_limit=10) -> list:
    """Cập nhật trường paidOnly -> isPaidOnly."""
    url = "https://leetcode.com/graphql"
    
    query = """
    query problemsetQuestionList($categorySlug: String, $limit: Int, $skip: Int, $filters: QuestionListFilterInput) {
      problemsetQuestionList: questionList(
        categorySlug: $categorySlug
        limit: $limit
        skip: $skip
        filters: $filters
      ) {
        total: totalNum
        data {
          titleSlug
          isPaidOnly
        }
      }
    }
    """
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://leetcode.com/problemset/all/",
        "Content-Type": "application/json",
        "Origin": "https://leetcode.com"
    }
    
    slugs = []
    page_size = 50
    skip = 0
    
    while len(slugs) < total_limit:
        current_limit = min(page_size, total_limit - len(slugs))
        
        payload = {
            "query": query,
            "variables": {
                "categorySlug": "all-code-essentials",
                "skip": skip,
                "limit": current_limit,
                "filters": {}
            }
        }
        
        res = requests.post(url, json=payload, headers=headers, impersonate="chrome120")
        
        if res.status_code != 200:
            print(f"Lỗi HTTP {res.status_code}: {res.text}")
            break
            
        data = res.json()
        if "errors" in data:
            print(f"Lỗi GraphQL API: {data['errors']}")
            break
            
        q_data = data.get('data', {}).get('problemsetQuestionList', {})
        questions = q_data.get('data', [])
        
        if not questions:
            break
            
        for q in questions:
            if not q.get('isPaidOnly'):
                slugs.append(q['titleSlug'])
                
        skip += len(questions)
        time.sleep(0.5)
        
    return slugs

def leetcode_to_humaneval(title_slug: str) -> dict:
    url = "https://leetcode.com/graphql"
    query = """
    query getQuestionDetail($titleSlug: String!) {
      question(titleSlug: $titleSlug) {
        questionId
        title
        titleSlug
        content
        codeSnippets {
          langSlug
          code
        }
      }
    }
    """
    payload = {
        "query": query,
        "variables": {"titleSlug": title_slug}
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": f"https://leetcode.com/problems/{title_slug}/"
    }
    
    res = requests.post(url, json=payload, headers=headers, impersonate="chrome120")
    if res.status_code != 200:
        raise Exception(f"HTTP {res.status_code}")
        
    data = res.json().get('data', {}).get('question')
    if not data or not data.get('content'):
        raise Exception("Không tìm thấy nội dung bài toán (Có thể là bài khóa Premium)")

    snippets = data.get('codeSnippets') or []
    py3_snippet = next((item['code'] for item in snippets if item['langSlug'] == 'python3'), None)
    if not py3_snippet:
        raise Exception("Không hỗ trợ Python3 snippet")

    method_match = re.search(r'def\s+([a-zA-Z0-9_]+)\s*\(\s*self\s*,?\s*(.*?)\)', py3_snippet)
    if not method_match:
        raise Exception("Không parse được signature của class Solution")

    original_func_name = method_match.group(1)
    params = method_match.group(2)
    entry_point = camel_to_snake(original_func_name)
    
    docstring = clean_html_text(data['content'])
    formatted_docstring = "\n".join(["    " + line for line in docstring.splitlines()])
    prompt = f"from typing import List, Optional, Dict, Set, Tuple\n\ndef {entry_point}({params}):\n    \"\"\"\n{formatted_docstring}\n    \"\"\"\n"

    asserts = parse_examples_from_html(data['content'])
    if asserts:
        test_code = "def check(candidate):\n" + "\n".join(asserts) + "\n"
    else:
        test_code = "def check(candidate):\n    pass\n"

    return {
        "task_id": f"LeetCode/{data['questionId']}_{data['titleSlug']}",
        "prompt": prompt,
        "entry_point": entry_point,
        "canonical_solution": "    pass\n",
        "test": test_code
    }

def crawl_all_leetcode(limit=10, delay=1.0):
    print(f"--> Đang lấy danh sách {limit} bài toán...")
    slugs = get_all_leetcode_slugs(total_limit=limit)
    print(f"--> Lấy thành công {len(slugs)} bài toán khả thi.")

    success, fail = 0, 0
    for idx, slug in enumerate(slugs, 1):
        print(f"[{idx}/{len(slugs)}] Cào {slug}...", end=" ")
        try:
            item = leetcode_to_humaneval(slug)
            with open("leetcode_humaneval_all.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print("THÀNH CÔNG")
            success += 1
        except Exception as e:
            print(f"LỖI ({e})")
            fail += 1

        time.sleep(delay)

    print(f"\n[Hoàn thành] Success: {success} | Fail: {fail}")

if __name__ == "__main__":
    crawl_all_leetcode(limit=4000, delay=2.0)