import sys
import subprocess
import importlib
import threading
import re
import time
import io

class Sandbox:
    @staticmethod
    def install_package(package_name):
        """Tự động cài đặt thư viện nếu thiếu"""
        print(f"📦 Đang kiểm tra/cài đặt thư viện: {package_name}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name, "--quiet"])
            return True
        except Exception as e:
            print(f"❌ Không thể cài đặt {package_name}: {e}")
            return False

    @staticmethod
    def run(code, test_cases, timeout=10):
        """
        Thực thi mã nguồn và test cases trong môi trường kiểm soát.
        """
        # 1. Tự động kiểm tra và cài đặt thư viện dựa trên import
        all_text = code + "\n" + test_cases
        libs = re.findall(r'^(?:from|import)\s+([\w\d_]+)', all_text, re.MULTILINE)
        for lib in set(libs):
            # Bỏ qua các thư viện chuẩn của Python thường gặp
            if lib in ('math', 'datetime', 're', 'os', 'sys', 'time', 'io', 'collections', 'itertools'):
                continue
            try:
                importlib.import_module(lib)
            except ImportError:
                Sandbox.install_package(lib)

        # 2. Chuẩn bị môi trường và bộ chứa kết quả
        result_container = {
            "success": False, 
            "output": "Timeout: Code chạy quá thời gian quy định!"
        }

        def target():
            # Chặn stdout để không làm rối console và có thể kiểm tra print nếu cần
            old_stdout = sys.stdout
            redirected_output = io.StringIO()
            sys.stdout = redirected_output
            
            try:
                # Tạo không gian tên sạch cho mỗi lần chạy
                # Cho phép truy cập các hàm built-in và một số thư viện phổ biến
                exec_globals = {
                    "__builtins__": __builtins__,
                    "math": __import__("math"),
                    "datetime": __import__("datetime"),
                    "re": __import__("re"),
                    "os": __import__("os"),
                    "time": time,
                    "io": io
                }
                
                # Nối code của Coder và Test của Tester
                full_script = f"{code}\n\n# --- EXECUTING TESTS ---\n{test_cases}"
                
                # Thực thi
                exec(full_script, exec_globals)
                
                result_container["success"] = True
                result_container["output"] = "Passed all tests"
            except AssertionError as e:
                # Lấy nội dung lỗi assert nếu AI có viết (ví dụ: assert x == y, "Lỗi")
                error_msg = str(e) if str(e) else "Giá trị thực tế không khớp với kỳ vọng."
                result_container["output"] = f"AssertionError: {error_msg}"
            except NameError as e:
                # Thường xảy ra khi Tester gọi biến cục bộ của Coder (như lỗi 'uppercase' bạn gặp)
                result_container["output"] = f"Lỗi tham chiếu (NameError): {str(e)}. Gợi ý: Kiểm tra lại tên biến hoặc phạm vi truy cập."
            except Exception as e:
                result_container["output"] = f"Lỗi thực thi ({type(e).__name__}): {str(e)}"
            finally:
                sys.stdout = old_stdout

        # 3. Chạy trong Thread để quản lý Timeout (Infinite Loop Protection)
        thread = threading.Thread(target=target)
        thread.daemon = True 
        thread.start()
        thread.join(timeout)

        if thread.is_alive():
            return False, "❌ LỖI: Code bị treo hoặc chạy quá lâu (có thể do vòng lặp vô tận)."
        
        return result_container["success"], result_container["output"]