import os
import subprocess
import sys

class DockerSandbox:
    @staticmethod
    def run(code, test_cases, timeout=15):
        full_code = f"{code}\n\n{test_cases}"
        
        # Kiểm tra an toàn tĩnh (Static safety guardrails)
        dangerous_calls = [
            "os.system", "subprocess.", "shutil.", "os.remove", "os.rmdir", 
            "os.unlink", "pathlib.Path.unlink", "builtins.eval", "exec("
        ]
        for danger in dangerous_calls:
            if danger in full_code:
                return False, f"Lỗi bảo mật: Phát hiện mã nguồn chứa thao tác hệ thống nguy hiểm ({danger})."

        tmp_file_path = os.path.abspath("temp_code.py")
        with open(tmp_file_path, "w", encoding="utf-8") as f:
            f.write(full_code)

        try:
            result = subprocess.run(
                [sys.executable, tmp_file_path],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            success = (result.returncode == 0)
            output = result.stdout
            if result.stderr:
                output += "\n[LỖI]: " + result.stderr
                
            return success, output
            
        except subprocess.TimeoutExpired:
            return False, "Timeout: Code chạy quá lâu (phát hiện vòng lặp vô tận)!"
        except Exception as e:
            return False, f"Lỗi thực thi hệ thống: {str(e)}"
        finally:
            if os.path.exists(tmp_file_path):
                try:
                    os.remove(tmp_file_path)
                except Exception:
                    pass