import subprocess
result = subprocess.run(
    [r"d:\AIWORK\PythonProject\.venv\Scripts\pip.exe", "install", "langgraph-checkpoint-sqlite"],
    capture_output=True, text=True
)
with open("pip_result.txt", "w") as f:
    f.write(f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}\n\nRC: {result.returncode}\n")