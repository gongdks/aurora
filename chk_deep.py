import sys, os, importlib

with open("chk_deep.txt", "w") as f:
    # Check if the module file exists
    sqlite_dir = r"D:\AIWORK\PythonProject\.venv\Lib\site-packages\langgraph\checkpoint\sqlite"
    if os.path.isdir(sqlite_dir):
        f.write(f"sqlite dir exists: {sqlite_dir}\n")
        for root, dirs, files in os.walk(sqlite_dir):
            for fn in files:
                f.write(f"  {os.path.join(root, fn)}\n")
    else:
        f.write(f"sqlite dir NOT found: {sqlite_dir}\n")
    
    f.write("\n")
    
    # Try minimal import - just check the module can be found
    try:
        spec = importlib.util.find_spec("langgraph.checkpoint.sqlite")
        f.write(f"find_spec: {spec}\n")
    except Exception as e:
        f.write(f"find_spec error: {e}\n")
    
    # Try lazy import
    try:
        mod = importlib.import_module("langgraph.checkpoint.sqlite")
        f.write(f"import_module OK: {dir(mod)[:20]}\n")
        if hasattr(mod, 'SqliteSaver'):
            f.write("SqliteSaver found in module!\n")
    except Exception as e:
        f.write(f"import_module error: {type(e).__name__}: {e}\n")