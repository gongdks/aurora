import sys

with open("check_checkpoint.txt", "w") as out:
    import langgraph.checkpoint
    out.write(f"langgraph.checkpoint path: {langgraph.checkpoint.__path__}\n\n")
    
    import pkgutil
    for importer, modname, ispkg in pkgutil.iter_modules(langgraph.checkpoint.__path__):
        out.write(f"  - {modname} (pkg={ispkg})\n")
    
    out.write("\n")
    
    try:
        from langgraph.checkpoint.memory import MemorySaver
        out.write("OK: MemorySaver works\n")
    except Exception as e:
        out.write(f"FAIL MemorySaver: {e}\n")
    
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        out.write("OK: SqliteSaver works\n")
    except Exception as e:
        out.write(f"FAIL SqliteSaver: {e}\n")
    
    out.write("\nSearching for SqliteSaver...\n")
    import importlib
    for modname in ["langgraph.checkpoint", "langgraph.checkpoint.memory", "langgraph.checkpoint.sqlite", "langgraph_saver.sqlite", "langgraph.checkpoint.base"]:
        try:
            mod = importlib.import_module(modname)
            out.write(f"  {modname}: OK -> {[x for x in dir(mod) if 'sql' in x.lower() or 'save' in x.lower() or 'Saver' in x]}\n")
        except Exception as e:
            out.write(f"  {modname}: {e}\n")