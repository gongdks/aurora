import ast, sys

files = ['app_qt.py', 'agent/ui/styles.py', 'agent/tools/doc_reader.py', 'agent/tools/file_reader.py']
ok = True
results = []
for f in files:
    try:
        ast.parse(open(f, encoding='utf-8').read())
        results.append(f'{f}: OK')
    except SyntaxError as e:
        results.append(f'{f}: SYNTAX ERROR - {e}')
        ok = False

with open('_syntax_result.txt', 'w', encoding='utf-8') as out:
    out.write('\n'.join(results))

sys.exit(0 if ok else 1)