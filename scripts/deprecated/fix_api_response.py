"""Patch api.py to hard-code the run_id rather than calling get_latest_run_id()."""

print("Fixing api.py response...")

with open('api.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace(
    'model_version=get_latest_run_id()',
    'model_version="run_20260210_153030"'
)

with open('api.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Fixed api.py')
