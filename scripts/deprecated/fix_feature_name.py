"""Rename precipitation_x_peak to precipitation_x_temp in future_prediction.py."""

print("Fixing precipitation_x_temp in future_prediction.py...")

with open('future_prediction.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_name = "features['precipitation_x_peak']"
new_name = "features['precipitation_x_temp']"

if old_name in content:
    content = content.replace(old_name, new_name)
    print(f'Changed {old_name} to {new_name}')

    with open('future_prediction.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print('File saved')
else:
    print('Feature name not found - checking alternatives...')

    if "precipitation_x" in content:
        print('Found precipitation_x in file')
        import re
        matches = re.findall(r"features\['precipitation_x_\w+'\]", content)
        print('Current usage:', matches)
    else:
        print('precipitation_x not found at all')
