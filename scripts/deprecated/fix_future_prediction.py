"""
Replace the line-encoding section of FutureDelayPredictor._engineer_features
so the line index is appended as a numeric 21st feature rather than one-hot
encoded.
"""

import re

print("Reading future_prediction.py...")
with open('future_prediction.py', 'r', encoding='utf-8') as f:
    content = f.read()

pattern = r'(# === CONVERT TO ARRAY ===.*?)# === ADD LINE ENCODING ===.*?return full_features'

replacement = r'''\1# === ADD LINE AS SINGLE FEATURE ===
        # Encode line as single number (0-10 for 11 lines)
        lines = self._get_valid_lines()
        line_idx = float(lines.index(line)) if line in lines else 0.0

        # Append as 21st feature
        feature_array = np.append(feature_array, line_idx)

        return feature_array'''

new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

if new_content != content:
    with open('future_prediction.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    print('Updated future_prediction.py')
else:
    print('Pattern not found - needs manual edit')
    print('\nLook for this section and replace it:')
    print('# === ADD LINE ENCODING ===')
    print('line_encoded = self._encode_line(line)')
    print('...')
    print('return full_features')
