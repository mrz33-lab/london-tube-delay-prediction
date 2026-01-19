"""
Replace the line-encoding section of FutureDelayPredictor._engineer_features
so the line name is passed as a categorical column rather than a numeric index.
"""

print("Fixing future_prediction.py to use line as categorical...")

with open('future_prediction.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_section = '''        # === ADD LINE AS SINGLE FEATURE ===
        lines = self._get_valid_lines()
        line_idx = float(lines.index(line)) if line in lines else 0.0
        feature_array = np.append(feature_array, line_idx)

        # Return as DataFrame
        return pd.DataFrame([feature_array], columns=feature_names)'''

new_section = '''        # === ADD LINE AS CATEGORICAL ===
        if 'line' not in feature_names:
            feature_names.append('line')

        features['line'] = line

        feature_dict = {name: [features.get(name, 0.0)] for name in feature_names if name != 'line'}
        feature_dict['line'] = [line]

        return pd.DataFrame(feature_dict)'''

if old_section in content:
    content = content.replace(old_section, new_section)
    print('Updated future_prediction.py')
else:
    if '# === ADD LINE AS SINGLE FEATURE ===' in content:
        import re
        pattern = r'# === ADD LINE AS SINGLE FEATURE ===.*?return pd\.DataFrame.*?\)'
        replacement = '''# === ADD LINE AS CATEGORICAL ===
        if 'line' not in feature_names:
            feature_names.append('line')

        features['line'] = line

        feature_dict = {name: [features.get(name, 0.0)] for name in feature_names if name != 'line'}
        feature_dict['line'] = [line]

        return pd.DataFrame(feature_dict)'''

        content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        print('Updated with regex')
    else:
        print('Pattern not found - needs manual edit')

with open('future_prediction.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Done!')
