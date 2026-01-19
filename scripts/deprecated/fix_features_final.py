"""
Replace the return section of FutureDelayPredictor._engineer_features with
a version that builds a DataFrame keyed by feature name rather than by
positional array index.
"""

print("Applying final fix to future_prediction.py...")

with open('future_prediction.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

output = []
in_engineer_method = False
skip_until_next_def = False

for i, line in enumerate(lines):
    if 'def _engineer_features(' in line:
        in_engineer_method = True
        skip_until_next_def = False

    if in_engineer_method and 'def _is_peak_time' in line:
        in_engineer_method = False

    if in_engineer_method and '# === ADD LINE AS CATEGORICAL ===' in line:
        skip_until_next_def = True
        output.append('        # === RETURN AS DATAFRAME ===\n')
        output.append('        if \'line\' not in feature_names:\n')
        output.append('            feature_names = list(feature_names) + [\'line\']\n')
        output.append('        \n')
        output.append('        features[\'line\'] = line\n')
        output.append('        \n')
        output.append('        df_dict = {}\n')
        output.append('        for name in feature_names:\n')
        output.append('            if name == \'line\':\n')
        output.append('                df_dict[name] = [line]\n')
        output.append('            else:\n')
        output.append('                df_dict[name] = [features.get(name, 0.0)]\n')
        output.append('        \n')
        output.append('        return pd.DataFrame(df_dict)\n')
        continue

    if skip_until_next_def:
        if 'def _is_peak_time' in line:
            skip_until_next_def = False
            output.append(line)
        continue

    output.append(line)

with open('future_prediction.py', 'w', encoding='utf-8') as f:
    f.writelines(output)

print('Fixed!')
