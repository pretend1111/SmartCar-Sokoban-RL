import os, json
root = r'e:\大二下\SmartCar-Sokoban-RL'
for p in range(1, 7):
    d = os.path.join(root, 'assets', 'maps', f'phase{p}')
    if os.path.isdir(d):
        count = len([f for f in os.listdir(d) if f.endswith('.txt')])
        print(f"Phase {p}: {count} map files")
    else:
        print(f"Phase {p}: no directory")
manifest_path = os.path.join(root, 'assets', 'maps', 'batch_manifest.json')
if os.path.exists(manifest_path):
    m = json.load(open(manifest_path, 'r', encoding='utf-8'))
    print("\nManifest entries:")
    for p in range(1, 7):
        key = f'phase{p}'
        if key in m:
            print(f"  {key}: {m[key].get('count', '?')} maps recorded")
        else:
            print(f"  {key}: NOT in manifest")
