import os
fp = os.path.join(os.getcwd(), 'smartcar_sokoban', 'solver', 'explorer.py')
with open(fp, 'r', encoding='utf-8') as f:
    c = f.read()
# Remove the mid-path break check
old = """        for dx, dy in path:\r\n            if not _entity_scan_still_needed(state, etype, eidx):\r\n                break  # \u76ee\u6807\u5df2\u88ab\u9014\u4e2d\u53d1\u73b0 / \u6392\u9664\u6cd5\u4e0d\u9700\u8981\u4e86\r\n            a = direction_to_action(dx, dy)"""
new = """        for dx, dy in path:\r\n            a = direction_to_action(dx, dy)"""
if old in c:
    c = c.replace(old, new)
    with open(fp, 'w', encoding='utf-8') as f:
        f.write(c)
    print("PATCHED OK")
else:
    print("Target not found, trying without \\r\\n")
    old2 = old.replace('\r\n', '\n')
    new2 = new.replace('\r\n', '\n')
    if old2 in c:
        c = c.replace(old2, new2)
        with open(fp, 'w', encoding='utf-8') as f:
            f.write(c)
        print("PATCHED OK (LF)")
    else:
        print("ERROR: pattern not found")
