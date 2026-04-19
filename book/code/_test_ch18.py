"""Test harness for chapter 18. Extracts python blocks and runs them.
Safe to delete; used only for end-to-end verification."""
import re
import pathlib
import sys
import os
import time

os.chdir("/Users/mikenguyen/Downloads/credit_score/book/chapters")
sys.path.insert(0, '/Users/mikenguyen/Downloads/credit_score/book/code')

src = pathlib.Path(
    "/Users/mikenguyen/Downloads/credit_score/book/chapters/18-open-banking.qmd"
).read_text()

blocks = re.findall(r"```\{python.*?\}\n(.*?)```", src, re.DOTALL)
print(f"found {len(blocks)} python blocks")

ns = {}
for i, code in enumerate(blocks):
    label_m = re.search(r"#\|\s*label:\s*(\S+)", code)
    label = label_m.group(1) if label_m else f"block{i}"
    if re.search(r"#\|\s*eval:\s*false", code):
        print(f"[skip] {label}")
        continue
    print(f"[run ] {label}")
    t0 = time.time()
    try:
        exec(code, ns)
    except Exception as e:
        print(f"  FAIL {label}: {type(e).__name__}: {e}")
        raise
    print(f"  ok   {label}  {time.time() - t0:.1f}s")
