#!/usr/bin/env python3
"""Patch start.sh: exclude the root-owned HF 'trees/' cache dir from the weight rsync."""
import sys

p = "start.sh"
s = open(p).read()
old = "rsync -a --partial --info=progress2,stats1 --exclude '.locks'"
new = old + " --exclude 'trees/'"
if old in s and "trees/" not in s.split("\n")[1834]:
    open(p, "w").write(s.replace(old, new, 1))
    print("PATCHED")
else:
    print("already patched or pattern missing")
sys.exit(0)
