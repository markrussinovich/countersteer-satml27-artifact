#!/usr/bin/env python
"""Aggregate writer parse/refusal counts over AutoDojo prompt logs (tripwire input).

Prints one line: `writer=<n> parse_fail=<n> refusal=<n> analyzer=<n> analyzer_empty=<n>`
over every prompt_log_*.jsonl under the given root (default runs/autodojo/full).
The owner tripwire (2026-09-05): systemic writer refusal rate >25% over the first
completed model-arm pauses the run.
"""
import glob
import json
import os
import re
import sys

_REFUSAL = re.compile(r"^\s*(i can['’]?t|i cannot|i won['’]?t|"
                      r"i['’]?m (sorry|not able)|sorry[,.])", re.IGNORECASE)

root = sys.argv[1] if len(sys.argv) > 1 else "runs/autodojo/full"
w = pf = rf = an = ae = 0
for p in glob.glob(os.path.join(root, "**", "prompt_log_*.jsonl"), recursive=True):
    for line in open(p):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("role") == "writer":
            w += 1
            if not r.get("parsed"):
                pf += 1
            if _REFUSAL.match((r.get("output") or "").strip()):
                rf += 1
        elif r.get("role") == "analyzer":
            an += 1
            if not (r.get("output") or "").strip():
                ae += 1
print(f"writer={w} parse_fail={pf} refusal={rf} analyzer={an} analyzer_empty={ae}")
