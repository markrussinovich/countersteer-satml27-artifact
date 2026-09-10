#!/usr/bin/env python
"""Activation steering against cross-prompt injection (XPIA) on tool output.

THIS FILE IS A COMPATIBILITY SHIM. The implementation now lives in `src/`, split by
functionality (see `src/__init__.py` for the layering). Everything is re-exported here
so that existing call sites -- `import xpia_defense as X` in tools/, and the probe pickles'
`__main__.TorchLogReg` alias -- keep working without a rewrite.

Run it exactly as before:

    python xpia_defense.py --model openai/gpt-oss-20b --stage sweep --corpus shipped ...

ONE THING TO KNOW IF YOU MONKEYPATCH. Names bound inside `src/*` resolve in their OWN
module namespace, so `X.judge = <stub>` on this shim no longer reaches `arms.run_arm`. Patch
`src.judge.judge` instead. (`run_arm` takes `run_judge=False` by default, so the judge is
not called at all unless asked for -- see FINDINGS.md section 1.2.)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import (arms, cli, common, corpora, judge as _judge_mod, model,  # noqa: E402
                 moe, probes, scoring, spans, steering, templates)

# Re-export every public and underscore-prefixed name, in dependency order so a later module
# wins on the (currently empty) set of collisions. Underscore names are included deliberately:
# tools/controls/score_table.py reaches for `_is_freetext` and `_call_tainted`, and dropping
# them would break the canonical scorer silently.
for _mod in (common, model, moe, templates, corpora, probes, spans, steering,
             scoring, _judge_mod, arms, cli):
    for _name, _val in vars(_mod).items():
        if not _name.startswith("__"):
            globals()[_name] = _val

main = cli.main

if __name__ == "__main__":
    main()
