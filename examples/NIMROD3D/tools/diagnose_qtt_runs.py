#!/usr/bin/env python3
"""Diagnose QTT allaxes scaling runs.
Produces a TSV report with columns:
run_name
diags_exists
diags_lines
diags_bytes
diags_last
slurm_files_found (semicolon-separated)
slurm_keywords (comma-separated)
status_guess
"""

from pathlib import Path
from collections import deque
import re
import sys

BASE = Path('examples/NIMROD3D/qtt_allaxes_scaling')
SLURM_DIR = BASE / 'slurm_output'

if not BASE.exists():
    print(f"Base directory not found: {BASE}", file=sys.stderr)
    raise SystemExit(1)

# load slurm outputs once (map file -> content last_lines)
slurm_files = []
if SLURM_DIR.exists():
    slurm_files = sorted(SLURM_DIR.glob('*.out'))

# pre-read slurm files tail and lowercase content for keyword search
SLURM_CACHE = {}
TAIL_LINES = 200
for f in slurm_files:
    try:
        with f.open('r', errors='ignore') as fh:
            dq = deque(maxlen=TAIL_LINES)
            for ln in fh:
                dq.append(ln.rstrip())
        SLURM_CACHE[f] = '\n'.join(dq).lower()
    except Exception as e:
        SLURM_CACHE[f] = ''

keywords = ['oom', 'out of memory', 'time_limit', 'time limit', 'killed', 'traceback', 'exception', 'error', 'segmentation fault']

out_lines = []
header = ['run_name','diags_exists','diags_lines','diags_bytes','diags_last','slurm_files_found','slurm_keywords','status_guess']
out_lines.append('\t'.join(header))

for child in sorted(BASE.iterdir()):
    if not child.is_dir():
        continue
    name = child.name
    # find epochs_diags_*.txt
    diags = None
    for p in child.glob('epochs_diags_*.txt'):
        diags = p
        break
    diags_exists = 'no' if diags is None else 'yes'
    diags_lines = ''
    diags_bytes = ''
    diags_last = ''
    if diags is not None:
        try:
            size = diags.stat().st_size
            diags_bytes = str(size)
            # count non-empty lines and capture last non-empty
            last_nonempty = ''
            count = 0
            with diags.open('r', errors='ignore') as fh:
                for ln in fh:
                    s = ln.strip()
                    if s:
                        last_nonempty = s
                        count += 1
            diags_lines = str(count)
            diags_last = last_nonempty
        except Exception as e:
            diags_lines = 'ERR'
            diags_bytes = 'ERR'
            diags_last = str(e)

    # search slurm outputs for this run name
    found_files = []
    found_keywords = set()
    for f, content in SLURM_CACHE.items():
        if name.lower() in content:
            found_files.append(str(f))
            for kw in keywords:
                if kw in content:
                    found_keywords.add(kw)
    slurm_files_found = ';'.join(found_files)
    slurm_keywords = ','.join(sorted(found_keywords))

    # also quick scan for keywords inside run directory files (recent logs)
    for lf in child.glob('*.log'):
        try:
            text = lf.read_text(errors='ignore').lower()
            for kw in keywords:
                if kw in text:
                    found_keywords.add(kw)
        except Exception:
            pass

    # status guess
    status = 'ok'
    if diags is None:
        status = 'missing_diags'
    else:
        try:
            if int(diags_lines) <= 1:
                status = 'incomplete_diags'
        except Exception:
            status = 'diags_err'
    if found_keywords:
        # more severe
        if 'oom' in found_keywords or 'out of memory' in found_keywords or 'segmentation fault' in found_keywords:
            status = 'oom'
        elif 'time_limit' in found_keywords or 'time limit' in found_keywords or 'killed' in found_keywords:
            status = 'time_limit_or_killed'
        else:
            # traceback/exception/error
            if status == 'ok':
                status = 'slurm_error'
            else:
                status += '+slurm_error'

    out_lines.append('\t'.join([
        name,
        diags_exists,
        diags_lines or '',
        diags_bytes or '',
        diags_last.replace('\t',' ')[:200] if diags_last else '',
        slurm_files_found,
        slurm_keywords,
        status
    ]))

report_path = BASE / 'diagnose_report.tsv'
report_path.write_text('\n'.join(out_lines))
print(f'Wrote report to {report_path}')
