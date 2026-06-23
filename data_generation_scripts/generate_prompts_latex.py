#!/usr/bin/env python
"""
generate_prompts_latex.py

Generates prompts.tex by importing the named prompt strings directly from
run_prompting_stages.py.  Strings that are f-string templates built inside
loops (payoff tables) or embedded inside functions (extractor / verifier
footer) are defined here as plain string literals and must be kept in sync
with the source manually if changed.

Run from the project directory:
    python generate_prompts_latex.py
"""

import os
import sys
import site
from unittest.mock import MagicMock

# ── Import run_prompting_stages without loading vllm ─────────────────────────
# Protect the generator's own sys.path from the path-stripping at the top of
# run_prompting_stages.py.
_saved_path      = sys.path[:]
_saved_user_site = site.ENABLE_USER_SITE

sys.modules["vllm"] = MagicMock()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_prompting_stages as rps

sys.path        = _saved_path
site.ENABLE_USER_SITE = _saved_user_site

# ── Strings not importable from the module ────────────────────────────────────
# Payoff tables are f-strings built inside a for-loop; shown here with literal
# {s}/{t} placeholders.  Keep in sync with run_prompting_stages.py.

PAYOFF_0 = """\
If you choose A and the other player chooses A. You earn 10 points, the other player earns 10 points.
If you choose A and the other player chooses B. You earn {s} points, the other player earns {t} points.
If you choose B and the other player chooses A. You earn {t} points, the other player earns {s} points.
If you choose B and the other player chooses B. You earn 5 points, the other player earns 5 points."""

PAYOFF_1 = """\
If you choose B and the other player chooses B. You earn 10 points, the other player earns 10 points.
If you choose B and the other player chooses A. You earn {s} points, the other player earns {t} points.
If you choose A and the other player chooses B. You earn {t} points, the other player earns {s} points.
If you choose A and the other player chooses A. You earn 5 points, the other player earns 5 points."""

PAYOFF_0_Q = PAYOFF_0 + "\n\nWhat do you choose? A or B?"
PAYOFF_1_Q = PAYOFF_1 + "\n\nWhat do you choose? A or B?"

# Inline strings from build_main_prompt
MAIN_SYS        = "You are a helpful assistant."
SUFFIX_SIMPLE   = "Choose between A and B. Answer only with A or B depending on your choice."
SUFFIX_EXTRACT  = "Make a choice and explain why"

# Inline strings from run_extract_or_multi / run_final
EXTRACTOR_SYS   = "You are a helpful assistant."
EXTRACTOR_USR1  = "The player who was asked to choose between A and B answered: {main_model_output}"
EXTRACTOR_USR2  = "What did the person who wrote the message choose? Answer shortly."

# Inline string from run_final
VERIFIER_SYS    = "You are a logical verifier. Your purpose is to look for inconsistencies and errors in a given text."

# Footer appended inside build_verifier_message (f-string; variables shown as placeholders)
VERIFIER_FOOTER = """\
Rules of the game to analyze:
{points}

Answer of Player 1, to analyze:
{answer1}

Important:
- Output ONLY one word: good or bad
- Do not add punctuation, extra spaces, or explanations.
"""

# Example block template (loop body inside build_verifier_message)
EXAMPLE_BLOCK = """\
Example {i}, {good|bad} reasoning:

Rules of Example {i}:

{rules text}

Answer of Example {i}:

{answer text}

Output of Example {i}: {good|bad}"""

# Short descriptions for the 10 verifier pool entries (in pool order)
EXAMPLE_DESCS = [
    "bad, S=7 T=6, chose B when A strictly dominates",
    "good, S=10 T=6, chose A cooperate",
    "bad, S=9 T=15, fabricated probabilities",
    "good, S=8 T=5, chose B cooperate in script-1",
    "bad, S=10 T=7, incomplete reasoning in script-1",
    "good, S=1 T=15, chose B defect",
    "good, S=0 T=12, chose B defect",
    "bad, S=2 T=12, chose A cooperate when B strictly dominates",
    "good, S=4 T=13, chose A defect in script-1",
    "bad, S=3 T=11, chose B cooperate when A strictly dominates",
]

# ── LaTeX helpers ─────────────────────────────────────────────────────────────

def lst(content):
    return "\\begin{lstlisting}\n" + content + "\n\\end{lstlisting}"

def sec(title):
    bar = "% " + "=" * 60
    return bar + "\n\\section*{" + title + "}\n" + bar

def subsec(title):
    return "\\subsection*{" + title + "}"

def subsubsec(title):
    return "\\subsubsection*{" + title + "}"

def note(text):
    return text

# ── Document assembly ─────────────────────────────────────────────────────────

B = []  # list of text blocks; joined with \n\n at the end

# B.append(r"""\documentclass{article}
# \usepackage{listings}
# \usepackage[margin=2.5cm]{geometry}
# \lstset{
#   basicstyle=\ttfamily\small,
#   breaklines=true,
#   frame=single,
#   keepspaces=true,
#   columns=fullflexible,
# }

# \begin{document}""")

# ── Shared components ─────────────────────────────────────────────────────────

B.append(sec("Shared Components"))

B.append(subsec("Main model system message (all stages)"))
B.append(lst(MAIN_SYS))

B.append(subsec("Instructions (all stages, User turn 1)"))
B.append(lst(rps.instructions_script_short))

B.append(subsec("Payoff table --- script-0 variant (A = cooperate)"))
B.append(lst(PAYOFF_0))

B.append(subsec("Payoff table --- script-1 variant (B = cooperate)"))
B.append(lst(PAYOFF_1))

B.append(subsec("Reasoning steps (multi and final stages, appended to payoff table)"))
B.append(lst(rps.steps))

# ── Stage: simple ─────────────────────────────────────────────────────────────

B.append(sec("Stage: simple"))
B.append(note(
    "One call to the main model. No extractor. Uses the \\texttt{\\_q} payoff variants\n"
    "(which include the closing question) plus the constrained suffix."
))

B.append(subsec("System message"))
B.append(lst("[main model system message -- see above]"))

B.append(subsec("User turn 1"))
B.append(lst("[instructions -- see above]"))

B.append(subsec("User turn 2 --- script-0-q variant"))
B.append(lst("[payoff table script-0 -- see above]\n\nWhat do you choose? A or B?\n\n" + SUFFIX_SIMPLE))

B.append(subsec("User turn 2 --- script-1-q variant"))
B.append(lst("[payoff table script-1 -- see above]\n\nWhat do you choose? A or B?\n\n" + SUFFIX_SIMPLE))

# ── Stage: extract ────────────────────────────────────────────────────────────

B.append(sec("Stage: extract"))
B.append(note("Two calls: main model, then Qwen extractor."))

B.append(subsec("Main model --- System"))
B.append(lst("[main model system message -- see above]"))

B.append(subsec("Main model --- User turn 1"))
B.append(lst("[instructions -- see above]"))

B.append(subsec("Main model --- User turn 2 (script-0)"))
B.append(lst("[payoff table script-0 -- see above]\n\n" + SUFFIX_EXTRACT))

B.append(subsec("Main model --- User turn 2 (script-1)"))
B.append(lst("[payoff table script-1 -- see above]\n\n" + SUFFIX_EXTRACT))

B.append(subsec("Qwen extractor --- System"))
B.append(lst(EXTRACTOR_SYS))

B.append(subsec("Qwen extractor --- User turn 1"))
B.append(lst(EXTRACTOR_USR1))

B.append(subsec("Qwen extractor --- User turn 2"))
B.append(lst(EXTRACTOR_USR2))

# ── Stage: multi ──────────────────────────────────────────────────────────────

B.append(sec("Stage: multi"))
B.append(note(
    "Same as extract but reasoning steps are appended to the payoff table\n"
    "instead of the free-form suffix."
))

B.append(subsec("Main model --- System"))
B.append(lst("[main model system message -- see above]"))

B.append(subsec("Main model --- User turn 1"))
B.append(lst("[instructions -- see above]"))

B.append(subsec("Main model --- User turn 2 (script-0)"))
B.append(lst("[payoff table script-0 -- see above]\n\n[reasoning steps -- see above]"))

B.append(subsec("Main model --- User turn 2 (script-1)"))
B.append(lst("[payoff table script-1 -- see above]\n\n[reasoning steps -- see above]"))

B.append(subsec("Qwen extractor"))
B.append(lst("[same as extract stage -- see above]"))

# ── Stage: final ──────────────────────────────────────────────────────────────

B.append(sec("Stage: final"))
B.append(note(
    "Three calls per iteration: main model (same as multi), then Qwen verifier,\n"
    "then Qwen extractor (only for games that passed the verifier)."
))

B.append(subsec("Main model"))
B.append(lst("[same prompts as multi stage -- see above]"))

B.append(subsec("Qwen verifier --- System"))
B.append(lst(VERIFIER_SYS))

B.append(subsec("Qwen verifier --- User turn"))
B.append(note(
    "The verifier message is assembled as:\n"
    "\\texttt{\\_VERIFIER\\_CRITERIA} + \\texttt{instructions\\_script\\_short}"
    " + 5 sampled examples + footer."
))

B.append(subsubsec("Criteria block (\\texttt{\\_VERIFIER\\_CRITERIA})"))
B.append(lst(rps._VERIFIER_CRITERIA))

B.append(subsubsec("Instructions block (\\texttt{instructions\\_script\\_short})"))
B.append(lst("[instructions -- see above]"))

B.append(subsubsec("Example block (repeated for each of 5 sampled examples)"))
B.append(lst(EXAMPLE_BLOCK))

B.append(subsubsec("Footer block"))
B.append(lst(VERIFIER_FOOTER))

# ── Verifier example pool ─────────────────────────────────────────────────────

for i, (ex, desc) in enumerate(zip(rps.VERIFIER_EXAMPLE_POOL, EXAMPLE_DESCS), 1):
    B.append(subsec(f"Verifier example pool --- Example {i} ({desc})"))
    B.append(lst(ex["rules"]))
    B.append(lst(ex["answer"]))
    B.append("\\noindent Output: \\texttt{" + ex["label"] + "}")

# ── Extractor (final stage) ───────────────────────────────────────────────────

B.append(subsec("Qwen extractor (final stage)"))
B.append(lst(
    "[same as extract/multi stages -- see above;"
    " only called for games that passed the verifier]"
))

# B.append("\\end{document}")

# ── Write file ────────────────────────────────────────────────────────────────

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.tex")
with open(output_path, "w") as f:
    f.write("\n\n".join(B) + "\n")

print(f"Written to {output_path}")
