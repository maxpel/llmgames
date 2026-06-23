#!/usr/bin/env python
# coding: utf-8

import sys
import site
site.ENABLE_USER_SITE = False
sys.path = [p for p in sys.path if '.local' not in p]

"""
run_prompting_stages.py

Runs the four prompting pipeline stages from Palatsi et al. (arXiv:2511.04500)
at the paper's generation temperature (0.8), for a single model.

Stages:
  simple  — instructions + payoffs, no reasoning steps, direct regex extraction
             from main model's output (no Qwen involved)
  extract — instructions + payoffs, no steps, Qwen extraction (double extraction)
  multi   — instructions + payoffs + reasoning steps, Qwen extraction, no verifier
  final   — full pipeline: steps + Qwen verifier + Qwen extraction, with retry loop

All stages use A/B label randomization and the paper's 121-game subspace
(S in 0..10, T in 5..15). The first three stages are single-pass (no retry):
games that fail extraction in a given repetition are recorded as 0.

Usage:
    python run_prompting_stages.py \
        --model llama \
        --stage simple \
        --output_dir stage_results

Arguments:
    --model       : llama | qwen | mistral
    --stage       : simple | extract | multi | final
    --repetitions : number of repetitions (default: 20)
    --output_dir  : directory for output files

Output files:
    {output_dir}/{model}_{stage}_raw.npy    shape: (repetitions, 11, 11)
    {output_dir}/{model}_{stage}_mean.txt   mean cooperation rate per game
"""

import argparse
import os
import re
import textwrap
import time
import numpy as np
import gc

from vllm import LLM, SamplingParams

# ── Argument parsing ────────────────────────────────────────────────────────────

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       type=str, required=True,
                        choices=["llama", "qwen", "mistral"])
    parser.add_argument("--stage",       type=str, required=True,
                        choices=["simple", "extract", "multi", "final"])
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--output_dir",  type=str, default="stage_results")
    parser.add_argument("--debug",       action="store_true",
                        help="Run 1 rep on 5 games only")
    return parser.parse_args()

# ── Model paths ─────────────────────────────────────────────────────────────────

MODEL_PATHS = {
    "llama":   "meta-llama/Llama-3.1-8B-Instruct",
    "qwen":    "Qwen/Qwen2.5-7B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
}

EXTRACTOR_PATH = MODEL_PATHS["qwen"]

# ── Game space: paper's 121 games (S 0..10, T 5..15) ───────────────────────────

S_VALUES = np.arange(0, 11)
T_VALUES = np.arange(5, 16)
N_S      = len(S_VALUES)
N_T      = len(T_VALUES)

game_order          = []
all_values_script_0 = []   # A = cooperate
all_values_script_1 = []   # B = cooperate
# Variants with closing question for simple stage (all original simple scripts include it)
all_values_script_0_q = []
all_values_script_1_q = []

for s in S_VALUES:
    for t in T_VALUES:
        game_order.append([s, t])

        all_values_script_0.append(textwrap.dedent(f"""\
        If you choose A and the other player chooses A. You earn 10 points, the other player earns 10 points.
        If you choose A and the other player chooses B. You earn {s} points, the other player earns {t} points.
        If you choose B and the other player chooses A. You earn {t} points, the other player earns {s} points.
        If you choose B and the other player chooses B. You earn 5 points, the other player earns 5 points."""))

        all_values_script_1.append(textwrap.dedent(f"""\
        If you choose B and the other player chooses B. You earn 10 points, the other player earns 10 points.
        If you choose B and the other player chooses A. You earn {s} points, the other player earns {t} points.
        If you choose A and the other player chooses B. You earn {t} points, the other player earns {s} points.
        If you choose A and the other player chooses A. You earn 5 points, the other player earns 5 points."""))

        all_values_script_0_q.append(textwrap.dedent(f"""\
        If you choose A and the other player chooses A. You earn 10 points, the other player earns 10 points.
        If you choose A and the other player chooses B. You earn {s} points, the other player earns {t} points.
        If you choose B and the other player chooses A. You earn {t} points, the other player earns {s} points.
        If you choose B and the other player chooses B. You earn 5 points, the other player earns 5 points.

        What do you choose? A or B?"""))

        all_values_script_1_q.append(textwrap.dedent(f"""\
        If you choose B and the other player chooses B. You earn 10 points, the other player earns 10 points.
        If you choose B and the other player chooses A. You earn {s} points, the other player earns {t} points.
        If you choose A and the other player chooses B. You earn {t} points, the other player earns {s} points.
        If you choose A and the other player chooses A. You earn 5 points, the other player earns 5 points.

        What do you choose? A or B?"""))

N_GAMES = len(game_order)   # 121

# ── Prompts (verbatim from paper) ───────────────────────────────────────────────

instructions_script_short = """\
        This one-shot game is made of only one round with an anonymous player. You will play only once.

        To play you must choose one of two options: A and B, the other player will also choose between A and B. Both players are subjected to the same rules.

        You won't know the other player's decision until the end of the round so you won't be able to change your choice after knowing the other player's decision.
        The other player won't know your decision until the end of the round so they won't be able to change their choice after knowing yours.

        You will be playing simultaneously with the other player.

        Both players will receive a monetary prize. The monetary prize for the player will be the amount of points the player has earned multiplied by 10.

        For example, if player 1 earns 9 points and player 2 earns 5 points.
        Player 1 will receive 9*10 = 90 euros and player 2 will receive 10*5 = 50 euros.

        The monetary prize of each player only depends on their number of points.
        The difference in points of both players has no effect on the prize.

        Both players will receive the prize, regardless of who earned more or less money.
        The outcome is solely determined by the number of points earned by each player."""
instructions_script_short = textwrap.dedent(instructions_script_short)

steps = """\
1. There are four possible outcomes in this game, depending on your choice (A or B) and the other player's choice (A or B).
2. Group the outcomes based on your decision:
   - If you choose A: (A,A) and (A,B)
   - If you choose B: (B,A) and (B,B)
3. Analyze and compare the outcomes in each group step by step.
4. Make sure your analysis is consistent with the rules of the game and the possible outcomes.
5. Based on your analysis, choose either A or B."""

# ── Chat template converters (verbatim from paper) ──────────────────────────────

def convert_messages_to_prompt_llama(messages):
    prompt = "<|begin_of_text|>\n"
    for msg in messages:
        prompt += (f"<|start_header_id|>{msg['role']}<|end_header_id|>\n"
                   f"{msg['content']}\n<|eot_id|>\n")
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n"
    return prompt

def convert_messages_to_prompt_qwen(messages):
    prompt = ""
    for msg in messages:
        prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt

def convert_messages_to_prompt_mistral(messages):
    prompt = ""
    for msg in messages:
        if msg["role"] == "user":
            prompt += "<s>[INST] " + msg["content"].strip() + " [/INST]"
        elif msg["role"] == "assistant":
            prompt += " " + msg["content"].strip() + " </s>"
    return prompt

def build_main_prompt(game_script, model, use_steps, constrained=False):
    if use_steps:
        suffix = steps
    elif constrained:
        suffix = "Choose between A and B. Answer only with A or B depending on your choice."
    else:
        suffix = "Make a choice and explain why"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": instructions_script_short},
        {"role": "user",   "content": game_script + "\n\n" + suffix},
    ]
    if model == "llama":
        return convert_messages_to_prompt_llama(messages)
    elif model == "qwen":
        return convert_messages_to_prompt_qwen(messages)
    else:
        return convert_messages_to_prompt_mistral(messages)

# ── Extraction helpers ───────────────────────────────────────────────────────────

def extract_choice(text):
    text  = re.sub(r'[^\w\s]', '', text.strip())
    words = text.split()
    if 'A' in words and 'B' not in words:
        return 'A', True
    elif 'B' in words and 'A' not in words:
        return 'B', True
    return 'Unknown', False

def extract_good_bad(text):
    text  = re.sub(r'[^\w\s]', '', text.lower().strip())
    words = text.split()
    if 'good' in words and 'bad' not in words:
        return 'good'
    elif 'bad' in words and 'good' not in words:
        return 'bad'
    return 'unknown'

# ── LLM loader ───────────────────────────────────────────────────────────────────

def load_llm(model_path, gpu_memory_utilization=0.44):
    return LLM(
        model=model_path,
        dtype="float16",
        trust_remote_code=True,
        download_dir=model_path,
        max_model_len=8192,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=1,
        enforce_eager=False
    )

# ── Logical verifier prompt ───────────────────────────────────────────────────────
# Pool of 10 examples (original 6 verbatim + 4 new). For each verifier call, 5 are
# sampled without replacement so that the cooperation bias from always showing the
# same 2:1 cooperative-to-defective "good" examples is averaged out across calls.
# Good examples: 2 cooperative (ex 2, 4) + 3 defective (ex 6, 7, 9) = balanced pool.
# Bad examples:  3 wrong-reasoning (ex 1, 3, 5) + 2 wrong-choice (ex 8, 10).

VERIFIER_EXAMPLE_POOL = [
    # ── Original example 1 (bad, chose B when A strictly dominates, S=7 T=6) ─────
    {
        "label": "bad",
        "rules": """\
                If you choose A and the other player chooses A. You earn 10 points, the other player earns 10 points.
                If you choose A and the other player chooses B. You earn 7 points, the other player earns 6 points.
                If you choose B and the other player chooses A. You earn 6 points, the other player earns 7 points.
                If you choose B and the other player chooses B. You earn 5 points, the other player earns 5 points.""",
        "answer": """\
                Based on the rules and possible outcomes, here's the analysis:

                - If you choose A, you can earn either 10 points (if the other player chooses A) or 7 points (if the other player chooses B).
                - If you choose B, you can earn either 5 points (if the other player chooses B) or 6 points (if the other player chooses A).

                However, the rules state that the difference in points between you and the other player has no effect on the prize. Therefore, it comes down to the following:

                - If you choose A, the highest possible prize is 10 points, but the lowest possible prize is 7 points.
                - If you choose B, the highest possible prize is 5 points, but the lowest possible prize is 6 points.

                Since the goal is to maximize the possible prize, it would seem that choosing A gives you the potential for a higher prize (100 euros) compared to choosing B (60 euros as the highest potential prize). However, it's important to keep in mind that the other player is making the same analysis as well, so it might be wiser to choose B to reduce the chances of both players choosing A and thus reducing the potential prize for both.

                As this is a one-shot game, it's difficult to say for certain which choice will lead to a higher average prize for both players. It could be a game of bluff and strategy rather than a purely mathematical decision.

                In this analysis, I would lean towards choosing B, to potentially lower the chances of both players receiving a lower prize, but it's ultimately up to your personal decision.""",
    },
    # ── Original example 2 (good, chose A cooperate, S=10 T=6) ──────────────────
    {
        "label": "good",
        "rules": """\
                If you choose A and the other player chooses A. You earn 10 points, the other player earns 10 points.
                If you choose A and the other player chooses B. You earn 10 points, the other player earns 6 points.
                If you choose B and the other player chooses A. You earn 6 points, the other player earns 10 points.
                If you choose B and the other player chooses B. You earn 5 points, the other player earns 5 points.""",
        "answer": """\
                To make a decision, let's analyze the outcomes based on the rules and possible outcomes:

                1. If you choose A:
                   - (A,A): Both players earn 10 points each.
                   - (A,B): You earn 10 points, and the other player earns 6 points.

                2. If you choose B:
                   - (B,A): You earn 6 points, and the other player earns 10 points.
                   - (B,B): Both players earn 5 points each.

                3. Since the monetary prize for each player is the number of points multiplied by 10, it is clear that earning more points results in a higher prize.

                4. Comparing the outcomes in each group:
                   - If you choose A: (A,A) gives you the highest reward, and (A,B) gives you the same reward but a lower-one for the other player.
                   - If you choose B: (B,B) gives you the lowest reward, and (B,A) gives a slightly higher reward, but still lower than all possible rewards when choosing A.

                5. Based on this information, choosing A would result in a higher potential reward, making it the better choice.
                Even though one might worry about the other player choice, it doesn't make sense since the only possible outcome of choosing A is 10 points, which is the maximum.
                The other player choice isn't going to affect your outcome at all, there is no risk.""",
    },
    # ── Original example 3 (bad, fabricated probabilities, S=9 T=15) ─────────────
    {
        "label": "bad",
        "rules": """\
                If you choose A and the other player chooses A. You earn 10 points, the other player earns 10 points.
                If you choose A and the other player chooses B. You earn 9 points, the other player earns 15 points.
                If you choose B and the other player chooses A. You earn 15 points, the other player earns 9 points.
                If you choose B and the other player chooses B. You earn 5 points, the other player earns 5 points.""",
        "answer": """\
                Let's analyze the outcomes:
                - If you choose A, the probability of the other player also choosing A is 50%.
                In this case, you would earn 10 points, and the other player would also earn 10 points (outcome A,A).
                The other possibility is that the other player chooses B, in which case you would earn 9 points, and they would earn 15 points (outcome A,B).
                - If you choose B, the probability of the other player also choosing B is 25%.
                In this case, you would earn 5 points, and the other player would also earn 5 points (outcome B,B).
                The other possibility is that the other player chooses A, in which case you would earn 15 points, and they would earn 9 points (outcome B,A).

                Now, let's compare the outcomes:
                - If you choose A, you have two possible outcomes: one with a higher reward (10 points) and one with a lower reward (9 points).
                - If you choose B, you have two possible outcomes: one with a lower reward (5 points) and one with a higher reward (15 points).

                Since you don't know the other player's choice, it's impossible to determine the exact outcome.
                However, it's interesting to note that if the other player chooses A, choosing A would result in a lower reward (9 points) compared to choosing B (15 points).
                On the other hand, if the other player chooses B, choosing A would result in a higher reward (10 points) compared to choosing B (5 points).
                Given this analysis, it seems that choosing B gives you the potential for a higher reward, but it's also possible that the other player will choose A in which case you would have a lower reward.
                It's truly a gamble. In this game, there is no optimal strategy as the game relies on the other player's choice, which remains unknown.
                Therefore, it's hard to make a definitive choice between A and B as there is no clear advantage in either option.
                Based on the analysis and the lack of a perfect answer, I'll leave the choice up to you.
                You can choose either A or B as there's no guaranteed outcome. Good luck!""",
    },
    # ── Original example 4 (good, chose B cooperate in script_1, S=8 T=5) ────────
    {
        "label": "good",
        "rules": """\
                If you choose B and the other player chooses B. You earn 10 points, the other player earns 10 points.
                If you choose B and the other player chooses A. You earn 8 points, the other player earns 5 points.
                If you choose A and the other player chooses B. You earn 5 points, the other player earns 8 points.
                If you choose A and the other player chooses A. You earn 5 points, the other player earns 5 points.""",
        "answer": """\
                Based on the given information, I will analyze the outcomes for each choice:

                If you choose A:
                - If the other player chooses A, you both will receive 5 points each.
                - If the other player chooses B, you will receive 5 points, while the other player receives 8 points.

                If you choose B:
                - If the other player chooses A, you will receive 8 points, while the other player receives 5 points.
                - If the other player chooses B, you both will receive 10 points each.

                If you choose B, the maximum amount of points you can get is 10 points and the minimum, 8 points.
                If you choose A the maximum amount of points you can get is 5 points and the minimum, 5 points.
                The maximum points is greater when you choose B, leading to a higher prize.
                The minimum points is greater when you choose B, leading to a lower risk of getting little points.
                Then, choice B is optimal since it is the option with the highest reward and the lower risk.""",
    },
    # ── Original example 5 (bad, incomplete reasoning in script_1, S=10 T=7) ─────
    {
        "label": "bad",
        "rules": """\
                If you choose B and the other player chooses B. You earn 10 points, the other player earns 10 points.
                If you choose B and the other player chooses A. You earn 10 points, the other player earns 7 points.
                If you choose A and the other player chooses B. You earn 6 points, the other player earns 10 points.
                If you choose A and the other player chooses A. You earn 5 points, the other player earns 5 points.""",
        "answer": """\
                Given the rules, choosing A would bring a maximum of 5 points, and choosing B would bring a maximum of 10 points if the other player also chooses B.
                Since we don't know the other player's choice, it's impossible to guarantee a higher score by choosing A or B.
                However, if we're forced to make a choice, choosing B seems like the better option since it has a higher potential reward if the other player also chooses B.
                But, ultimately, this game is a gamble, and both choices have an element of risk.""",
    },
    # ── Original example 6 (good, chose B defect, S=1 T=15) ─────────────────────
    {
        "label": "good",
        "rules": """\
                If you choose A and the other player chooses A. You earn 10 points, the other player earns 10 points.
                If you choose A and the other player chooses B. You earn 1 points, the other player earns 15 points.
                If you choose B and the other player chooses A. You earn 15 points, the other player earns 1 points.
                If you choose B and the other player chooses B. You earn 5 points, the other player earns 5 points.""",
        "answer": """\
                Choosing A can lead to the worst result (earning only 1 point) and choosing B can lead to the best result (earning 15 points).
                Then I choose B to earn at least 5 points and also have the potential for the highest reward.""",
    },
    # ── New example 7 (good, chose B defect, S=0 T=12) ──────────────────────────
    {
        "label": "good",
        "rules": """\
                If you choose A and the other player chooses A. You earn 10 points, the other player earns 10 points.
                If you choose A and the other player chooses B. You earn 0 points, the other player earns 12 points.
                If you choose B and the other player chooses A. You earn 12 points, the other player earns 0 points.
                If you choose B and the other player chooses B. You earn 5 points, the other player earns 5 points.""",
        "answer": """\
                Based on the rules, let me analyze each possible choice:

                1. If I choose A:
                   - (A,A): I earn 10 points.
                   - (A,B): I earn 0 points.

                2. If I choose B:
                   - (B,A): I earn 12 points.
                   - (B,B): I earn 5 points.

                Comparing the outcomes: when the other player chooses A, B yields 12 points versus 10 for A. When the other player chooses B, B yields 5 points versus 0 for A. In both scenarios B gives a higher payoff than A. B is therefore the dominant strategy regardless of what the other player does, and I choose B.""",
    },
    # ── New example 8 (bad, chose A cooperate when B strictly dominates, S=2 T=12)
    {
        "label": "bad",
        "rules": """\
                If you choose A and the other player chooses A. You earn 10 points, the other player earns 10 points.
                If you choose A and the other player chooses B. You earn 2 points, the other player earns 12 points.
                If you choose B and the other player chooses A. You earn 12 points, the other player earns 2 points.
                If you choose B and the other player chooses B. You earn 5 points, the other player earns 5 points.""",
        "answer": """\
                Both players want to maximize their points. If both choose A, we both earn 10 points, which is fair and gives the best combined outcome of 20 points total. Choosing B is risky: if both players choose B we only earn 5 each, which is worse for everyone. Since we are rational players and cooperation leads to the best mutual outcome, I choose A.""",
    },
    # ── New example 9 (good, chose A defect in script_1, S=4 T=13) ──────────────
    {
        "label": "good",
        "rules": """\
                If you choose B and the other player chooses B. You earn 10 points, the other player earns 10 points.
                If you choose B and the other player chooses A. You earn 4 points, the other player earns 13 points.
                If you choose A and the other player chooses B. You earn 13 points, the other player earns 4 points.
                If you choose A and the other player chooses A. You earn 5 points, the other player earns 5 points.""",
        "answer": """\
                Let me compare the two possible choices:

                1. If I choose A:
                   - (A,B): I earn 13 points.
                   - (A,A): I earn 5 points.

                2. If I choose B:
                   - (B,B): I earn 10 points.
                   - (B,A): I earn 4 points.

                Comparing outcome by outcome: if the other player chooses B, A gives me 13 while B gives me 10. If the other player chooses A, A gives me 5 while B gives me 4. In both cases A yields a strictly higher reward. A dominates B regardless of the other player's choice, so I choose A.""",
    },
    # ── New example 10 (bad, chose B cooperate when A strictly dominates, S=3 T=11)
    {
        "label": "bad",
        "rules": """\
                If you choose B and the other player chooses B. You earn 10 points, the other player earns 10 points.
                If you choose B and the other player chooses A. You earn 3 points, the other player earns 11 points.
                If you choose A and the other player chooses B. You earn 11 points, the other player earns 3 points.
                If you choose A and the other player chooses A. You earn 5 points, the other player earns 5 points.""",
        "answer": """\
                This game presents a trade-off between individual gain and mutual benefit. If both players choose B, we each earn 10 points - a fair and cooperative outcome. If I choose A and the other player chooses B, I gain 11 but the other player earns only 3, which is selfish. Since we are both rational players and the rules are symmetric, the other player is likely reasoning the same way. We should both prefer the cooperative equilibrium where we each earn 10 points. I choose B.""",
    },
]

for _ex in VERIFIER_EXAMPLE_POOL:
    _ex["rules"]  = textwrap.dedent(_ex["rules"])
    _ex["answer"] = textwrap.dedent(_ex["answer"])

_VERIFIER_CRITERIA = """\
                You have to read the answer of Player 1 and output good or bad according to the following criteria:

                Criteria for good:
                - ALL arithmetic/mathematical comparisons and claims are CORRECT.
                - ALL descriptions of the possible outcomes are CORRECT.
                - ONLY TWO possible outputs per choice, for A: (A,A) and (A,B), for B: (B,A) and (B,B).
                - The final choice is CONSISTENT with the previous reasoning.
                - ALL statements make sense.
                - GOOD UNDERSTANDING of the rules.

                Criteria for bad:
                - ONE OR MORE arithmetic/mathematical comparisons or claims are WRONG.
                - ONE OR MORE descriptions of the possible outcomes are WRONG.
                - Does not understand that there are ONLY TWO possible outcomes per choice.
                - Final choice is NOT CONSISTENT with the previous reasoning.
                - ONE OR MORE statements do not make sense.
                - BAD UNDERSTANDING of the rules.

                Rules of the game:"""
_VERIFIER_CRITERIA = textwrap.dedent(_VERIFIER_CRITERIA)

def build_verifier_message(points, answer1):
    indices = np.random.choice(len(VERIFIER_EXAMPLE_POOL), size=5, replace=False)
    sampled = [VERIFIER_EXAMPLE_POOL[i] for i in indices]

    examples_text = ""
    for i, ex in enumerate(sampled, 1):
        examples_text += f"""
Example {i}, {ex['label']} reasoning:

Rules of Example {i}:

{ex['rules']}

Answer of Example {i}:

{ex['answer']}

Output of Example {i}: {ex['label']}
"""

    return (_VERIFIER_CRITERIA + "\n" + instructions_script_short + examples_text + f"""
Rules of the game to analyze:
{points}

Answer of Player 1, to analyze:
{answer1}

Important:
- Output ONLY one word: good or bad
- Do not add punctuation, extra spaces, or explanations.
""")


# ── Matrix index maps ────────────────────────────────────────────────────────────
# Reverses S axis so row 0 = max S, matching the original script's [20-s, t] indexing.

s_to_idx = {s: N_S - 1 - i for i, s in enumerate(S_VALUES)}
t_to_idx = {t: i for i, t in enumerate(T_VALUES)}


# ── Stage runners ────────────────────────────────────────────────────────────────

def run_simple(args, game_order, all_values_script_0, all_values_script_1,
               N_GAMES, params_explain):
    """
    Main model only. No Qwen. Constrained suffix forces single-letter answer.
    Retries failed games each repetition until all 121 succeed or MAX_ITER is reached.
    Returns (raw_results, maxiter_count) where maxiter_count[s,t] counts reps in which
    that cell was left unresolved when the retry limit was hit.
    """
    repetitions   = args.repetitions
    MAX_ITER      = 50
    raw_results   = np.zeros((repetitions, N_S, N_T))
    maxiter_count = np.zeros((N_S, N_T))

    llm = load_llm(MODEL_PATHS[args.model])

    for rep in range(repetitions):
        print(f"\nRepetition {rep+1}/{repetitions}")
        game_matrix   = np.zeros([N_S, N_T])
        games_to_play = list(range(N_GAMES))
        random_list   = [None] * N_GAMES
        iter_loop     = 0

        while games_to_play:
            print(f"  games left: {len(games_to_play)} | iter: {iter_loop}")
            prompts_main = []
            for game in games_to_play:
                if random_list[game] is None:
                    random_list[game] = 1 if np.random.rand() <= 0.5 else 2
                script = all_values_script_0[game] if random_list[game] == 1 else all_values_script_1[game]
                prompts_main.append(build_main_prompt(script, args.model,
                                                      use_steps=False, constrained=True))

            outputs      = llm.generate(prompts_main, params_explain)
            long_answers = [o.outputs[0].text for o in outputs]

            games_to_play_copy = games_to_play.copy()
            for ans_index, game in enumerate(games_to_play):
                choice, state = extract_choice(long_answers[ans_index])
                if state:
                    games_to_play_copy.remove(game)
                    s_idx = s_to_idx[game_order[game][0]]
                    t_idx = t_to_idx[game_order[game][1]]
                    if random_list[game] == 1:
                        if choice == 'A':
                            game_matrix[s_idx, t_idx] += 1
                    else:
                        if choice == 'B':
                            game_matrix[s_idx, t_idx] += 1
            games_to_play = games_to_play_copy

            iter_loop += 1
            if iter_loop >= MAX_ITER and games_to_play:
                print(f"  Max iterations ({MAX_ITER}) reached. {len(games_to_play)} games unresolved.")
                print(f"  Stuck games: {games_to_play}")
                for game in games_to_play:
                    s_idx = s_to_idx[game_order[game][0]]
                    t_idx = t_to_idx[game_order[game][1]]
                    maxiter_count[s_idx, t_idx] += 1
                break

        raw_results[rep] = game_matrix
        print(f"  done. Mean cooperation: {game_matrix.mean():.3f}")

    del llm
    gc.collect()

    return raw_results, maxiter_count


def run_extract_or_multi(args, game_order, all_values_script_0, all_values_script_1,
                         N_GAMES, params_explain, params_extract, use_steps):
    """
    Main model + Qwen extractor, with retry loop matching the original paper scripts.
    use_steps=False → extract stage; use_steps=True → multi stage.
    Games that fail extraction are retried until successful or MAX_ITER is reached.
    Returns (raw_results, maxiter_count) where maxiter_count[s,t] counts reps in which
    that cell was left unresolved when the retry limit was hit.
    """
    repetitions   = args.repetitions
    MAX_ITER      = 50
    raw_results   = np.zeros((repetitions, N_S, N_T))
    maxiter_count = np.zeros((N_S, N_T))

    llm_long = load_llm(MODEL_PATHS[args.model])
    if MODEL_PATHS[args.model] == EXTRACTOR_PATH:
        llm_short = llm_long
    else:
        time.sleep(10)
        llm_short = load_llm(EXTRACTOR_PATH)

    for rep in range(repetitions):
        print(f"\nRepetition {rep+1}/{repetitions}")

        game_matrix   = np.zeros([N_S, N_T])
        condition     = False
        games_to_play = list(range(N_GAMES))
        random_list   = [None] * N_GAMES
        iter_loop     = 0

        while not condition:
            print(f"  rep: {rep} | games left: {len(games_to_play)} | iter: {iter_loop}")

            # Stage 1: main model (batched over remaining games)
            prompts_main = []
            for game in games_to_play:
                if random_list[game] is None:
                    random_list[game] = 1 if np.random.rand() <= 0.5 else 2
                script = all_values_script_0[game] if random_list[game] == 1 else all_values_script_1[game]
                prompts_main.append(build_main_prompt(script, args.model, use_steps=use_steps))

            outputs_main = llm_long.generate(prompts_main, params_explain)
            long_answers = [o.outputs[0].text for o in outputs_main]

            # Stage 2: Qwen extractor (batched over remaining games)
            extract_prompts = []
            for ans_index in range(len(games_to_play)):
                prompt2 = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user",
                     "content": "The player who was asked to choose between A and B answered: " + long_answers[ans_index]},
                    {"role": "user",
                     "content": "What did the person who wrote the message choose? Answer shortly."}
                ]
                extract_prompts.append(convert_messages_to_prompt_qwen(prompt2))

            extract_outputs = llm_short.generate(extract_prompts, params_extract)

            games_to_play_copy = games_to_play.copy()
            for ans_index in range(len(games_to_play)):
                game = games_to_play[ans_index]
                choice, state = extract_choice(extract_outputs[ans_index].outputs[0].text)
                if state:
                    games_to_play_copy.remove(game)
                    s_idx = s_to_idx[game_order[game][0]]
                    t_idx = t_to_idx[game_order[game][1]]
                    if random_list[game] == 1:
                        if choice == 'A':
                            game_matrix[s_idx, t_idx] += 1
                    else:
                        if choice == 'B':
                            game_matrix[s_idx, t_idx] += 1

            games_to_play = games_to_play_copy

            if not games_to_play:
                condition = True
            else:
                iter_loop += 1

            if iter_loop >= MAX_ITER:
                print(f"  Max iterations ({MAX_ITER}) reached. {len(games_to_play)} games unresolved.")
                print(f"  Stuck games: {games_to_play}")
                for game in games_to_play:
                    s_idx = s_to_idx[game_order[game][0]]
                    t_idx = t_to_idx[game_order[game][1]]
                    maxiter_count[s_idx, t_idx] += 1
                condition = True

        raw_results[rep] = game_matrix
        print(f"  Repetition {rep+1} done. Mean cooperation: {game_matrix.mean():.3f}")

    if llm_short is not llm_long:
        del llm_short
    del llm_long
    gc.collect()

    return raw_results, maxiter_count


def run_final(args, game_order, all_values_script_0, all_values_script_1,
              N_GAMES, params_explain, params_extract):
    """
    Full pipeline: steps + Qwen verifier + Qwen extractor, with retry loop.
    Verifier is bypassed per-game after BYPASS_THRESHOLD stuck iterations rather
    than globally, matching the spirit of the original sticking_matrix logic but
    without contaminating quality filtering for other games in the same repetition.
    Returns (raw_results, bypass_count) where bypass_count[s,t] is the number of
    repetitions in which that cell was resolved without verifier quality check.
    """
    repetitions     = args.repetitions
    MAX_ITER        = 50
    BYPASS_THRESHOLD = 3
    raw_results  = np.zeros((repetitions, N_S, N_T))
    bypass_count = np.zeros((N_S, N_T))

    llm_long = load_llm(MODEL_PATHS[args.model])
    if MODEL_PATHS[args.model] == EXTRACTOR_PATH:
        llm_short = llm_long
    else:
        time.sleep(10)
        llm_short = load_llm(EXTRACTOR_PATH)

    for rep in range(repetitions):
        print(f"\nRepetition {rep+1}/{repetitions}")

        game_matrix      = np.zeros([N_S, N_T])
        condition        = False
        games_to_play    = list(range(N_GAMES))
        random_list      = [None] * N_GAMES
        game_stuck_iters = {g: 0 for g in range(N_GAMES)}
        game_bypass      = set()
        iter_loop        = 0

        while not condition:
            n_bypassed = len(game_bypass & set(games_to_play))
            print(f"  rep: {rep} | games left: {len(games_to_play)} | iter: {iter_loop} | bypassed: {n_bypassed}")

            # Stage 1: main model (batched)
            prompts_main = []
            for game in games_to_play:
                if random_list[game] is None:
                    random_list[game] = 1 if np.random.rand() <= 0.5 else 2
                script = all_values_script_0[game] if random_list[game] == 1 else all_values_script_1[game]
                prompts_main.append(build_main_prompt(script, args.model, use_steps=True))

            outputs_main = llm_long.generate(prompts_main, params_explain)
            long_answers = [o.outputs[0].text for o in outputs_main]

            games_to_play_copy = games_to_play.copy()

            # Stage 2a: verifier (skipped per-game for those that have been stuck too long)
            needs_verifier = [i for i, g in enumerate(games_to_play) if g not in game_bypass]
            qualities      = ['good'] * len(games_to_play)  # bypassed games default to 'good'

            if needs_verifier:
                verifier_prompts = []
                for ans_index in needs_verifier:
                    game   = games_to_play[ans_index]
                    points = (all_values_script_0[game] if random_list[game] == 1
                              else all_values_script_1[game])
                    prompt3 = [
                        {"role": "system",
                         "content": "You are a logical verifier. Your purpose is to look for inconsistencies and errors in a given text."},
                        {"role": "user", "content": build_verifier_message(points, long_answers[ans_index])}
                    ]
                    verifier_prompts.append(convert_messages_to_prompt_qwen(prompt3))

                verifier_outputs = llm_short.generate(verifier_prompts, params_extract)
                for batch_pos, ans_index in enumerate(needs_verifier):
                    qualities[ans_index] = extract_good_bad(verifier_outputs[batch_pos].outputs[0].text)

            # Stage 2b: extractor (batched, only for games that passed verifier)
            extract_indices = [i for i, q in enumerate(qualities) if q == 'good']

            if extract_indices:
                extract_prompts = []
                for ans_index in extract_indices:
                    answer1 = long_answers[ans_index]
                    prompt2 = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user",
                         "content": "The player who was asked to choose between A and B answered: " + answer1},
                        {"role": "user",
                         "content": "What did the person who wrote the message choose? Answer shortly."}
                    ]
                    extract_prompts.append(convert_messages_to_prompt_qwen(prompt2))

                extract_outputs = llm_short.generate(extract_prompts, params_extract)

                for batch_pos, ans_index in enumerate(extract_indices):
                    game  = games_to_play[ans_index]
                    s     = game_order[game][0]
                    t     = game_order[game][1]
                    choice, state = extract_choice(extract_outputs[batch_pos].outputs[0].text)

                    if state:
                        games_to_play_copy.remove(game)
                        s_idx = s_to_idx[s]
                        t_idx = t_to_idx[t]
                        if game in game_bypass:
                            bypass_count[s_idx, t_idx] += 1
                        if random_list[game] == 1:
                            if choice == 'A':
                                game_matrix[s_idx, t_idx] += 1
                        else:
                            if choice == 'B':
                                game_matrix[s_idx, t_idx] += 1

            games_to_play = games_to_play_copy

            # Increment stuck counter for every game not resolved this iteration;
            # promote to per-game bypass once threshold is reached.
            for game in games_to_play:
                game_stuck_iters[game] += 1
                if game_stuck_iters[game] >= BYPASS_THRESHOLD and game not in game_bypass:
                    game_bypass.add(game)
                    s, t = game_order[game]
                    print(f"  Bypassing verifier for game (S={s}, T={t}) after {BYPASS_THRESHOLD} stuck iters")

            if not games_to_play:
                condition = True
            else:
                iter_loop += 1

            if iter_loop >= MAX_ITER:
                print(f"  Max iterations ({MAX_ITER}) reached. {len(games_to_play)} games unresolved.")
                print(f"  Stuck games: {games_to_play}")
                condition = True

        raw_results[rep] = game_matrix
        print(f"  Repetition {rep+1} done. Mean cooperation: {game_matrix.mean():.3f} | games bypassed: {len(game_bypass)}")

    if llm_short is not llm_long:
        del llm_short
    del llm_long
    gc.collect()

    return raw_results, bypass_count


# ── Main ─────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    args = parse_arguments()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Model: {args.model} | Stage: {args.stage} | Repetitions: {args.repetitions}")

    params_explain = SamplingParams(temperature=0.8, max_tokens=1000)
    params_extract = SamplingParams(temperature=0.3, max_tokens=50)

    _game_order = game_order
    _script_0   = all_values_script_0
    _script_1   = all_values_script_1
    _script_0_q = all_values_script_0_q
    _script_1_q = all_values_script_1_q
    _n_games    = N_GAMES

    if args.debug:
        _game_order = game_order[:5]
        _script_0   = all_values_script_0[:5]
        _script_1   = all_values_script_1[:5]
        _script_0_q = all_values_script_0_q[:5]
        _script_1_q = all_values_script_1_q[:5]
        _n_games    = 5
        args.repetitions = 1

    if args.stage == "simple":
        raw_results, maxiter_count = run_simple(args, _game_order, _script_0_q, _script_1_q,
                                                _n_games, params_explain)

    elif args.stage == "extract":
        raw_results, maxiter_count = run_extract_or_multi(args, _game_order, _script_0, _script_1,
                                                          _n_games, params_explain, params_extract,
                                                          use_steps=False)

    elif args.stage == "multi":
        raw_results, maxiter_count = run_extract_or_multi(args, _game_order, _script_0, _script_1,
                                                          _n_games, params_explain, params_extract,
                                                          use_steps=True)

    elif args.stage == "final":
        raw_results, bypass_count = run_final(args, _game_order, _script_0, _script_1,
                                              _n_games, params_explain, params_extract)

    stem      = f"{args.model}_{args.stage}"
    raw_path  = os.path.join(args.output_dir, f"{stem}_raw.npy")
    mean_path = os.path.join(args.output_dir, f"{stem}_mean.txt")

    np.save(raw_path, raw_results)
    print(f"\nSaved raw results to {raw_path}  shape: {raw_results.shape}")

    mean_matrix = raw_results.mean(axis=0)
    np.savetxt(mean_path, mean_matrix, fmt="%.4f")
    print(f"Saved mean matrix to {mean_path}")
    print(f"Overall mean cooperation: {mean_matrix.mean():.3f}")

    if args.stage == "final":
        bypass_path = os.path.join(args.output_dir, f"{stem}_bypass.txt")
        np.savetxt(bypass_path, bypass_count, fmt="%.4f")
        print(f"Saved bypass count matrix to {bypass_path}")
        print(f"Mean bypass rate: {bypass_count.mean():.3f} | Max: {bypass_count.max():.0f} reps")
    else:
        maxiter_path = os.path.join(args.output_dir, f"{stem}_maxiter.txt")
        np.savetxt(maxiter_path, maxiter_count, fmt="%.4f")
        print(f"Saved maxiter count matrix to {maxiter_path}")
        n_affected = int((maxiter_count > 0).sum())
        print(f"Games affected by maxiter: {n_affected}/121 | Max: {maxiter_count.max():.0f} reps")
