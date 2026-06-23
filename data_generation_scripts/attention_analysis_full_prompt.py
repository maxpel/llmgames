"""
Attention-based payoff salience analysis for Palatsi et al. (arXiv:2511.04500)

Uses the full prompt from the paper (instructions + payoff values + multi-step
reasoning scaffold), with chat templates identical to the original simulation
scripts (Llama_Mistral_finalversion.py, Qwen_finalversion.py).

Key prompt notes:
  Llama: system + two user turns, Llama chat template
  Qwen:  system + two user turns, Qwen im_start template
  Mistral: convert_messages_to_prompt_mistral silently drops the system role,
           producing two bare [INST]...[/INST] blocks -- matches paper exactly.

Runs both prompt orderings (A=cooperate, B=cooperate) and averages the
resulting attention-cooperation correlations, matching the randomization
in the original simulation.

Payoff token positions found via character-level offset mapping, validated
across edge cases before the full run.

References:
  Jain & Wallace (2019) -- attention is not explanation
  Wiegreffe & Pinter (2019) -- attention is not not explanation
"""

import sys
import site
site.ENABLE_USER_SITE = False
sys.path = [p for p in sys.path if '.local' not in p]

import os
import re
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# ── Model paths ────────────────────────────────────────────────────────────────
MODELS = {
    "llama":   "meta-llama/Llama-3.1-8B-Instruct",
    "qwen":    "Qwen/Qwen2.5-7B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
}

# ── Constants ──────────────────────────────────────────────────────────────────
R, P = 10, 5
S_VALUES = list(range(0, 11))    # 0..10
T_VALUES = list(range(5, 16))    # 5..15
GAMES    = [(s, t) for s in S_VALUES for t in T_VALUES]  # 121 games, row-major

COLORS = {"R": "#2ca02c", "S": "#1f77b4", "T": "#d62728", "P": "#9467bd"}


# ── Prompt components (verbatim from paper scripts) ───────────────────────────

INSTRUCTIONS = """\
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

STEPS = """\
1. There are four possible outcomes in this game, depending on your choice (A or B) and the other player's choice (A or B).
2. Group the outcomes based on your decision:
   - If you choose A: (A,A) and (A,B)
   - If you choose B: (B,A) and (B,B)
3. Analyze and compare the outcomes in each group step by step.
4. Make sure your analysis is consistent with the rules of the game and the possible outcomes.
5. Based on your analysis, choose either A or B."""


def build_values_script(s, t, cooperation_is_A=True):
    """
    cooperation_is_A=True  -> all_values_script_0 (A=cooperate)
    cooperation_is_A=False -> all_values_script_1 (B=cooperate)

    Player's own payoffs always appear as '. You earn N' in R,S,T,P order
    regardless of ordering, so find_payoff_positions needs no changes.
    """
    if cooperation_is_A:
        return f"""\
If you choose A and the other player chooses A. You earn 10 points, the other player earns 10 points.
If you choose A and the other player chooses B. You earn {s} points, the other player earns {t} points.
If you choose B and the other player chooses A. You earn {t} points, the other player earns {s} points.
If you choose B and the other player chooses B. You earn 5 points, the other player earns 5 points."""
    else:
        return f"""\
If you choose B and the other player chooses B. You earn 10 points, the other player earns 10 points.
If you choose B and the other player chooses A. You earn {s} points, the other player earns {t} points.
If you choose A and the other player chooses B. You earn {t} points, the other player earns {s} points.
If you choose A and the other player chooses A. You earn 5 points, the other player earns 5 points."""


# ── Chat templates (verbatim from paper scripts) ───────────────────────────────

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
    """
    Mistral template from paper. System role is silently dropped -- only
    user and assistant turns are handled. Two user messages therefore produce
    two separate [INST]...[/INST] blocks with no system content.
    """
    prompt = ""
    for msg in messages:
        if msg["role"] == "user":
            prompt += "<s>[INST] " + msg["content"].strip() + " [/INST]"
        elif msg["role"] == "assistant":
            prompt += " " + msg["content"].strip() + " </s>"
    return prompt


def build_full_prompt(model_key, s, t, cooperation_is_A=True):
    """
    Build full prompt identical to the paper for each model.

    All three models receive the same three-message structure:
      system: "You are a helpful assistant."
      user:   INSTRUCTIONS
      user:   values_script + STEPS

    For Mistral the system message is silently dropped by the converter,
    matching the paper exactly -- no special handling needed here.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": INSTRUCTIONS},
        {"role": "user",   "content": build_values_script(s, t, cooperation_is_A) + "\n\n" + STEPS},
    ]
    if model_key == "llama":
        return convert_messages_to_prompt_llama(messages)
    elif model_key == "qwen":
        return convert_messages_to_prompt_qwen(messages)
    else:
        return convert_messages_to_prompt_mistral(messages)


# ── Payoff token position finding and validation ───────────────────────────────

def find_payoff_positions(tokenizer, prompt):
    """
    Locate token indices of each payoff value (R, S, T, P) using
    character-level offset mapping.

    Finds the four '. You earn N' occurrences in the values script in order
    R, S, T, P. This order holds for both prompt orderings:
      A=cooperate: (A,A)=R, (A,B)=S, (B,A)=T, (B,B)=P
      B=cooperate: (B,B)=R, (B,A)=S, (A,B)=T, (A,A)=P

    For multi-token values (e.g. Qwen splitting '10' -> '1','0'), all
    covering token indices are returned and attention is averaged downstream.

    Returns dict: {payoff_name -> list of token indices}
    """
    matches = list(re.finditer(r'\. You earn (\d+)', prompt))
    assert len(matches) == 4, (
        f"Expected 4 '. You earn N' matches, found {len(matches)}."
    )

    payoff_names = ["R", "S", "T", "P"]
    char_spans = {
        name: (m.start(1), m.end(1))
        for name, m in zip(payoff_names, matches)
    }

    encoding = tokenizer(
        prompt,
        return_offsets_mapping=True,
        return_tensors="pt",
        add_special_tokens=False,
    )
    offsets = encoding["offset_mapping"][0].tolist()

    positions = {}
    for name, (char_start, char_end) in char_spans.items():
        token_indices = [
            i for i, (tok_start, tok_end) in enumerate(offsets)
            if tok_start < char_end and tok_end > char_start and tok_start < tok_end
        ]
        assert len(token_indices) > 0, (
            f"No tokens found for payoff {name} at chars {char_start}:{char_end}"
        )
        positions[name] = token_indices

    return positions


def validate_payoff_positions(tokenizer, model_key, s, t,
                              cooperation_is_A=True, verbose=True):
    """
    Verify that find_payoff_positions correctly identifies each payoff value.
    Decodes detected tokens and checks against expected numeric values.
    Prints surrounding context for visual confirmation.

    Returns True if all payoffs match, False otherwise.
    """
    prompt   = build_full_prompt(model_key, s, t, cooperation_is_A)
    expected = {"R": str(R), "S": str(s), "T": str(t), "P": str(P)}
    positions = find_payoff_positions(tokenizer, prompt)

    encoding    = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    all_ids     = encoding["input_ids"][0]
    tok_strings = [tokenizer.decode([tid]) for tid in all_ids]

    if verbose:
        ordering = "A=coop" if cooperation_is_A else "B=coop"
        print(f"\n  [{model_key} | S={s} T={t} | {ordering}]")

    all_ok = True
    for name, idxs in positions.items():
        decoded      = "".join(tokenizer.decode([all_ids[i]]) for i in idxs).strip()
        expected_val = expected[name]
        ok           = decoded == expected_val

        if verbose:
            ctx_start = max(0, idxs[0] - 2)
            ctx_end   = min(len(tok_strings), idxs[-1] + 3)
            context   = tok_strings[ctx_start:ctx_end]
            markers   = ["^^" if ctx_start + j in idxs else "  "
                         for j in range(len(context))]
            print(f"    {name}: {idxs} -> '{decoded}' "
                  f"(expected '{expected_val}') {'OK' if ok else '*** MISMATCH ***'}")
            print(f"       context: {list(zip(markers, context))}")

        if not ok:
            all_ok = False

    return all_ok


def run_full_validation(model_key):
    """
    Validate payoff position detection across edge cases before the full run.
    Tests single-digit and double-digit values, both prompt orderings.
    Raises AssertionError on first failure.
    """
    print(f"\n{'='*60}")
    print(f"Validating: {model_key}")
    print(f"{'='*60}")
    tokenizer = AutoTokenizer.from_pretrained(MODELS[model_key])

    test_cases = [
        (0,  5,  True),    # both single digit, A=coop
        (0,  15, True),    # S single, T double, A=coop
        (10, 5,  True),    # S double, T single, A=coop
        (10, 15, True),    # both double digit, A=coop
        (0,  15, False),   # extremes, B=coop
        (10, 15, False),   # both double, B=coop
        (5,  10, True),    # mid-range
    ]

    for s, t, coop_A in test_cases:
        ok = validate_payoff_positions(tokenizer, model_key, s, t, coop_A)
        assert ok, (
            f"Validation FAILED for {model_key} S={s} T={t} "
            f"cooperation_is_A={coop_A}"
        )

    print(f"\n  All {len(test_cases)} cases passed for {model_key}.")


# ── Attention extraction ───────────────────────────────────────────────────────

def get_attention_to_payoffs(model, tokenizer, model_key, s, t,
                             cooperation_is_A=True):
    """
    For one game and one prompt ordering, compute mean attention (averaged
    over heads) from the last prompt token to each payoff token, per layer.
    For multi-token values, attention is averaged across covering tokens.

    Returns dict: {payoff_name -> np.array of shape (n_layers,)}
    """
    prompt   = build_full_prompt(model_key, s, t, cooperation_is_A)
    inputs   = tokenizer(prompt, return_tensors="pt",
                         add_special_tokens=False).to(next(model.parameters()).device)
    last_idx = inputs["input_ids"].shape[1] - 1

    payoff_positions = find_payoff_positions(tokenizer, prompt)

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # outputs.attentions: tuple of n_layers tensors
    # each shape: (batch=1, n_heads, seq_len, seq_len)
    # layer_attn[0]: (n_heads, seq_len, seq_len)
    # .mean(dim=0): average over heads -> (seq_len, seq_len)
    result = {name: [] for name in payoff_positions}
    for layer_attn in outputs.attentions:
        mean_attn = layer_attn[0].mean(dim=0).cpu().float()
        for name, positions in payoff_positions.items():
            attn_val = np.mean([mean_attn[last_idx, pos].item()
                                for pos in positions])
            result[name].append(attn_val)

    return {name: np.array(vals) for name, vals in result.items()}


def collect_attention_all_games(model_key):
    """
    Run all 121 games for both prompt orderings and return the average,
    matching the randomization procedure of the original simulation.

    Each ordering is cached separately. The average is also cached.
    Returns dict: {payoff_name -> np.array of shape (121, n_layers)}
    """
    save_path_avg = f"attn_full_prompt_{model_key}_avg.npy"
    if os.path.exists(save_path_avg):
        print(f"  Found cached averaged attention at {save_path_avg}, loading...")
        return dict(np.load(save_path_avg, allow_pickle=True).item())

    payoff_names       = ["R", "S", "T", "P"]
    results_by_ordering = []

    for cooperation_is_A in [True, False]:
        ordering_label = "A_coop" if cooperation_is_A else "B_coop"
        save_path      = f"attn_full_prompt_{model_key}_{ordering_label}.npy"

        if os.path.exists(save_path):
            print(f"  Found cached {ordering_label} attention, loading...")
            results_by_ordering.append(
                dict(np.load(save_path, allow_pickle=True).item())
            )
            continue

        print(f"\nLoading {model_key} for {ordering_label}...")
        tokenizer = AutoTokenizer.from_pretrained(MODELS[model_key])
        model     = AutoModelForCausalLM.from_pretrained(
            MODELS[model_key],
            torch_dtype=torch.float16,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        ).to("cuda")
        model.eval()
        print(f"  Model loaded on: {next(model.parameters()).device}")

        all_attn = {name: [] for name in payoff_names}
        for idx, (s, t) in enumerate(GAMES):
            if idx % 20 == 0:
                print(f"  Game {idx+1}/121  (S={s}, T={t})")
            attn = get_attention_to_payoffs(
                model, tokenizer, model_key, s, t, cooperation_is_A
            )
            for name in payoff_names:
                all_attn[name].append(attn[name])

        result = {name: np.stack(vals) for name, vals in all_attn.items()}
        np.save(save_path, result)
        print(f"  Saved to {save_path}")
        results_by_ordering.append(result)

        del model
        torch.cuda.empty_cache()

    avg = {
        name: (results_by_ordering[0][name] + results_by_ordering[1][name]) / 2.0
        for name in payoff_names
    }
    np.save(save_path_avg, avg)
    print(f"  Saved averaged attention to {save_path_avg}")

    return avg


# ── Correlation with cooperation rates ────────────────────────────────────────

def correlate_attention_with_cooperation(attn_data, cooperation_rates):
    """
    For each payoff and each layer, compute Pearson r between attention
    weight and cooperation rate across 121 games.

    cooperation_rates: np.array of shape (121,), flattened row-major
    (S outer, T inner) matching GAMES order.
    """
    n_layers     = list(attn_data.values())[0].shape[1]
    correlations = {name: np.zeros(n_layers) for name in attn_data}

    for name, attn_matrix in attn_data.items():
        for layer in range(n_layers):
            r, _ = pearsonr(attn_matrix[:, layer], cooperation_rates)
            correlations[name][layer] = r

    return correlations


def load_cooperation_matrix(path):
    m = np.loadtxt(path)
    assert m.shape == (len(S_VALUES), len(T_VALUES)), (
        f"Expected ({len(S_VALUES)}, {len(T_VALUES)}), got {m.shape}."
    )
    # Saved matrices have S reversed (row 0 = S=10, row 10 = S=0) because the
    # simulation scripts use s_to_idx = N_S-1-i. Flip so row 0 = S=0, matching
    # the GAMES ordering [(s,t) for s in range(0,11) for t in range(5,16)].
    return np.flipud(m).flatten()    # (121,) S=0 first, T inner


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_attention_results(correlations_by_model,
                           save_path="attention_results_full_prompt.pdf"):
    """
    One panel per model.
    correlations_by_model: {model_label -> {payoff_name -> np.array (n_layers,)}}
    """
    n_models = len(correlations_by_model)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 5), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, (model_label, corr) in zip(axes, correlations_by_model.items()):
        layers = np.arange(len(list(corr.values())[0]))
        for name, vals in corr.items():
            ax.plot(layers, vals, label=f"Attention to {name}",
                    color=COLORS[name], linewidth=1.8)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(model_label, fontsize=13)
        ax.set_xlabel("Layer", fontsize=11)
        ax.set_ylabel("Pearson r (attention vs. cooperation rate)", fontsize=10)
        ax.legend(fontsize=9)
        ax.set_xlim(0, layers[-1])

    plt.suptitle(
        "Payoff Salience: Attention to Payoff Values vs. Cooperation Rate\n"
        "(full paper prompt, averaged over both prompt orderings)",
        fontsize=13
    )
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    print(f"\nFigure saved to {save_path}")
    plt.show()


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── STEP 0: validate payoff position detection for all models ──────────────
    for model_key in MODELS:
        run_full_validation(model_key)

    # ── STEP 1: load cooperation matrices ─────────────────────────────────────
    TEMP_DIR = "temperature_results"
    coop = {
        "Llama-3.1-8B": load_cooperation_matrix(f"{TEMP_DIR}/llama_temp_0.8_mean.txt"),
        "Qwen2.5-7B":   load_cooperation_matrix(f"{TEMP_DIR}/qwen_temp_0.8_mean.txt"),
        "Mistral-7B":   load_cooperation_matrix(f"{TEMP_DIR}/mistral_temp_0.8_mean.txt"),
    }

    model_keys = {
        "Llama-3.1-8B": "llama",
        "Qwen2.5-7B":   "qwen",
        "Mistral-7B":   "mistral",
    }

    # ── STEP 2: collect attention weights (both orderings, averaged) ───────────
    attn = {
        label: collect_attention_all_games(key)
        for label, key in model_keys.items()
    }

    # ── STEP 3: correlate with cooperation rates ───────────────────────────────
    print("\nComputing correlations...")
    correlations = {
        label: correlate_attention_with_cooperation(attn[label], coop[label])
        for label in model_keys
    }

    # ── STEP 4: plot ───────────────────────────────────────────────────────────
    plot_attention_results(correlations)
