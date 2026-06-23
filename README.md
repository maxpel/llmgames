# Large language models replicate and predict human cooperation across experiments in game theory

Code and data for the paper: Palatsi et al. ([arXiv:2511.04500](https://arxiv.org/abs/2511.04500))

## Repository structure

```
data_generation_scripts/   Python scripts to reproduce LLM simulations
plot_scripts/              R scripts to reproduce all figures and tables
data/                      Pre-generated simulation results
plots/                     Pre-generated figures
```

## Requirements

**Python** (data generation): install dependencies with `pip install -r requirements.txt`

**R** (plotting): packages used are listed at the top of each script in `plot_scripts/`.

## Reproducing the data

Each script accepts `--model llama|qwen|mistral`. Models are loaded via [vLLM](https://github.com/vllm-project/vllm) and must be available locally or on HuggingFace.

```bash
# Prompting pipeline stages (simple / extract / multi / final)
python data_generation_scripts/run_prompting_stages.py --model llama --stage final --output_dir data/prompting_stages

# Extended 441-game space
python data_generation_scripts/run_extended_games.py --model llama --output_dir data/extended_games

# Temperature ablation (T in 0.0..1.0)
python data_generation_scripts/temperature_ablation.py --model llama --output_dir data/temperature_ablation

# Attention-based payoff salience analysis
python data_generation_scripts/attention_analysis_full_prompt.py --model llama --output_dir data/attention
```

## Citation

```bibtex
@misc{palatsiLargeLanguageModels2025,
  title = {Large language models replicate and predict human cooperation across experiments in game theory},
  author = {Palatsi, Andrea Cera and {Martin-Gutierrez}, Samuel and Cardenal, Ana S. and Pellert, Max},
  year = {2025},
  month = nov,
  number = {arXiv:2511.04500},
  eprint = {2511.04500},
  primaryclass = {cs},
  publisher = {arXiv},
  doi = {10.48550/arXiv.2511.04500},
  urldate = {2025-11-07},
  abstract = {Large language models (LLMs) are increasingly used both to make decisions in domains such as health, education and law, and to simulate human behavior. Yet how closely LLMs mirror actual human decision-making remains poorly understood. This gap is critical: misalignment could produce harmful outcomes in practical applications, while failure to replicate human behavior renders LLMs ineffective for social simulations. Here, we address this gap by developing a digital twin of game-theoretic experiments and introducing a systematic prompting and probing framework for machine-behavioral evaluation. Testing three open-source models (Llama, Mistral and Qwen), we find that Llama reproduces human cooperation patterns with high fidelity, capturing human deviations from rational choice theory, while Qwen aligns closely with Nash equilibrium predictions. Notably, we achieved population-level behavioral replication without persona-based prompting, simplifying the simulation process. Extending beyond the original human-tested games, we generate and preregister testable hypotheses for novel game configurations outside the original parameter grid. Our findings demonstrate that appropriately calibrated LLMs can replicate aggregate human behavioral patterns and enable systematic exploration of unexplored experimental spaces, offering a complementary approach to traditional research in the social and behavioral sciences that generates new empirical predictions about human social decision-making.},
  archiveprefix = {arXiv},
  keywords = {Computer Science - Artificial Intelligence,Computer Science - Computation and Language,Computer Science - Computer Science and Game Theory,Computer Science - Multiagent Systems}
}
```
