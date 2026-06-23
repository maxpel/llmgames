library(reticulate)
library(ggplot2)
library(ggtext)
library(dplyr)
library(purrr)

source("plotting_settings.R")

# ── Settings ───────────────────────────────────────────────────────────────────

DATA_DIR  <- "../data/attention"
COOP_DIR  <- "../data/temperature_ablation"

S_VALUES <- 0:10
T_VALUES <- 5:15
N_GAMES  <- 121

SMOOTH_WINDOW <- 3

PAYOFF_COLORS <- c(
  R = "#009E73",
  S = "#56B4E9",
  T = "#D55E00",
  P = "#E69F00"
)

MODELS <- list(
  "Llama-3.1-8B" = list(
    attn = "attn_full_prompt_llama_avg.npy",
    coop = "llama_temp_0.8_mean.txt"
  ),
  "Qwen2.5-7B" = list(
    attn = "attn_full_prompt_qwen_avg.npy",
    coop = "qwen_temp_0.8_mean.txt"
  ),
  "Mistral-7B" = list(
    attn = "attn_full_prompt_mistral_avg.npy",
    coop = "mistral_temp_0.8_mean.txt"
  )
)

# ── Data loading ───────────────────────────────────────────────────────────────

np <- import("numpy")

load_attn <- function(path) {
  py_run_string(sprintf("
import numpy as np
_raw = np.load('%s', allow_pickle=True).item()
_R = _raw['R']; _S = _raw['S']; _T = _raw['T']; _P = _raw['P']
  ", path))
  list(R = py$`_R`, S = py$`_S`, T = py$`_T`, P = py$`_P`)
}

load_coop <- function(path) {
  m <- as.matrix(read.table(path))
  as.vector(t(m[nrow(m):1, ]))
}

rolling_mean <- function(x, k) {
  hw <- floor(k / 2)
  sapply(seq_along(x), function(i) mean(x[max(1, i-hw):min(length(x), i+hw)]))
}

# ── Build correlation data ─────────────────────────────────────────────────────

all_data <- imap_dfr(MODELS, function(cfg, model_label) {
  attn     <- load_attn(file.path(DATA_DIR, cfg$attn))
  coop     <- load_coop(file.path(COOP_DIR, cfg$coop))
  n_layers <- ncol(attn[["R"]])

  imap_dfr(attn, function(mat, payoff) {
    r_raw <- apply(mat, 2, function(col) cor(col, coop))
    tibble(
      model   = model_label,
      payoff  = payoff,
      layer   = seq(0, n_layers - 1),
      pearson = rolling_mean(r_raw, SMOOTH_WINDOW),
      mean_r  = mean(r_raw)
    )
  })
}) %>%
  mutate(
    model  = factor(model,  levels = names(MODELS)),
    payoff = factor(payoff, levels = c("R", "S", "T", "P"))
  )

# ── Print means ───────────────────────────────────────────────────────────────

all_data %>%
  distinct(model, payoff, mean_r) %>%
  arrange(model, payoff) %>%
  print(n = Inf)

# ── Plot ───────────────────────────────────────────────────────────────────────

make_plot <- function(data, scale = 1) {
  mean_data <- data %>% distinct(model, payoff, mean_r)

  ggplot(data, aes(x = layer, y = pearson, color = payoff, group = payoff)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey50", linewidth = 0.5 * scale) +
    geom_hline(data = mean_data, aes(yintercept = mean_r, color = payoff),
               linetype = "dotted", linewidth = 0.8 * scale, show.legend = FALSE) +
    geom_line(linewidth = 0.9 * scale) +
    facet_wrap(~ model, nrow = 1, scales = "free_x") +
    scale_color_manual(values = PAYOFF_COLORS, name = NULL) +
    labs(x = "Layer", y = "Pearson *r*<br>(attention vs. cooperation rate)") +
    theme_paper(
      scale             = scale,
      legend.position   = "bottom",
      legend.direction  = "horizontal",
      legend.title      = element_blank(),
      legend.box.spacing = unit(-5 * scale, "pt"),
      legend.key.size   = unit(0.8 * scale, "cm"),
      panel.spacing.x   = unit(1.5 * scale, "lines"),
      axis.title.y      = element_markdown(size = 11 * scale, margin = margin(r = 6 * scale))
    )
}

# ── Save ───────────────────────────────────────────────────────────────────────

p_llama  <- make_plot(filter(all_data, model == "Llama-3.1-8B"), scale = 0.85)
p_others <- make_plot(filter(all_data, model != "Llama-3.1-8B"))

ggsave("../plots/attention_llama.pdf",  p_llama,  width = TEXTWIDTH/2, height = figure_height(4, 0.5, 2), device = cairo_pdf)
ggsave("../plots/attention_others.pdf", p_others, width = TEXTWIDTH,   height = 3.5,                       device = cairo_pdf)
system("pdfcrop ../plots/attention_llama.pdf  ../plots/attention_llama.pdf")
system("pdfcrop ../plots/attention_others.pdf ../plots/attention_others.pdf")
message("Saved ../plots/attention_llama.pdf and ../plots/attention_others.pdf")
