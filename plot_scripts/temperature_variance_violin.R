library(reticulate)
library(ggplot2)
library(ggtext)
library(dplyr)
library(tibble)

source("plotting_settings.R")

# ── Settings ───────────────────────────────────────────────────────────────────

TEMP_DIR <- "../data/temperature_ablation"

MODEL_ORDER <- c("Llama-3.1-8B", "Qwen2.5-7B", "Mistral-7B")

MODEL_COLORS <- c(
  "Llama-3.1-8B" = "#009E73",
  "Qwen2.5-7B"   = "#D55E00",
  "Mistral-7B"   = "#0072B2"
)

TEMP_PREFIXES <- c(
  "Llama-3.1-8B" = "llama",
  "Qwen2.5-7B"   = "qwen",
  "Mistral-7B"   = "mistral"
)

TEMP_VALUES <- seq(0.0, 1.0, by = 0.1)

# ── Data loading ───────────────────────────────────────────────────────────────
# Raw npy files have shape (repetitions, n_S, n_T).
# Per-game variance is computed across repetitions for each (S, T) cell.

np <- import("numpy")

variance_data <- bind_rows(lapply(names(TEMP_PREFIXES), function(model_label) {
  prefix <- TEMP_PREFIXES[[model_label]]
  bind_rows(lapply(TEMP_VALUES, function(temp) {
    raw <- np$load(file.path(TEMP_DIR, sprintf("%s_temp_%.1f_raw.npy", prefix, temp)))
    n_s <- dim(raw)[2]
    n_t <- dim(raw)[3]
    tibble(
      model       = model_label,
      temperature = temp,
      variance    = as.vector(apply(raw, c(2, 3), var))
    )
  }))
})) %>%
  mutate(
    model      = factor(model, levels = MODEL_ORDER),
    temp_factor = factor(temperature)
  )

# ── Plot ───────────────────────────────────────────────────────────────────────

p <- ggplot(variance_data, aes(x = temp_factor, y = variance, fill = model)) +
  geom_violin(scale = "area", alpha = 0.7, trim = TRUE) +
  stat_summary(
    fun.data = function(x) data.frame(y = median(x), ymin = quantile(x, 0.25), ymax = quantile(x, 0.75)),
    geom = "errorbar", width = 0.1, color = "black"
  ) +
  stat_summary(fun = median, geom = "point", size = 1.5, color = "black") +
  facet_wrap(~ model, ncol = 1) +
  scale_fill_manual(values = MODEL_COLORS) +
  scale_y_continuous(limits = c(0, NA)) +
  labs(x = "Temperature", y = "Variance within games") +
  theme_paper(scale = 1.75, grid_x = FALSE, legend = FALSE)

# ── Save ───────────────────────────────────────────────────────────────────────

ggsave("../plots/temperature_variance_violin_half.pdf", p,
       width  = TEXTWIDTH,
       height = figure_height(5.5, scale = 1.75, fixed = 3),
       device = cairo_pdf)
system("pdfcrop ../plots/temperature_variance_violin_half.pdf ../plots/temperature_variance_violin_half.pdf")
message("Saved ../plots/temperature_variance_violin_half.pdf")
