library(ggplot2)
library(ggtext)
library(dplyr)
library(purrr)
library(patchwork)

source("plotting_settings.R")

# ── Settings ───────────────────────────────────────────────────────────────────

STAGE_DIR <- "../data/prompting_stages"

S_VALUES <- 0:10
T_VALUES <- 5:15

MODEL_ORDER <- c("Llama-3.1-8B", "Qwen2.5-7B", "Mistral-7B")

TEMP_PREFIXES <- c(
  "Llama-3.1-8B" = "llama",
  "Qwen2.5-7B"   = "qwen",
  "Mistral-7B"   = "mistral"
)

N_REPS <- 20
SCALE  <- 0.8

# ── Data loading ───────────────────────────────────────────────────────────────
# bypass.txt rows = S_VALUES (0:10), cols = T_VALUES (5:15)

bypass_data <- imap_dfr(TEMP_PREFIXES, function(prefix, model_label) {
  mat <- as.matrix(read.table(file.path(STAGE_DIR, sprintf("%s_final_bypass.txt", prefix))))
  expand.grid(s_idx = seq_len(nrow(mat)), t_idx = seq_len(ncol(mat))) %>%
    mutate(
      s     = S_VALUES[s_idx],
      t     = T_VALUES[t_idx],
      count = mat[cbind(s_idx, t_idx)],
      model = model_label
    )
}) %>%
  mutate(
    model = factor(model, levels = MODEL_ORDER),
    rate  = count / N_REPS
  )

# ── Descriptive statistics for Appendix A.7 text ───────────────────────────────
# Per-model summary of bypass rates across all 121 games, plus the worst single
# game, plus correlation of bypass rate with S and T (spatial pattern claim).

bypass_summary <- bypass_data %>%
  group_by(model) %>%
  summarise(
    mean_rate    = mean(rate),
    median_rate  = median(rate),
    max_rate     = max(rate),
    max_s        = s[which.max(rate)],
    max_t        = t[which.max(rate)],
    n_games      = n(),
    n_over_25pct = sum(rate > 0.25),
    n_over_50pct = sum(rate > 0.50),
    n_over_75pct = sum(rate > 0.75),
    cor_with_s   = cor(s, rate),
    cor_with_t   = cor(t, rate),
    .groups = "drop"
  )

cat("\n==================== Bypass rate summary (Appendix A.7) ====================\n")
for (i in seq_len(nrow(bypass_summary))) {
  row <- bypass_summary[i, ]
  cat(sprintf(
    "%-13s | mean=%.1f%%  median=%.1f%%  max=%.1f%% (at S=%d, T=%d)\n",
    row$model, 100 * row$mean_rate, 100 * row$median_rate, 100 * row$max_rate,
    as.integer(row$max_s), as.integer(row$max_t)
  ))
  cat(sprintf(
    "              | games >25%%: %d/%d   >50%%: %d/%d   >75%%: %d/%d\n",
    row$n_over_25pct, row$n_games, row$n_over_50pct, row$n_games,
    row$n_over_75pct, row$n_games
  ))
  cat(sprintf(
    "              | corr(rate, S)=%.3f   corr(rate, T)=%.3f\n\n",
    row$cor_with_s, row$cor_with_t
  ))
}
cat("===============================================================================\n\n")

# ── Panels ─────────────────────────────────────────────────────────────────────
# Only Qwen (middle) shows x-axis label; only Llama (first) shows y-axis label.

panels <- map(seq_along(MODEL_ORDER), function(idx) {
  model_label <- MODEL_ORDER[idx]
  is_middle   <- idx == 2
  
  p <- ggplot(filter(bypass_data, model == model_label),
              aes(x = t, y = s, fill = count)) +
    geom_tile(color = NA) +
    scale_fill_gradient(
      low = "white", high = "black",
      name   = "\\# Reps",
      limits = c(0, N_REPS),
      guide  = guide_colorbar(barwidth = 0.8 * SCALE, barheight = unit(1, "null"))
    ) +
    scale_x_continuous(breaks = seq(min(T_VALUES), max(T_VALUES), by = 2)) +
    scale_y_continuous(breaks = seq(min(S_VALUES), max(S_VALUES), by = 2)) +
    coord_fixed(ratio = 1, expand = FALSE) +
    labs(x = if (is_middle) "T" else NULL, y = "S", title = model_label) +
    theme_paper(
      scale        = SCALE,
      panel.grid   = element_blank(),
      panel.border = element_rect(color = "grey70", fill = NA, linewidth = 0.5),
      plot.title   = element_text(size = 10 * SCALE, face = "bold", hjust = 0.5),
      legend.title = element_markdown(size = 10 * SCALE, margin = margin(b = 6))
    )
  
  if (!is_middle)
    p <- p + theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())
  if (idx != 1)
    p <- p + theme(axis.text.y  = element_blank(), axis.ticks.y  = element_blank(),
                   axis.title.y = element_blank())
  p
})

# ── Assemble and save ──────────────────────────────────────────────────────────

p_out <- wrap_plots(panels, ncol = 3) +
  plot_layout(guides = "collect") &
  theme(legend.position = "right")

ggsave("../plots/problematic_games.pdf", p_out,
       width  = TEXTWIDTH,
       height = figure_height(2.5, scale = SCALE, fixed = 0.5),
       device = cairo_pdf)
system("pdfcrop ../plots/problematic_games.pdf ../plots/problematic_games.pdf")
message("Saved ../plots/problematic_games.pdf")