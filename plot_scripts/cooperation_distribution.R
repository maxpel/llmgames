library(ggplot2)
library(ggtext)
library(dplyr)
library(purrr)
library(tibble)

source("plotting_settings.R")

# ── Settings ───────────────────────────────────────────────────────────────────

DATA_DIR <- "../data"
TEMP_DIR <- "../data/temperature_ablation"

MODEL_ORDER <- c("Llama-3.1-8B", "Qwen2.5-7B", "Mistral-7B")

SOURCE_ORDER  <- c("Human", MODEL_ORDER, "Nash")
SOURCE_COLORS <- c(
  "Human"        = "grey20",
  "Llama-3.1-8B" = "#009E73",
  "Qwen2.5-7B"   = "#D55E00",
  "Mistral-7B"   = "#0072B2",
  "Nash"         = "grey60"
)

TEMP_PREFIXES <- c(
  "Llama-3.1-8B" = "llama",
  "Qwen2.5-7B"   = "qwen",
  "Mistral-7B"   = "mistral"
)

# ── Data loading ───────────────────────────────────────────────────────────────

load_matrix <- function(path) {
  as.vector(t(as.matrix(read.table(path))))
}

human_vec <- load_matrix(file.path(DATA_DIR, "matrix_human.txt"))
nash_vec  <- load_matrix(file.path(DATA_DIR, "matrix_nash.txt"))

dist_data <- bind_rows(
  imap_dfr(TEMP_PREFIXES, function(prefix, model_label) {
    tibble(source      = model_label,
           cooperation = load_matrix(file.path(TEMP_DIR, sprintf("%s_temp_0.8_mean.txt", prefix))))
  }),
  tibble(source = "Human", cooperation = human_vec),
  tibble(source = "Nash",  cooperation = nash_vec)
) %>%
  mutate(source = factor(source, levels = SOURCE_ORDER))

# ── Summary ────────────────────────────────────────────────────────────────────

avg_coop <- dist_data %>%
  group_by(source) %>%
  summarise(mean_cooperation = round(mean(cooperation, na.rm = TRUE), 3), .groups = "drop")

message("Average cooperation rates:")
print(as.data.frame(avg_coop))

# ── Wasserstein-1 distance matrix ──────────────────────────────────────────────

wasserstein1d <- function(x, y) {
  pts    <- sort(c(x, y))
  widths <- c(diff(pts), 0)
  sum(abs(ecdf(x)(pts) - ecdf(y)(pts)) * widths)
}

coop_by_source <- split(dist_data$cooperation, dist_data$source)
sources        <- levels(dist_data$source)

w1_mat <- outer(sources, sources, Vectorize(function(a, b) {
  wasserstein1d(coop_by_source[[a]], coop_by_source[[b]])
}))
rownames(w1_mat) <- colnames(w1_mat) <- sources

message("Wasserstein-1 distance matrix:")
print(round(w1_mat, 3))

# ── Plot ───────────────────────────────────────────────────────────────────────

p <- ggplot(dist_data, aes(x = cooperation, fill = source, color = source)) +
  geom_histogram(binwidth = 0.05, boundary = 0, alpha = 0.7) +
  facet_wrap(~ source, ncol = 1, scales = "free_y") +
  scale_fill_manual(values  = SOURCE_COLORS) +
  scale_color_manual(values = SOURCE_COLORS) +
  scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
  labs(x = "Cooperation rate (mean across 20 repetitions per game)",
       y = "Number of games") +
  theme_paper(scale = 1.5, grid_x = FALSE, legend = FALSE)

# ── Save ───────────────────────────────────────────────────────────────────────

save_cropped_transparent(
  p,
  filename = "../plots/cooperation_distribution_half.pdf",
  width    = TEXTWIDTH,
  height   = figure_height(7, scale = 1.5, fixed = 3)
)
message("Saved ../plots/cooperation_distribution_half.pdf")
