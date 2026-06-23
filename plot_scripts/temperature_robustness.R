library(ggplot2)
library(ggtext)
library(dplyr)
library(tidyr)
library(purrr)
library(tibble)

source("plotting_settings.R")

# ── Settings ───────────────────────────────────────────────────────────────────

DATA_DIR <- "../data"
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

load_matrix <- function(path) as.vector(t(as.matrix(read.table(path))))

human_vec <- load_matrix(file.path(DATA_DIR, "matrix_human.txt"))
nash_vec  <- load_matrix(file.path(DATA_DIR, "matrix_nash.txt"))

temp_data <- imap_dfr(TEMP_PREFIXES, function(prefix, model_label) {
  map_dfr(TEMP_VALUES, function(temp) {
    vec <- load_matrix(file.path(TEMP_DIR, sprintf("%s_temp_%.1f_mean.txt", prefix, temp)))
    tibble(
      model         = model_label,
      temperature   = temp,
      pearson_human = cor(vec, human_vec),
      msd_human     = mean((vec - human_vec)^2),
      pearson_nash  = cor(vec, nash_vec),
      msd_nash      = mean((vec - nash_vec)^2)
    )
  })
}) %>%
  mutate(model = factor(model, levels = MODEL_ORDER))

# ── Plot ───────────────────────────────────────────────────────────────────────

temp_long <- temp_data %>%
  pivot_longer(c(pearson_human, msd_human, pearson_nash, msd_nash),
               names_to = "metric", values_to = "value") %>%
  mutate(
    reference   = factor(ifelse(grepl("human", metric), "Human", "Nash"),
                         levels = c("Human", "Nash")),
    metric_type = factor(ifelse(grepl("pearson", metric), "Pearson r", "MSD"),
                         levels = c("Pearson r", "MSD"))
  )

p <- ggplot(temp_long, aes(x = temperature, y = value, color = model, group = model)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 2.5) +
  geom_vline(xintercept = 0.8, linetype = "dashed", color = "grey50", linewidth = 0.6) +
  facet_grid(metric_type ~ reference, scales = "free_y") +
  scale_color_manual(values = MODEL_COLORS, name = NULL) +
  scale_x_continuous(breaks = seq(0, 1, 0.2)) +
  labs(x = "Temperature", y = "") +
  theme_paper(scale = 1.75, grid_x = FALSE,
              legend.position   = "bottom",
              legend.direction  = "horizontal",
              legend.title      = element_blank(),
              legend.box.spacing = unit(-0.2, "pt"))

# ── Save ───────────────────────────────────────────────────────────────────────

ggsave("../plots/temperature_robustness_half.pdf", p,
       width  = TEXTWIDTH,
       height = figure_height(4, scale = 1.75, fixed = 3),
       device = cairo_pdf)
system("pdfcrop ../plots/temperature_robustness_half.pdf ../plots/temperature_robustness_half.pdf")
message("Saved ../plots/temperature_robustness_half.pdf")
