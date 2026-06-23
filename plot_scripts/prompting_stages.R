library(ggplot2)
library(ggtext)
library(dplyr)
library(tidyr)
library(purrr)
library(tibble)

source("plotting_settings.R")

# ── Settings ───────────────────────────────────────────────────────────────────

DATA_DIR  <- "../data"
STAGE_DIR <- "../data/prompting_stages"

MODEL_ORDER <- c("Llama-3.1-8B", "Qwen2.5-7B", "Mistral-7B")
STAGE_ORDER <- c("Simple", "Double", "Multi-step", "Verifier")

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

STAGE_FILES <- c(
  "Simple"    = "simple",
  "Double"    = "extract",
  "Multi-step"= "multi",
  "Verifier"  = "final"
)

SCALE      <- 1.5
PT_SIZE    <- 2 * SCALE
PT_STROKE  <- 1.5 * SCALE
LW         <- 1.2 * SCALE

# ── Data loading ───────────────────────────────────────────────────────────────

load_matrix <- function(path) as.vector(t(as.matrix(read.table(path))))

human_vec <- load_matrix(file.path(DATA_DIR, "matrix_human.txt"))
nash_vec  <- load_matrix(file.path(DATA_DIR, "matrix_nash.txt"))

stages_data <- imap_dfr(TEMP_PREFIXES, function(prefix, model_label) {
  imap_dfr(STAGE_FILES, function(suffix, stage_label) {
    vec <- load_matrix(file.path(STAGE_DIR, sprintf("%s_%s_mean.txt", prefix, suffix)))
    tibble(
      model         = model_label,
      stage         = stage_label,
      pearson_human = cor(vec, human_vec),
      msd_human     = mean((vec - human_vec)^2),
      pearson_nash  = cor(vec, nash_vec),
      msd_nash      = mean((vec - nash_vec)^2)
    )
  })
}) %>%
  mutate(
    model = factor(model, levels = MODEL_ORDER),
    stage = factor(stage, levels = STAGE_ORDER)
  )

# ── Plot ───────────────────────────────────────────────────────────────────────

stages_long <- stages_data %>%
  pivot_longer(c(pearson_human, msd_human, pearson_nash, msd_nash),
               names_to = "metric", values_to = "value") %>%
  mutate(
    reference   = factor(ifelse(grepl("human", metric), "Human", "Nash"),
                         levels = c("Human", "Nash")),
    metric_type = factor(ifelse(grepl("pearson", metric), "Pearson r", "MSD"),
                         levels = c("Pearson r", "MSD"))
  ) %>%
  filter(!is.na(value))

p <- ggplot(stages_long, aes(x = stage, y = value, color = model, group = model)) +
  geom_line(linewidth = LW, alpha = 0.8) +
  geom_point(size = PT_SIZE + PT_STROKE, shape = 16, color = "white") +
  geom_point(size = PT_SIZE, shape = 1, stroke = PT_STROKE, alpha = 0.8) +
  facet_grid(metric_type ~ reference, scales = "free_y",
             labeller = as_labeller(c(Human = "Human", Nash = "Nash",
                                      `Pearson r` = "Pearson *r*", MSD = "MSD"))) +
  scale_color_manual(values = MODEL_COLORS, name = NULL) +
  scale_x_discrete(guide = guide_axis(angle = 30)) +
  scale_y_continuous(
    breaks = function(x) if (max(x, na.rm = TRUE) > 0.3) c(0.2, 0.4, 0.6, 0.8)
                         else c(0.05, 0.10, 0.15, 0.20),
    limits = function(x) c(min(x, na.rm = TRUE), max(x, na.rm = TRUE))
  ) +
  labs(x = NULL, y = NULL) +
  theme_paper(scale = SCALE,
              legend.position   = "bottom",
              legend.direction  = "horizontal",
              legend.title      = element_blank(),
              legend.box.spacing = unit(-0.2, "pt"),
              axis.title.x      = element_blank(),
              axis.title.y      = element_blank())

# ── Save ───────────────────────────────────────────────────────────────────────

save_cropped_transparent(
  p,
  filename = "../plots/prompting_stages_half.pdf",
  width    = TEXTWIDTH,
  height   = figure_height(5.5, scale = 1.5, fixed = 2)
)
message("Saved ../plots/prompting_stages_half.pdf")
