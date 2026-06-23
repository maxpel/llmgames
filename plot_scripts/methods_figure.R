library(ggplot2)
library(ggtext)
library(dplyr)
library(patchwork)

source("plotting_settings.R")

# ── Settings ───────────────────────────────────────────────────────────────────

DATA_DIR  <- "../data"
STAGE_DIR <- "../data/prompting_stages"

SCALE <- 1.4

# ── Data loading ───────────────────────────────────────────────────────────────
# 11×11 matrices: row i (1-indexed) = S = 11-i, col j = T = j+4

load_matrix <- function(path) {
  m <- as.matrix(read.table(path))
  df <- expand.grid(T = 5:15, S = 0:10)
  df$value <- mapply(function(t, s) m[11 - s, t - 4], df$T, df$S)
  df
}

human_df <- load_matrix(file.path(DATA_DIR, "matrix_human.txt"))

stages <- list(
  llama   = c("simple", "extract", "multi", "final"),
  qwen    = c("simple", "extract", "multi", "final"),
  mistral = c("simple", "extract", "multi", "final")
)

stage_data <- lapply(names(stages), function(model) {
  lapply(setNames(stages[[model]], stages[[model]]), function(stage) {
    load_matrix(file.path(STAGE_DIR, sprintf("%s_%s_mean.txt", model, stage)))
  })
}) |> setNames(names(stages))

# ── Panel factory ──────────────────────────────────────────────────────────────

make_panel <- function(df, title, show_axes = FALSE, title_scale = 1) {
  p <- ggplot(df, aes(x = T, y = S, fill = value)) +
    geom_raster() +
    scale_fill_viridis_c(
      limits = c(0, 1), breaks = seq(0, 1, 0.2),
      name  = "**p(C)**",
      guide = guide_colorbar(barheight      = unit(1, "null"),
                             barwidth       = unit(0.5, "cm"),
                             title.position = "top")
    ) +
    scale_x_continuous(breaks = c(5, 10, 15),
                       limits = c(4.5, 15.5), expand = c(0, 0)) +
    scale_y_continuous(breaks = c(0, 5, 10),
                       limits = c(-0.5, 10.5), expand = c(0, 0)) +
    coord_fixed() +
    labs(title = title,
         x = if (show_axes) "T" else NULL,
         y = if (show_axes) "S" else NULL) +
    theme_paper(scale = SCALE, grid_x = FALSE) +
    theme(
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      axis.ticks       = element_blank(),
      plot.title       = element_text(hjust = 0.5, size = rel(title_scale),
                                      margin = margin(t = 0, b = 0)),
      axis.title.x     = element_text(margin = margin(t = 4)),
      axis.title.y     = element_text(margin = margin(r = -4)),
      legend.title     = element_markdown(margin = margin(b = 10))
    )

  if (!show_axes)
    p <- p + theme(axis.text = element_blank())

  p
}

# ── methods_figure.pdf ─────────────────────────────────────────────────────────
# Layout: human (tall, 2 rows) | Llama stages (2×2) | Mistral/Qwen final

p_human         <- make_panel(human_df,                     "Human",               show_axes = TRUE, title_scale = 1.5)
p_llama_simple  <- make_panel(stage_data$llama$simple,      "Llama - Simple",      show_axes = TRUE, title_scale = 1)
p_llama_extract <- make_panel(stage_data$llama$extract,     "Llama - Double",                        title_scale = 1)
p_llama_multi   <- make_panel(stage_data$llama$multi,       "Llama - Multi-steps",                  title_scale = 1)
p_llama_final   <- make_panel(stage_data$llama$final,       "Llama - Verifier",                     title_scale = 1)
p_mistral_final <- make_panel(stage_data$mistral$final,     "Mistral - Verifier",                   title_scale = 1)
p_qwen_final    <- make_panel(stage_data$qwen$final,        "Qwen - Verifier",     show_axes = TRUE, title_scale = 1)

p_methods <- (p_human + p_llama_simple + p_llama_extract + p_mistral_final +
              p_llama_multi  + p_llama_final  + p_qwen_final) +
  plot_layout(
    design = "
    A#BC#D
    A#FG#H
    ",
    widths  = c(2, 0.02, 1, 1, 0.02, 1),
    guides  = "collect"
  ) &
  theme(
    legend.position   = "right",
    legend.box.margin = margin(l = 4),
    legend.title      = element_markdown(margin = margin(b = 6))
  )

ggsave("../plots/methods_figure.pdf", p_methods,
       width  = TEXTWIDTH * 1.6 * 1.1,
       height = TEXTWIDTH * 0.705 * 1.1,
       device = cairo_pdf)
system("pdfcrop ../plots/methods_figure.pdf ../plots/methods_figure.pdf")
message("Saved ../plots/methods_figure.pdf")

# ── appendix_stages.pdf ────────────────────────────────────────────────────────
# Qwen + Mistral all 4 stages side by side, 2×2 blocks with shared legend

p_qwen_simple    <- make_panel(stage_data$qwen$simple,    "Qwen - Simple",        show_axes = TRUE, title_scale = 1)
p_qwen_extract   <- make_panel(stage_data$qwen$extract,   "Qwen - Double",                          title_scale = 1)
p_qwen_multi     <- make_panel(stage_data$qwen$multi,     "Qwen - Multi-steps",                    title_scale = 1)
p_qwen_final2    <- make_panel(stage_data$qwen$final,     "Qwen - Verifier",                       title_scale = 1)
p_mistral_simple <- make_panel(stage_data$mistral$simple, "Mistral - Simple",     show_axes = TRUE, title_scale = 1)
p_mistral_extract<- make_panel(stage_data$mistral$extract,"Mistral - Double",                       title_scale = 1)
p_mistral_multi  <- make_panel(stage_data$mistral$multi,  "Mistral - Multi-steps",                 title_scale = 1)
p_mistral_final2 <- make_panel(stage_data$mistral$final,  "Mistral - Verifier",                    title_scale = 1)

p_appendix <- (p_qwen_simple + p_qwen_extract + p_mistral_simple + p_mistral_extract +
               p_qwen_multi  + p_qwen_final2  + p_mistral_multi  + p_mistral_final2) +
  plot_layout(
    design = "
    AB#CD
    EF#GH
    ",
    widths  = c(1, 1, 0.1, 1, 1),
    guides  = "collect"
  ) &
  theme(
    legend.position   = "right",
    legend.box.margin = margin(l = 4),
    legend.title      = element_markdown(margin = margin(b = 6))
  )

ggsave("../plots/appendix_stages.pdf", p_appendix,
       width  = TEXTWIDTH * 1.45,
       height = TEXTWIDTH * 0.72,
       device = cairo_pdf)
system("pdfcrop ../plots/appendix_stages.pdf ../plots/appendix_stages.pdf")
message("Saved ../plots/appendix_stages.pdf")
