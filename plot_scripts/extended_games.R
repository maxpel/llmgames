library(ggplot2)
library(ggtext)
library(dplyr)
library(patchwork)

source("plotting_settings.R")

# ── Settings ───────────────────────────────────────────────────────────────────

DATA_DIR <- "../data/extended_games"

R_VAL <- 10
P_VAL <- 5

# ── Data loading ───────────────────────────────────────────────────────────────
# Matrix stored row 0 = S=20 (top), col j = T=j (left = T=0).

load_extended <- function(path) {
  m <- as.matrix(read.table(path))
  df <- expand.grid(T = 0:20, S = 0:20)
  df$value  <- mapply(function(t, s) m[21 - s, t + 1], df$T, df$S)
  df$inside <- df$T >= 5 & df$T <= 15 & df$S <= 10
  df
}

# ── Nash equilibrium cooperation probability ───────────────────────────────────
# Mixed Nash: p = (P-S)/(P+R-S-T) when T>R and S>P.
# Risk dominance when both pure Nash: cooperate iff T < S+P.

nash_p <- function(S, T) {
  case_when(
    T <= R_VAL & S >  P_VAL                     ~ 1,
    T >  R_VAL & S <= P_VAL                     ~ 0,
    T <= R_VAL & S <= P_VAL & T <  S + P_VAL   ~ 1,
    T <= R_VAL & S <= P_VAL & T >  S + P_VAL   ~ 0,
    T <= R_VAL & S <= P_VAL & T == S + P_VAL   ~ 0.5,
    T >  R_VAL & S >  P_VAL                     ~ (P_VAL - S) / (P_VAL + R_VAL - S - T),
    TRUE                                         ~ NA_real_
  )
}

llama_df   <- load_extended(file.path(DATA_DIR, "llama_extended_mean.txt"))
qwen_df    <- load_extended(file.path(DATA_DIR, "qwen_extended_mean.txt"))
mistral_df <- load_extended(file.path(DATA_DIR, "mistral_extended_mean.txt"))

nash_df <- expand.grid(T = 0:20, S = 0:20) %>%
  mutate(value  = nash_p(S, T),
         inside = T >= 5 & T <= 15 & S <= 10)

# ── Panel factory ──────────────────────────────────────────────────────────────
# Black rectangle marks the original 121-game subspace (T in [5,15], S in [0,10]).

make_panel <- function(df, title) {
  ggplot() +
    geom_raster(data = filter(df, !inside),
                aes(x = T, y = S, fill = value), alpha = 0.6) +
    geom_raster(data = filter(df,  inside),
                aes(x = T, y = S, fill = value), alpha = 1.0) +
    annotate("rect",
             xmin = 4.5, xmax = 15.5, ymin = -0.5, ymax = 10.5,
             fill = NA, color = "black", linewidth = 0.7) +
    scale_fill_viridis_c(
      limits = c(0, 1), breaks = seq(0, 1, 0.2),
      name   = NULL,
      guide  = guide_colorbar(barheight      = unit(1, "null"),
                              barwidth       = unit(0.5, "cm"),
                              title          = "**p(C)**",
                              title.position = "top")
    ) +
    scale_x_continuous(breaks = c(0, 5, 10, 15, 20),
                       limits = c(-0.5, 20.5), expand = c(0, 0)) +
    scale_y_continuous(breaks = c(0, 5, 10, 15, 20),
                       limits = c(-0.5, 20.5), expand = c(0, 0)) +
    coord_fixed() +
    labs(title = title, x = "T", y = "S") +
    theme_paper(scale = 1.8, grid_x = FALSE) +
    theme(
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      axis.ticks       = element_blank(),
      plot.title       = element_text(hjust = 0.5),
      legend.title     = element_markdown(margin = margin(b = 10))
    )
}

# ── Combine and save ───────────────────────────────────────────────────────────

combine_panels <- function(left, right) {
  (left | plot_spacer() | right) +
    plot_layout(widths = c(1, 0.05, 1), guides = "collect") +
    plot_annotation(tag_levels = "A", tag_suffix = ")") &
    theme(
      plot.tag          = element_text(size = 11 * 2.5, face = "bold"),
      plot.tag.position = c(0.01, 0.98),
      legend.position   = "right"
    )
}

p_llama   <- make_panel(llama_df,   "Llama")
p_nash    <- make_panel(nash_df,    "Nash")
p_qwen    <- make_panel(qwen_df,    "Qwen")
p_mistral <- make_panel(mistral_df, "Mistral")

W <- TEXTWIDTH * 2
H <- TEXTWIDTH * 0.95

save_cropped_transparent(combine_panels(p_llama, p_nash),    "../plots/extended_games_llama.pdf",        W, H)
save_cropped_transparent(combine_panels(p_qwen,  p_mistral), "../plots/extended_games_qwen_mistral.pdf", W, H)
message("Saved ../plots/extended_games_llama.pdf and ../plots/extended_games_qwen_mistral.pdf")
