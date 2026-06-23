library(ggplot2)
library(ggtext)
library(ggrepel)
library(dplyr)
library(purrr)
library(tibble)
library(patchwork)

source("plotting_settings.R")

# ── Settings ───────────────────────────────────────────────────────────────────

DATA_DIR <- "../data"
TEMP_DIR <- "../data/temperature_ablation"

SCALE_TILE <- 1.6
SCALE_MAIN <- 1.75

R_VAL    <- 10
P_VAL    <- 5
S_VALUES <- 0:10
T_VALUES <- 5:15

GAMES <- expand.grid(S = S_VALUES, T = T_VALUES) %>% arrange(S, T)

EMPIRICAL_ORDER <- c("Human", "Llama", "Qwen", "Mistral", "Nash")
PHENOTYPE_ORDER <- c("Optimist", "Pessimist", "Envious", "Trustful")

MODEL_COLORS <- c(
  "Human"        = "#000000",
  "Llama-3.1-8B" = "#009E73",
  "Qwen2.5-7B"   = "#D55E00",
  "Mistral-7B"   = "#0072B2",
  "Nash"         = "#999999"
)

PHENOTYPE_COLORS <- c(
  "Optimist"  = "#555555",
  "Pessimist" = "#555555",
  "Envious"   = "#555555",
  "Trustful"  = "#555555"
)

# ── Data loading ───────────────────────────────────────────────────────────────

load_matrix <- function(path) {
  m <- as.matrix(read.table(path))
  as.vector(t(m[nrow(m):1, ]))
}

empirical_short <- list(
  "Human"   = load_matrix(file.path(DATA_DIR, "matrix_human.txt")),
  "Llama"   = load_matrix(file.path(TEMP_DIR, "llama_temp_0.8_mean.txt")),
  "Qwen"    = load_matrix(file.path(TEMP_DIR, "qwen_temp_0.8_mean.txt")),
  "Mistral" = load_matrix(file.path(TEMP_DIR, "mistral_temp_0.8_mean.txt")),
  "Nash"    = load_matrix(file.path(DATA_DIR, "matrix_nash.txt"))
)

empirical_long <- list(
  "Human"        = empirical_short[["Human"]],
  "Llama-3.1-8B" = empirical_short[["Llama"]],
  "Qwen2.5-7B"   = empirical_short[["Qwen"]],
  "Mistral-7B"   = empirical_short[["Mistral"]],
  "Nash"         = empirical_short[["Nash"]]
)

# Boundary cells (T==R, S==P, S==T) get 0.5 (mixed strategy) so the purple
# strip visible in the paper figure is reproduced.
ideal <- list(
  "Optimist"  = dplyr::case_when(
    GAMES$T < R_VAL               ~ 1.0,
    GAMES$T > R_VAL               ~ 0.0,
    TRUE                          ~ 0.5),
  "Pessimist" = dplyr::case_when(
    GAMES$S > P_VAL               ~ 1.0,
    GAMES$S < P_VAL               ~ 0.0,
    TRUE                          ~ 0.5),
  "Envious"   = dplyr::case_when(
    GAMES$S >= GAMES$T             ~ 1.0,
    GAMES$S < GAMES$T             ~ 0.0),
  "Trustful"  = rep(1.0, nrow(GAMES))
)

# ── Panel A: Ideal phenotype tile plots ────────────────────────────────────────
# geom_tile (vector, not raster) avoids the PNG colour-shift in the
# svglite→rsvg pipeline; scale_fill_viridis_c matches methods_figure.R.
# y labels/ticks: first panel only.  x labels/ticks: last panel only.

phenotype_df <- imap_dfr(ideal, function(vec, name) {
  GAMES %>% mutate(phenotype = name, cooperate = as.numeric(vec))
}) %>%
  mutate(phenotype = factor(phenotype, levels = PHENOTYPE_ORDER))

make_phenotype_tile <- function(ph_name, show_y = FALSE, show_x = FALSE) {
  df <- filter(phenotype_df, phenotype == ph_name)
  ggplot(df, aes(x = T, y = S, fill = cooperate)) +
    geom_raster(interpolate = FALSE) +
    # scale_fill_gradient2(low = "#1A4FAD", mid = "#993399", high = "#CC2222",
    #                      midpoint = 0.5, limits = c(0, 1), guide = "none") +
    scale_fill_viridis_c(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
    scale_x_continuous(breaks = c(5, 10, 15),
                       limits = c(4.5, 15.5), expand = c(0, 0)) +
    scale_y_continuous(breaks = c(0, 5, 10),
                       limits = c(-0.5, 10.5), expand = c(0, 0)) +
    coord_fixed() +
    labs(title = ph_name,
         x = if (show_x) "T" else NULL,
         y = if (show_y) "S" else NULL) +
    theme_paper(scale = SCALE_TILE, grid_x = FALSE, grid_y = FALSE, legend = FALSE) +
    theme(
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.border     = element_blank(),
      plot.title       = element_text(hjust = 0.5, face = "bold",
                                      margin = margin(t = 0, b = 2)),
      axis.text.x      = if (show_x) element_text() else element_blank(),
      axis.ticks.x     = element_blank(),
      axis.text.y      = if (show_y) element_text() else element_blank(),
      axis.ticks.y     = element_blank(),
      plot.margin      = margin(t = 2, r = 6, b = 2, l = 2)
    )
}

n_ph            <- length(PHENOTYPE_ORDER)
tile_plots      <- imap(PHENOTYPE_ORDER, function(ph, i)
  make_phenotype_tile(ph, show_y = (i == 1), show_x = (i == n_ph)))
tile_plots[[1]] <- tile_plots[[1]] + labs(tag = "A)") +
  theme(plot.tag.position = c(0.2, 0.98))
panel_a         <- (tile_plots[[1]] + plot_spacer() + tile_plots[[2]] +
                    plot_spacer() + tile_plots[[3]] + plot_spacer() + tile_plots[[4]]) +
  plot_layout(widths = c(1.35, 0.05, 1.35, 0.05, 1.35, 0.05, 1.35), nrow = 1)

# ── Pearson r computation ──────────────────────────────────────────────────────

pearson_sim <- imap_dfr(empirical_short, function(emp_vec, emp_name) {
  imap_dfr(ideal, function(ideal_vec, ideal_name) {
    is_trustful <- ideal_name == "Trustful"
    r <- suppressWarnings(cor(emp_vec, as.numeric(ideal_vec)))
    if (is.na(r)) r <- mean(emp_vec)
    tibble(empirical = emp_name, phenotype = ideal_name,
           pearson_r = r, is_proxy = is_trustful)
  })
}) %>%
  mutate(
    empirical  = factor(empirical, levels = EMPIRICAL_ORDER),
    phenotype  = factor(phenotype, levels = PHENOTYPE_ORDER),
    cell_label = if_else(is_proxy,
                         sprintf("(%.2f)", pearson_r),
                         sprintf("%.2f",  pearson_r))
  )

# ── Panel B: Pearson r heatmap ─────────────────────────────────────────────────

panel_b <- ggplot(pearson_sim, aes(x = phenotype, y = empirical, fill = pearson_r)) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_text(aes(label = cell_label), color = "white", size = 6, fontface = "bold") +
  scale_fill_gradientn(
    colors = c("#D55E00", "#F0E442", "#009E73"),
    limits = range(pearson_sim$pearson_r),
    name   = "*r*",
    guide  = guide_colorbar(barheight = unit(8, "cm"), barwidth = unit(0.5, "cm"))
  ) +
  scale_y_discrete(limits = rev(EMPIRICAL_ORDER)) +
  labs(x = "Ideal phenotype", tag = "B)") +
  theme(plot.tag.position = c(0.03, 0.98)) +
  theme_paper(scale = SCALE_MAIN, grid_x = FALSE, grid_y = FALSE, legend = TRUE) +
  theme(
    panel.grid   = element_blank(),
    axis.text.x  = element_text(angle = 30, hjust = 1),
    axis.title.y = element_blank(),
    legend.title = element_markdown(),
    plot.margin  = margin(t = 2, r = 4, b = 2, l = 2)
  )

# ── MDS computation ────────────────────────────────────────────────────────────

safe_cor <- function(a, b) {
  r <- suppressWarnings(cor(a, b))
  if (!is.na(r)) r else if (sd(a) == 0) mean(b) else mean(a)
}

all_vecs   <- c(empirical_long, ideal)
all_colors <- c(MODEL_COLORS, PHENOTYPE_COLORS)
n          <- length(all_vecs)

D <- matrix(0, n, n, dimnames = list(names(all_vecs), names(all_vecs)))
for (i in seq_len(n))
  for (j in seq_len(n))
    D[i, j] <- 1 - safe_cor(all_vecs[[i]], all_vecs[[j]])

mds     <- cmdscale(D, k = 2, eig = TRUE)
var_exp <- 100 * mds$eig / sum(abs(mds$eig))
stress1 <- sqrt(sum((as.vector(dist(mds$points)) - as.vector(as.dist(D)))^2) /
                sum(as.vector(as.dist(D))^2))
cat(sprintf("Dim1 = %.1f%%  Dim2 = %.1f%%  Stress-1 = %.3f\n",
            var_exp[1], var_exp[2], stress1))

mds_df <- tibble(
  label    = names(all_vecs),
  x        = mds$points[, 1],
  y        = mds$points[, 2],
  is_ideal = label %in% names(ideal),
  color    = as.character(all_colors[label])
)

# ── Panel C: MDS plot ──────────────────────────────────────────────────────────

panel_c <- ggplot(mds_df, aes(x = x, y = y)) +
  geom_point(
    data  = filter(mds_df, is_ideal),
    fill  = "#AAAAAA", size = 8, shape = 23, color = "white", stroke = 1.0
  ) +
  geom_point(
    data  = filter(mds_df, !is_ideal),
    aes(color = color),
    fill  = NA, size = 8, shape = 21, stroke = 2.0
  ) +
  geom_text_repel(
    aes(label = label, color = color),
    size          = 5,
    fontface      = "bold",
    box.padding   = 0.5,
    point.padding = 0.7,
    force         = 3,
    segment.color = "transparent",
    max.overlaps  = 20,
    seed          = 42
  ) +
  annotate("text",
    x = Inf, y = -Inf, hjust = 1.05, vjust = -0.6,
    label    = sprintf("Stress-1 = %.3f", stress1),
    size     = 6, color = "#777777", fontface = "italic"
  ) +
  scale_color_identity(guide = "none") +
  labs(
    x   = sprintf("MDS Dimension 1 (%.1f%% variance)", var_exp[1]),
    y   = sprintf("MDS Dimension 2 (%.1f%% variance)", var_exp[2]),
    tag = "C)"
  ) +
  theme_paper(scale = SCALE_MAIN, grid_x = FALSE, legend = FALSE) +
  theme(plot.margin = margin(t = 2, r = 2, b = 2, l = 4),
        plot.tag.position = c(0.03, 0.98))

# ── Combine panels ─────────────────────────────────────────────────────────────
# A) spans full top row; B) and C) share bottom row.
# Spacer commented out — re-enable if heatmap has room to breathe.

bottom_row <- panel_b + plot_spacer() + panel_c +
  plot_layout(widths = c(1, 0.06, 0.9))

combined <- panel_a / bottom_row +
  plot_layout(heights = c(1, 2)) &
  theme(
    plot.tag = element_text(face = "bold", size = 11 * 2.5)
  )

# ── Save ───────────────────────────────────────────────────────────────────────

ggsave("../plots/phenotype_heatmaps_mds_pearson.svg", combined,
       width  = TEXTWIDTH * 2,
       height = figure_height(10, scale = 1.5, fixed = 0.5),
       device = svglite::svglite,
       bg     = "transparent")
message("Saved ../plots/phenotype_heatmaps_mds_pearson.svg")

