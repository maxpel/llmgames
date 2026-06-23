library(dplyr)
library(tibble)

# ── Settings ───────────────────────────────────────────────────────────────────

DATA_DIR <- "../data"
TEMP_DIR <- "../data/temperature_ablation"

MODELS <- c("Llama-3.1-8B" = "llama",
            "Qwen2.5-7B"   = "qwen",
            "Mistral-7B"   = "mistral")

LABELS <- c("Llama-3.1-8B" = "\\textbf{Llama}",
            "Qwen2.5-7B"   = "\\textbf{Qwen}",
            "Mistral-7B"   = "\\textbf{Mistral}")

# ── Data loading ───────────────────────────────────────────────────────────────

load_matrix <- function(path) as.vector(t(as.matrix(read.table(path))))

human_vec <- load_matrix(file.path(DATA_DIR, "matrix_human.txt"))
nash_vec  <- load_matrix(file.path(DATA_DIR, "matrix_nash.txt"))
n         <- length(human_vec)

# ── CI helpers ─────────────────────────────────────────────────────────────────

pearson_ci <- function(r) {
  z <- atanh(r); q <- qnorm(0.975)
  c(tanh(z - q / sqrt(n - 3)), tanh(z + q / sqrt(n - 3)))
}

msd_ci <- function(a, b) {
  d <- (a - b)^2; q <- qnorm(0.975); se <- sd(d) / sqrt(length(d))
  c(mean(d) - q * se, mean(d) + q * se)
}

# ── Cell formatting ────────────────────────────────────────────────────────────

cell <- function(num, ci, bold = FALSE) {
  num_str <- if (bold) sprintf("\\textbf{%s}", num) else num
  ci_str  <- sprintf("{\\small \\textcolor{gray}{[%s]}}", ci)
  sprintf("\\shortstack{\\rule{0pt}{12pt}%s \\\\[3pt] %s}", num_str, ci_str)
}

fmt_r   <- function(r)  sprintf("%.2f", r)
fmt_msd <- function(m)  sprintf("%.3f", m)
fmt_r_ci  <- function(ci) sprintf("%.2f, %.2f", ci[1], ci[2])
fmt_msd_ci <- function(ci) sprintf("%.3f, %.3f", ci[1], ci[2])

# ── Compute metrics ────────────────────────────────────────────────────────────

rows <- lapply(names(MODELS), function(label) {
  vec <- load_matrix(file.path(TEMP_DIR, sprintf("%s_temp_0.8_mean.txt", MODELS[label])))
  rh  <- cor(vec, human_vec);  rn  <- cor(vec, nash_vec)
  mh  <- mean((vec - human_vec)^2); mn <- mean((vec - nash_vec)^2)
  list(label  = label,
       rh = rh, mh = mh, rn = rn, mn = mn,
       ci_rh  = pearson_ci(rh), ci_rn = pearson_ci(rn),
       ci_mh  = msd_ci(vec, human_vec), ci_mn = msd_ci(vec, nash_vec))
})

best_mh <- which.min(sapply(rows, `[[`, "mh"))
best_rh <- which.max(sapply(rows, `[[`, "rh"))
best_mn <- which.min(sapply(rows, `[[`, "mn"))
best_rn <- which.max(sapply(rows, `[[`, "rn"))

# ── Build LaTeX table ──────────────────────────────────────────────────────────

lines <- c(
  "\\begin{tabular}{lcccc}",
  "\\toprule",
  "& \\multicolumn{2}{c}{\\textbf{Human}} & \\multicolumn{2}{c}{\\textbf{Nash}} \\\\",
  "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}",
  "& \\textbf{MSD} & $\\boldsymbol{r}$ & \\textbf{MSD} & $\\boldsymbol{r}$ \\\\",
  "\\midrule"
)

for (i in seq_along(rows)) {
  r <- rows[[i]]
  lines <- c(lines, sprintf(
    "%s &\n  %s\n& %s\n& %s\n& %s \\\\[4pt]",
    LABELS[r$label],
    cell(fmt_msd(r$mh), fmt_msd_ci(r$ci_mh), bold = (i == best_mh)),
    cell(fmt_r(r$rh),   fmt_r_ci(r$ci_rh),   bold = (i == best_rh)),
    cell(fmt_msd(r$mn), fmt_msd_ci(r$ci_mn), bold = (i == best_mn)),
    cell(fmt_r(r$rn),   fmt_r_ci(r$ci_rn),   bold = (i == best_rn))
  ))
}

rnh <- cor(nash_vec, human_vec)
mnh <- mean((nash_vec - human_vec)^2)
lines <- c(lines,
  "\\midrule",
  sprintf("\\textbf{Nash} &\n  %s\n& %s \\\\[4pt]",
    cell(fmt_msd(mnh), fmt_msd_ci(msd_ci(nash_vec, human_vec))),
    cell(fmt_r(rnh),   fmt_r_ci(pearson_ci(rnh)))),
  "\\bottomrule",
  "\\end{tabular}"
)

# ── Write output ───────────────────────────────────────────────────────────────

out_path <- "../plots/metrics_table.tex"
writeLines(paste(lines, collapse = "\n"), out_path)
message("Saved ", out_path)
