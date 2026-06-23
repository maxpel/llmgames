TEXTWIDTH <- 6.926

PT <- 1   # scale factor: 1 = full width, 2 = half width embed

theme_paper <- function(scale = PT, grid_x = TRUE, grid_y = TRUE, legend = TRUE, ...) {
  t <- theme_bw(base_size = 9 * scale) +
    theme(
      axis.title.x     = element_text(size = 11 * scale, margin = margin(t = 6*scale)),
      axis.title.y     = element_text(size = 11 * scale, margin = margin(r = 6*scale)),
      axis.text        = element_text(size = 10 * scale),
      strip.text.x = element_markdown(size = 10 * scale, face="bold",margin = margin(t = 5 * scale, b = 4 * scale)),
      strip.text.y = element_markdown(size = 10 * scale, face="bold",margin = margin(r = 5 * scale, l = 4 * scale)),
      # strip.background = element_rect(fill = "white"),
      legend.text      = element_text(size = 10 * scale),
      legend.title     = element_markdown(size = 11 * scale),
      plot.title       = element_text(size = 12 * scale, face = "bold"),
      plot.subtitle    = element_text(size = 10 * scale, color = "grey40"),
      panel.grid.minor = element_blank(),
      plot.background = element_rect(fill = NA, colour = NA),
      panel.background = element_rect(fill = NA, colour = NA),
      legend.background = element_rect(fill = NA, colour = NA),
      legend.box.background = element_rect(fill = NA, colour = NA),
      legend.key = element_rect(fill = NA, colour = NA),      # legend key backgrounds
      strip.background = element_rect(fill = NA, colour = NA) # facet strip backgrounds
    )
  if (!grid_x)
    t <- t + theme(panel.grid.minor.x = element_blank(),
                   panel.grid.major.x = element_blank())
  if (!grid_y)
    t <- t + theme(panel.grid.minor.y = element_blank(),
                   panel.grid.major.y = element_blank())
  if (!legend)
    t <- t + theme(legend.position = "none")
  
  t <- t + theme(...)
  t
}

save_cropped_transparent <- function(plot, filename, width, height) {
  svgfile=sub("\\.pdf$", ".svg",filename)
  ggsave(svgfile, plot,
         width = width, height = height,
         device = svglite::svglite, bg = "transparent")
  # remove the white background rect svglite adds
  # system(paste0("sed -i 's/fill: #ffffff/fill: none/gi' ", svgfile))
  # system(paste0("sed -i 's/fill: white/fill: none/gi' ", svgfile))
  # system(paste0("sed -i 's/fill:#ffffff/fill:none/gi' ", svgfile))
  # system(paste0("sed -i 's/fill:#FFFFFF/fill:none/gi' ", svgfile))
  system(paste0("sed -i 's/<rect \\(.*\\)fill: #FFFFFF/<rect \\1fill: none/g' ", svgfile))
  system(paste0("rsvg-convert -f pdf -o ", filename, " ", svgfile))
  system(paste("pdfcrop", filename, filename))
  system(paste0("rm ",svgfile))
  message("Saved transparently and cropped ", filename)
}

figure_height <- function(full_height = 5.5, scale = 1,
                          fixed = 1.25) {
  data_area <- full_height - fixed
  data_area + fixed * scale
}