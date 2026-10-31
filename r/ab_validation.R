#!/usr/bin/env Rscript
# Independent re-analysis of the experiments in R, used to cross-check the Python engine.
# Usage: Rscript r/ab_validation.R <outputs_dir>
suppressPackageStartupMessages({
  library(dplyr)
  library(jsonlite)
  library(boot)
})

args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args) > 0) args[1] else "outputs"

analyse <- function(exp_id) {
  df <- read.csv(file.path(out_dir, sprintf("exp_%s_users.csv", exp_id))) %>%
    mutate(variant = factor(variant, levels = c("control", "treatment")))

  counts <- table(df$variant)
  srm <- chisq.test(as.numeric(counts), p = c(0.5, 0.5))

  conv <- df %>% group_by(variant) %>% summarise(x = sum(converted), n = n(), .groups = "drop")
  pt <- prop.test(x = rev(conv$x), n = rev(conv$n), correct = FALSE)   # treatment - control

  tt <- t.test(revenue ~ variant, data = df, var.equal = FALSE)
  # ANCOVA on the pre-period covariate: the regression form of CUPED.
  ancova <- lm(revenue ~ variant + pre_revenue, data = df)
  co <- summary(ancova)$coefficients["varianttreatment", ]
  ci <- confint(ancova)["varianttreatment", ]

  set.seed(42)
  ratio <- function(d, i) {
    s <- d[i, ]
    mean(s$revenue[s$variant == "treatment"]) / mean(s$revenue[s$variant == "control"]) - 1
  }
  b <- boot(df[, c("variant", "revenue")], ratio, R = 1000, strata = df$variant)
  bci <- boot.ci(b, type = "perc")$percent[4:5]

  list(
    experiment_id = exp_id,
    n = nrow(df),
    srm = list(chi2 = unname(srm$statistic), p_value = srm$p.value),
    conversion = list(p_control = conv$x[1] / conv$n[1], p_treatment = conv$x[2] / conv$n[2],
                      diff = unname(pt$estimate[1] - pt$estimate[2]),
                      diff_ci = unname(pt$conf.int), p_value = pt$p.value),
    revenue_welch = list(diff = unname(tt$estimate[2] - tt$estimate[1]), t = unname(-tt$statistic),
                         p_value = tt$p.value),
    revenue_ancova = list(estimate = unname(co["Estimate"]), se = unname(co["Std. Error"]),
                          p_value = unname(co["Pr(>|t|)"]), ci = unname(ci)),
    revenue_bootstrap_rel_ci = bci
  )
}

exps <- c("one_click_checkout", "reco_widget_v2")
res <- setNames(lapply(exps, analyse), exps)
res$r_version <- R.version.string
write_json(res, file.path(out_dir, "r_validation.json"), auto_unbox = TRUE, digits = 10, pretty = TRUE)
cat("R validation written to", file.path(out_dir, "r_validation.json"), "\n")
