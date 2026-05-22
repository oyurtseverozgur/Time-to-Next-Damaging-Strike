# =============================================================================
# frailty_analysis.R
# Appendix B — REML Gamma Frailty Estimation for PWP-GT Survival Model
#
# Study: "Time-to-Next Damaging Strike: Recurrent Event Survival Analysis
#         of Wildlife Strike Sequences at US Commercial Airports"
#
# Purpose:
#   Estimate the gamma frailty variance (theta) and Kendall's tau via
#   REML penalised partial likelihood, replacing the log-normal
#   approximation used in the main Python analysis.
#
# Data input:
#   gap_time_panel.csv — exported from the Python pipeline via:
#     df[cols].to_csv("gap_time_panel.csv", index=False)
#
# R version: >= 4.1.0
# Required packages: survival (>= 3.4), frailtypack (>= 3.4)
# =============================================================================


# ── 0. Install packages (run once) ───────────────────────────────────────────

# Uncomment if packages are not yet installed:
# install.packages("survival")
# install.packages("frailtypack")
# install.packages("ggplot2")   # optional, for diagnostic plots


# ── 1. Load packages ─────────────────────────────────────────────────────────

library(survival)
library(frailtypack)


# ── 2. Load data ──────────────────────────────────────────────────────────────

# Set working directory to the folder containing gap_time_panel.csv
# setwd("C:/Users/YourName/Documents/survival_analysis")   # Windows
# setwd("/home/user/survival_analysis")                     # Linux / macOS

df <- read.csv("gap_time_panel.csv", stringsAsFactors = FALSE)

cat("Panel loaded:\n")
cat("  Rows:     ", nrow(df), "\n")
cat("  Airports: ", length(unique(df$airport)), "\n")
cat("  Events:   ", sum(df$event), "\n")
cat("  Strata:   ", paste(sort(unique(df$stratum)), collapse=", "), "\n\n")

# Verify column names
expected_cols <- c("airport","gap_days","event","stratum",
                   "hub_large_exog","hub_medium_exog",
                   "autumn","spring","summer")
missing <- setdiff(expected_cols, names(df))
if (length(missing) > 0) {
  stop(paste("Missing columns:", paste(missing, collapse=", ")))
}


# ── 3. Descriptive check ──────────────────────────────────────────────────────

cat("Hub classification (exogenous NPIAS):\n")
print(table(df[df$stratum==1, "hub_large_exog"]))   # k=1 only to avoid counting same airport multiple times

cat("\nMedian gap time by stratum:\n")
for (k in 1:5) {
  sub <- df[df$stratum == k, ]
  med <- median(sub$gap_days[sub$event == 1])
  cat(sprintf("  Stratum k=%d : n=%d  median gap (events only) = %d days\n",
              k, nrow(sub), round(med)))
}


# ── 4. Static PWP-GT model (baseline, no frailty) ─────────────────────────────

cat("\n=== Static PWP-GT model (no frailty) ===\n")

fit_static <- coxph(
  Surv(gap_days, event) ~
    hub_large_exog + hub_medium_exog +
    autumn + spring + summer +
    strata(stratum),
  data    = df,
  ties    = "efron",
  cluster = airport
)

print(summary(fit_static))
cat(sprintf("Concordance (C-index): %.4f\n", fit_static$concordance["concordance"]))


# ── 5. Gamma frailty model via survival::coxph (fast REML approximation) ─────
#
# This uses the built-in frailty() term in the survival package.
# The theta reported here is the REML penalised likelihood estimate.

cat("\n=== Gamma frailty model — survival::coxph ===\n")

fit_frailty_coxph <- coxph(
  Surv(gap_days, event) ~
    hub_large_exog + hub_medium_exog +
    autumn + spring + summer +
    strata(stratum) +
    frailty(airport, distribution = "gamma"),
  data  = df,
  ties  = "efron"
)

print(summary(fit_frailty_coxph))

# Extract theta directly from the model object
theta_coxph <- fit_frailty_coxph$history[[1]]$theta
if (is.null(theta_coxph)) {
  # Alternative extraction path depending on survival package version
  theta_coxph <- fit_frailty_coxph$frail
  theta_coxph <- var(theta_coxph)
}
cat(sprintf("\nsurval::coxph  theta (frailty variance) = %.4f\n", theta_coxph))
tau_coxph <- theta_coxph / (theta_coxph + 2)
cat(sprintf("               Kendall's tau            = %.4f\n", tau_coxph))


# ── 6. Gamma frailty via frailtypack::frailtyPenal (full REML) ───────────────
#
# frailtypack uses full REML (not penalised partial likelihood),
# providing asymptotically unbiased theta and SE(theta).

cat("\n=== Gamma frailty model — frailtypack::frailtyPenal ===\n")

fit_frailtypack <- frailtyPenal(
  Surv(gap_days, event) ~
    cluster(airport) +
    hub_large_exog + hub_medium_exog +
    autumn + spring + summer +
    strata(stratum),
  data      = df,
  n.knots   = 8,
  kappa     = 1,
  RandDist  = "Gamma"
)

print(summary(fit_frailtypack))

theta_fp <- fit_frailtypack$theta
se_theta  <- fit_frailtypack$seTheta
cat(sprintf("\nfrailtypack  theta  = %.4f  (SE = %.4f)\n", theta_fp, se_theta))
cat(sprintf("             95%% CI = [%.4f, %.4f]\n",
            theta_fp - 1.96*se_theta, theta_fp + 1.96*se_theta))
tau_fp <- theta_fp / (theta_fp + 2)
cat(sprintf("             Kendall's tau = %.4f\n", tau_fp))


# ── 7. Time-varying coefficient model (hub × log(t)) ─────────────────────────
#
# Addresses the Grambsch-Therneau PH violation (hub rho = -0.426).
# tt() implements beta(t) = beta0 + beta1 * log(t).

cat("\n=== Time-varying coefficient model — survival::coxph with tt() ===\n")

fit_tvc <- coxph(
  Surv(gap_days, event) ~
    tt(hub_large_exog) + tt(hub_medium_exog) +
    autumn + spring + summer +
    strata(stratum),
  data = df,
  tt   = function(x, t, ...) x * log(t + 1),
  ties = "efron"
)

print(summary(fit_tvc))
cat(sprintf("TVC Concordance: %.4f\n", fit_tvc$concordance["concordance"]))

# Compute effective hub_large HR at representative gap times
beta0 <- coef(fit_tvc)["tt(hub_large_exog)"]
beta1 <- coef(fit_tvc)["tt(hub_medium_exog)"]   # this is beta for hub_large*log(t)

# Note: coef names depend on the tt() expansion — inspect with names(coef(fit_tvc))
cat("\nCoefficient names in TVC model:\n")
print(names(coef(fit_tvc)))

# Effective Large Hub HR = exp(beta0 + beta1*log(t))
# (Adjust coefficient names below if needed after inspecting names above)
tvc_coefs <- coef(fit_tvc)
if ("tt(hub_large_exog)" %in% names(tvc_coefs)) {
  b_hub_large_main <- tvc_coefs["hub_large_exog"]
  b_hub_large_tt   <- tvc_coefs["tt(hub_large_exog)"]

  if (!is.na(b_hub_large_main) && !is.na(b_hub_large_tt)) {
    cat("\nEffective Large Hub HR = exp(b0 + b1*log(t+1)):\n")
    for (t_ref in c(30, 180, 365, 730, 1460)) {
      hr_t <- exp(b_hub_large_main + b_hub_large_tt * log(t_ref + 1))
      cat(sprintf("  t = %5d days : HR = %.3f\n", t_ref, hr_t))
    }
  }
}


# ── 8. Schoenfeld residuals and PH test ───────────────────────────────────────

cat("\n=== Grambsch-Therneau PH Test (Schoenfeld residuals) ===\n")

ph_test <- cox.zph(fit_static, transform = "log")
print(ph_test)

# Global test
cat(sprintf("\nGlobal PH test: chi2 = %.2f  df = %d  p = %.4f\n",
            ph_test$table["GLOBAL","chisq"],
            ph_test$table["GLOBAL","df"],
            ph_test$table["GLOBAL","p"]))


# ── 9. Restricted Mean Survival Time (RMST) ───────────────────────────────────

cat("\n=== Restricted Mean Survival Time (t* = 2000 days) ===\n")

t_star <- 2000

for (k in 1:5) {
  sub  <- df[df$stratum == k, ]
  km   <- survfit(Surv(gap_days, event) ~ 1, data = sub)
  rmst <- summary(km, rmean = t_star)$table["rmean"]
  med  <- summary(km)$table["median"]
  lbl  <- if (k < 5) paste0("k=", k) else "k>=5"
  cat(sprintf("  %-6s : KM median = %5.0f d   RMST(%d d) = %5.0f d\n",
              lbl, med, t_star, rmst))
}


# ── 10. Save results to CSV ───────────────────────────────────────────────────

results <- data.frame(
  model               = c("coxph_frailty", "frailtyPenal"),
  theta               = c(theta_coxph, theta_fp),
  se_theta            = c(NA, se_theta),
  kendall_tau         = c(tau_coxph, tau_fp),
  method              = c("REML_penalised_PL", "REML_full")
)

write.csv(results, "frailty_results.csv", row.names = FALSE)
cat("\nSaved frailty_results.csv\n")

# Summary
cat("\n=== SUMMARY ===\n")
cat(sprintf("survival::coxph   theta = %.4f   Kendall tau = %.4f\n",
            theta_coxph, tau_coxph))
cat(sprintf("frailtypack       theta = %.4f   Kendall tau = %.4f  SE(theta) = %.4f\n",
            theta_fp, tau_fp, se_theta))
cat("\nInterpretation:\n")
cat("  theta > 0 confirms unobserved between-airport heterogeneity.\n")
cat("  Kendall tau = theta / (theta+2) = within-airport correlation.\n")
cat("  Compare to log-normal approximation (theta_approx = 1.11) in manuscript.\n")
cat("  Report frailtypack result as primary; coxph result as sensitivity check.\n")
