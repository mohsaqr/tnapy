# R script to generate expected values for Python TNA package verification
# Run this in R with the tna package installed

library(tna)

# Load the data
data(group_regulation)

# Build model
model <- tna(group_regulation)

# Print weight matrix
cat("=== WEIGHT MATRIX ===\n")
print(round(model$weights, 6))

# Print initial probabilities
cat("\n=== INITIAL PROBABILITIES ===\n")
print(round(model$inits, 6))

# Print state labels
cat("\n=== LABELS ===\n")
print(model$labels)

# Compute centralities without loops
cat("\n=== CENTRALITIES (loops=FALSE) ===\n")
cent <- centralities(model, loops = FALSE)
print(cent)

# Compute centralities with loops
cat("\n=== CENTRALITIES (loops=TRUE) ===\n")
cent_loops <- centralities(model, loops = TRUE)
print(cent_loops)

# Compute normalized centralities
cat("\n=== CENTRALITIES (normalized=TRUE) ===\n")
cent_norm <- centralities(model, normalize = TRUE)
print(cent_norm)

# Test frequency model
cat("\n=== FREQUENCY MODEL WEIGHTS ===\n")
fmodel <- ftna(group_regulation)
print(round(fmodel$weights, 2))

# Test co-occurrence model
cat("\n=== CO-OCCURRENCE MODEL WEIGHTS ===\n")
cmodel <- ctna(group_regulation)
print(round(cmodel$weights, 6))

# Bootstrap analysis
cat("\n=== BOOTSTRAP (seed=42, iter=100) ===\n")
set.seed(42)
boot <- bootstrap(model, iter = 100)

cat("\nweights_orig (column-major vector):\n")
cat(sprintf("%.15f\n", as.vector(boot$weights_orig)))

cat("\np_values (column-major vector):\n")
cat(sprintf("%.15f\n", as.vector(boot$p_values)))

cat("\nweights_mean (column-major vector):\n")
cat(sprintf("%.15f\n", as.vector(boot$weights_mean)))

cat("\nweights_sd (column-major vector):\n")
cat(sprintf("%.15f\n", as.vector(boot$weights_sd)))

cat("\nci_lower (column-major vector):\n")
cat(sprintf("%.15f\n", as.vector(boot$ci_lower)))

cat("\nci_upper (column-major vector):\n")
cat(sprintf("%.15f\n", as.vector(boot$ci_upper)))

cat("\ncr_lower (column-major vector):\n")
cat(sprintf("%.15f\n", as.vector(boot$cr_lower)))

cat("\ncr_upper (column-major vector):\n")
cat(sprintf("%.15f\n", as.vector(boot$cr_upper)))

cat("\nweights_sig (column-major vector):\n")
cat(sprintf("%.15f\n", as.vector(boot$weights_sig)))

# Permutation test
cat("\n=== PERMUTATION TEST (seed=42, iter=100) ===\n")
n <- nrow(group_regulation)
group1 <- group_regulation[1:(n %/% 2), ]
group2 <- group_regulation[(n %/% 2 + 1):n, ]

model1 <- tna(group1)
model2 <- tna(group2)

set.seed(42)
perm <- permutation_test(model1, model2, iter = 100)

cat("\ndiffs_true (column-major vector):\n")
cat(sprintf("%.15f\n", as.vector(perm$edges$diffs_true)))

cat("\ndiffs_sig (column-major vector):\n")
cat(sprintf("%.15f\n", as.vector(perm$edges$diffs_sig)))

cat("\nedge stats:\n")
stats <- perm$edges$stats
for (i in seq_len(nrow(stats))) {
  cat(sprintf("  %s -> %s: diff=%.15f, effect=%.15f, p=%.15f\n",
              stats$from[i], stats$to[i],
              stats$diff_true[i], stats$effect_size[i], stats$p_value[i]))
}
