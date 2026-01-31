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
