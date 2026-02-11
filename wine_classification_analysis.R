# ============================================================
# STA 135 Final Project - Wine Data Analysis (Final Version)
# ============================================================

# -----------------------------
# Load libraries
# -----------------------------
library(tidyverse)
library(GGally)        
library(corrplot)      
library(factoextra)    
library(caret)         
library(pROC)          
library(broom)         
library(MASS)
library(Hotelling)

set.seed(7)

# -----------------------------
# 0) Load & cast categorical
# -----------------------------

# Load the data
wine <- read.csv("/Users/royho/Desktop/wine.csv")

# Treat red and quality as categorical
wine <- wine %>%
  mutate(
    red = factor(red, levels = c(0,1), labels = c("White","Red")),
    quality = factor(quality, levels = sort(unique(quality)))
  )

# Grouping vars (categorical) and continuous vars
grouping_vars <- c("red","quality")
continuous_vars <- names(wine)[sapply(wine, is.numeric)]

cat("\nContinuous vars:\n"); print(continuous_vars)
cat("\nGrouping vars (categorical):\n"); print(grouping_vars)

glimpse(wine)
summary(wine[, c(grouping_vars, continuous_vars)])

# -----------------------------
# 1) Correlations
# -----------------------------

cor_mat <- cor(wine[continuous_vars], use = "pairwise.complete.obs")

cor_long <- as.data.frame(as.table(cor_mat)) %>%
  filter(Var1 != Var2) %>%
  group_by(pair = paste(pmin(Var1, Var2), pmax(Var1, Var2), sep = " :: ")) %>%
  summarize(Var1 = first(Var1), Var2 = first(Var2), r = first(Freq), .groups = "drop") %>%
  mutate(abs_r = abs(r)) %>%
  arrange(desc(abs_r))

cor_long

corrplot(cor_mat, method = "color", type = "upper",
         tl.col = "black", addCoef.col = "black", number.cex = 0.6)

# -----------------------------
# 2) PCA
# -----------------------------

pca <- prcomp(wine[continuous_vars], center = TRUE, scale. = TRUE)

eig <- pca$sdev^2
var_explained <- eig / sum(eig)
cum_var <- cumsum(var_explained)

pca_tbl <- tibble(
  PC = paste0("PC", seq_along(var_explained)),
  VarExplained = var_explained,
  CumVarExplained = cum_var
) %>%
  mutate(across(where(is.numeric), ~ round(.x, 4)))

cat("\nPCA variance explained (first 6 PCs):\n")
print(head(pca_tbl, 6))

fviz_eig(pca, addlabels = TRUE)

fviz_pca_biplot(
  pca,
  geom.ind = "point",
  habillage = wine$red,
  addEllipses = TRUE,
  ellipse.level = 0.95,
  title = "PCA Biplot (colored by red vs white)",
  palette = c("blue", "red")
)

# -----------------------------
# 3) Boxplots
# -----------------------------

wine %>%
  pivot_longer(all_of(continuous_vars), names_to = "feature", values_to = "value") %>%
  ggplot(aes(x = red, y = value, fill = red)) +
  geom_boxplot(outlier.alpha = 0.3) +
  facet_wrap(~ feature, scales = "free_y") +
  labs(title = "Continuous features by wine type", x = "", y = "") +
  theme_bw() +
  theme(legend.position = "none")

wine %>%
  pivot_longer(all_of(continuous_vars), names_to = "feature", values_to = "value") %>%
  ggplot(aes(x = quality, y = value, fill = quality)) +
  geom_boxplot(outlier.alpha = 0.3) +
  facet_wrap(~ feature, scales = "free_y") +
  labs(title = "Continuous features by quality (5,6,7)", x = "Quality", y = "") +
  theme_bw() +
  theme(legend.position = "none")

# -----------------------------
# 4) Logistic Regression: Red vs White
# -----------------------------

idx <- createDataPartition(wine$red, p = 0.7, list = FALSE)
train <- wine[idx, ]
test  <- wine[-idx, ]

fit_red <- glm(red ~ ., data = train[, c("red", continuous_vars)], family = binomial)

cat("\nRed-vs-White logistic regression summary:\n")
print(summary(fit_red))

prob_red <- predict(fit_red, newdata = test, type = "response")

roc_red <- roc(response = test$red, predictor = prob_red, levels = c("White","Red"))
cat("\nRed vs White: AUC (full model): ", round(auc(roc_red), 4), "\n")

pred_red <- factor(if_else(prob_red >= 0.5, "Red", "White"),
                   levels = c("White","Red"))

cm_red <- confusionMatrix(pred_red, test$red)
cat("\nConfusion Matrix:\n")
print(cm_red)

# -----------------------------
# 5) Highest Quality (7) vs Others
# -----------------------------

wine <- wine %>%
  mutate(high_quality = factor(if_else(as.integer(as.character(quality)) == 7,
                                       "Top", "Other")))

table(wine$high_quality)

idx2 <- createDataPartition(wine$high_quality, p = 0.7, list = FALSE)
train2 <- wine[idx2, ]
test2  <- wine[-idx2, ]

fit_hq <- glm(high_quality ~ ., 
              data = train2[, c("high_quality", continuous_vars)], 
              family = binomial)

cat("\nHighest-quality logistic regression summary:\n")
print(summary(fit_hq))

prob_hq <- predict(fit_hq, newdata = test2, type = "response")

roc_hq <- roc(response = test2$high_quality, predictor = prob_hq,
              levels = c("Other","Top"))

cat("\nHighest quality vs others: AUC: ", round(auc(roc_hq), 4), "\n")

pred_hq <- factor(if_else(prob_hq >= 0.5, "Top", "Other"),
                  levels = c("Other","Top"))

cm_hq <- confusionMatrix(pred_hq, test2$high_quality)
cat("\nConfusion Matrix:\n")
print(cm_hq)

# -----------------------------
# 6) LDA
# -----------------------------

lda_red <- lda(red ~ ., data = wine[, c("red", continuous_vars)])
cat("\n--- LDA Red vs White ---\n")
print(lda_red)

pred_lda_red <- predict(lda_red)$class
print(table(True = wine$red, Pred = pred_lda_red))

plot(lda_red, col = c("blue", "red"))

lda_q <- lda(quality ~ ., data = wine[, c("quality", continuous_vars)])
cat("\n--- LDA Quality ---\n")
print(lda_q)

plot(lda_q, col = c("green", "orange", "purple"))

# -----------------------------
# 7) Hotelling's T²
# -----------------------------

red_data   <- wine[wine$red == "Red", continuous_vars]
white_data <- wine[wine$red == "White", continuous_vars]

cat("\n--- Hotelling's T^2 Test ---\n")
htest <- hotelling.test(red_data, white_data)
print(htest)

# -----------------------------
# 8) MANOVA
# -----------------------------

manova_fit <- manova(as.matrix(wine[continuous_vars]) ~ quality, data = wine)

cat("\n--- MANOVA: Effect of Quality ---\n")
print(summary(manova_fit, test = "Wilks"))
