# Chapter 1: Introduction

## NextStep AI — Intelligent Career Prediction System

---

## 1.1 Introduction

Career selection is one of the most impactful decisions in a student's life. With the rapid expansion of technology-driven career paths, students often face difficulty in identifying which profession aligns best with their academic strengths, personality, and interests. **NextStep AI** addresses this challenge by leveraging Machine Learning to predict suitable career paths based on a student's profile — encompassing academic performance, personality traits (Big Five), technical and soft skills, behavioural habits, and domain interests.

The system collects student data through a 15-question behavioural assessment, transforms the responses into a 22-feature numeric vector, and feeds it into a trained classification model. The output is a personalised career recommendation with confidence scoring, a 5-phase career roadmap, an AI mentor chatbot, and a resume analyser — all running locally without any cloud dependency.

---

## 1.2 Significance of Machine Learning

Machine Learning (ML) is a subset of Artificial Intelligence that enables systems to learn patterns from data and make predictions without being explicitly programmed for every scenario. Its significance in this project is as follows:

| Aspect | Role of ML |
|---|---|
| **Pattern Recognition** | ML models identify complex, non-linear relationships between a student's profile features and suitable career paths — relationships that are difficult to capture with simple rule-based systems. |
| **Data-Driven Decisions** | Instead of relying on subjective career counselling, ML provides objective, data-backed recommendations based on measurable student attributes. |
| **Scalability** | Once trained, the model can serve thousands of students instantly, making personalised career guidance accessible at zero marginal cost. |
| **Continuous Improvement** | As more real-world career outcome data is collected, the model can be retrained to improve accuracy over time. |
| **Personalisation** | ML enables predictions tailored to each individual's unique combination of skills, personality, and interests — not generic advice. |

In the context of NextStep AI, ML replaces expensive, appointment-based career counselling with an instant, free, and data-driven alternative that any student can access.

---

## 1.3 Overview of Supervised Learning

This project uses **Supervised Learning**, a category of ML where the model is trained on labelled data — i.e., each training sample has a known input (student features) and a known output (career label).

### How Supervised Learning Works

1. **Training Phase:** The algorithm receives a dataset of student profiles (X) paired with their assigned career labels (Y). It learns a mapping function `f(X) → Y` by minimising prediction error.
2. **Testing Phase:** The trained model is evaluated on unseen data (20% held-out test set) to measure generalisation.
3. **Inference Phase:** New student profiles (from the quiz) are passed through the learned function to predict their career.

### Types of Supervised Learning

| Type | Description | Application in this Project |
|---|---|---|
| **Classification** | Predicts a discrete category from a set of labels | ✅ Used — predicting one of 5 career categories |
| **Regression** | Predicts a continuous numeric value | ❌ Not used |

### Why Supervised Learning for this Project?

- The problem is inherently a **multi-class classification** task — mapping student features to one of five career categories.
- Labelled training data is available (1,000 samples with assigned career labels).
- The output is categorical (not numeric), making classification algorithms the natural fit.

---

## 1.4 Features and Model Selection

### 1.4.1 Feature Categories

The model uses **22 input features** across 6 categories:

| Category | Features | Count |
|---|---|---|
| **Academic** | grade1, grade2, final_grade, study_time, failures, absences | 6 |
| **Personality (Big Five)** | openness, conscientiousness, extraversion, agreeableness, neuroticism | 5 |
| **Skills** | coding_skill, communication_skill, analytical_skill | 3 |
| **Behaviour** | study_hours, consistency, participation | 3 |
| **Interest** | tech_interest, art_interest, business_interest | 3 |
| **Socioeconomic** | family_income, internet_access (one-hot: 2 cols) | 2 |
| | **Total** | **22** |

### 1.4.2 Feature Selection Techniques

Not all 22 features contribute equally to prediction. Three feature selection methods were evaluated:

| Method | Type | Principle | Features Selected |
|---|---|---|---|
| **SHAP** | Model-based | Game-theory Shapley values measure each feature's contribution to individual predictions | Top 15 |
| **Mutual Information** | Filter-based | Measures statistical dependency between each feature and the target label | Top 15 |
| **Boruta** | Wrapper-based | Compares real feature importance against random shadow features using Random Forest | Auto (all-relevant) |

### 1.4.3 Model Selection Strategy

Model selection was conducted through a **10-configuration comparative study**, systematically varying:

- **Algorithm:** Decision Tree vs. LightGBM
- **Feature set:** All 22, SHAP top-15, MI top-15, Boruta-selected
- **Tuning:** Baseline (default hyperparameters) vs. Optuna-tuned (40 trials)

The best model was selected based on the highest **AUC (Macro OVO)** score on the held-out test set.

---

## 1.5 Evaluation Metrics

Five metrics were used to evaluate model performance:

| Metric | Formula / Description | Why It Matters |
|---|---|---|
| **Accuracy** | `Correct Predictions / Total Predictions` | Overall correctness of the model. |
| **Precision (Macro)** | `TP / (TP + FP)`, averaged across all classes | Measures how many predicted careers are actually correct. Penalises false positives. |
| **Recall (Macro)** | `TP / (TP + FN)`, averaged across all classes | Measures how many actual careers are correctly identified. Penalises missed predictions. |
| **F1 Score (Macro)** | `2 × (Precision × Recall) / (Precision + Recall)` | Harmonic mean of Precision and Recall. Balances both errors. |
| **AUC (Macro OVO)** | Area Under the ROC Curve, One-vs-One | Measures the model's ability to discriminate between all pairs of classes. The primary selection metric. |

### Why AUC as the Primary Metric?

- **Accuracy alone is misleading** when classes are imbalanced (Research Scientist = 56.9% of data). A model predicting only the majority class would achieve ~57% accuracy while being completely useless.
- **AUC measures ranking quality** — whether the model assigns higher probabilities to correct classes — making it robust to class imbalance.
- **Macro averaging** ensures equal weight to all career categories, including rare ones (Data Scientist = 4.5%).

---

## 1.6 Algorithms Used in this Project

Five classification algorithms were considered for this project:

### 1.6.1 Decision Tree

- **Type:** Single tree-based classifier
- **Mechanism:** Recursively splits the feature space using Gini impurity or Entropy to separate classes.
- **Pros:** Highly interpretable, no feature scaling needed, fast inference.
- **Cons:** Prone to overfitting, sensitive to noisy features, unstable with small data changes.

### 1.6.2 Random Forest

- **Type:** Bagging ensemble of multiple Decision Trees
- **Mechanism:** Trains many trees on random subsets of data and features, then aggregates predictions via majority voting.
- **Pros:** Reduces overfitting, handles noisy data, provides feature importance.
- **Cons:** Less interpretable than a single tree, slower training.
- **Role in Project:** Used internally by Boruta for feature selection.

### 1.6.3 LightGBM (Light Gradient Boosting Machine)

- **Type:** Gradient-boosted decision tree ensemble (by Microsoft)
- **Mechanism:** Builds trees sequentially — each new tree corrects the errors of the previous ensemble. Uses leaf-wise growth for efficiency.
- **Pros:** High accuracy, fast training, built-in regularisation, handles imbalanced data well.
- **Cons:** Requires hyperparameter tuning, less interpretable than single trees.

### 1.6.4 Logistic Regression

- **Type:** Linear classifier
- **Mechanism:** Fits a linear decision boundary using the sigmoid/softmax function to estimate class probabilities.
- **Pros:** Simple, fast, interpretable coefficients, works well for linearly separable data.
- **Cons:** Cannot capture non-linear relationships in the feature space — a significant limitation for this dataset.

### 1.6.5 Support Vector Machine (SVM)

- **Type:** Margin-based classifier
- **Mechanism:** Finds the optimal hyperplane that maximises the margin between classes. Can use kernel tricks for non-linear boundaries.
- **Pros:** Effective in high-dimensional spaces, robust to outliers.
- **Cons:** Slow on large datasets, difficult to interpret, sensitive to feature scaling.

### Algorithm Comparison Summary

| Algorithm | Interpretability | Non-linear | Speed | Overfitting Risk | Suitability |
|---|---|---|---|---|---|
| Decision Tree | ★★★★★ | ✅ | Fast | High | ⭐⭐⭐ |
| Random Forest | ★★★ | ✅ | Medium | Low | ⭐⭐⭐⭐ |
| **LightGBM** | ★★ | ✅ | **Fast** | **Low** | **⭐⭐⭐⭐⭐** |
| Logistic Regression | ★★★★ | ❌ | Fast | Low | ⭐⭐ |
| SVM | ★★ | ✅ (kernel) | Slow | Medium | ⭐⭐⭐ |

---

## 1.7 Best Algorithm: LightGBM

### Why LightGBM is the Best Choice for NextStep AI

After benchmarking 10 model configurations, **LightGBM with Optuna hyperparameter tuning (Model M4)** was identified as the best-performing algorithm. The reasons are:

**1. Highest Accuracy and AUC**

LightGBM achieved **87% accuracy** and **0.93 AUC** on the test set — outperforming all Decision Tree variants by ~7% on average.

**2. Ensemble Gradient Boosting**

Unlike a single Decision Tree that makes one pass through the data, LightGBM builds hundreds of weak learners sequentially. Each tree focuses on correcting the mistakes of the previous ensemble, resulting in progressively better predictions.

**3. Robust to Irrelevant Features**

LightGBM's `colsample_bytree` parameter randomly samples feature subsets per tree, naturally ignoring noisy or redundant features. This explains why feature selection had minimal impact on LightGBM (~1% accuracy change) — it already performs implicit feature selection.

**4. Built-in Regularisation**

Parameters like `min_child_samples`, `max_depth`, and `subsample` prevent the model from memorising training data, reducing overfitting — a critical advantage over Decision Trees on a small 1,000-sample dataset.

**5. Efficient Leaf-Wise Growth**

LightGBM grows trees leaf-wise (choosing the leaf with maximum loss reduction) rather than level-wise. This produces deeper, more accurate trees with fewer splits, leading to better performance with lower computational cost.

**6. Native Multi-Class Support**

LightGBM natively handles multi-class classification with `objective='multiclass'` and provides calibrated probability estimates via `predict_proba()` — essential for the confidence scoring and top-3 ranking features of NextStep AI.

### Performance Summary

| Metric | Decision Tree (Best: M3) | LightGBM (Best: M4) | Improvement |
|---|---|---|---|
| Accuracy | 0.8200 | **0.8700** | +6.1% |
| Precision | 0.7000 | **0.7800** | +11.4% |
| Recall | 0.6800 | **0.7600** | +11.8% |
| F1 Score | 0.6900 | **0.7700** | +11.6% |
| AUC | 0.8800 | **0.9300** | +5.7% |

**Conclusion:** LightGBM is the optimal algorithm for NextStep AI due to its superior accuracy, robust handling of small and imbalanced datasets, and native probability estimation — making it the backbone of the deployed career prediction system.
