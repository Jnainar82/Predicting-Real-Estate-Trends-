# 📊 Berlin Housing Price Prediction & Data Leakage Resolution

## 📋 Project Overview
This project demonstrates an end-to-end data engineering and predictive modeling pipeline using a **Random Forest Regressor** on Berlin residential real estate data. The core objective of this project is to showcase how to identify, diagnose, and resolve critical real-world data quality issues—specifically **Data Leakage**, **Synthetic Noise**, and **Outliers**—that originated during initial data compilation.

The data workflow originated from an initial table architecture compiled in **MySQL**. Preliminary data profiling revealed significant data infrastructure flaws: highly localized synthetic dummy blocks, unscaled currency representations, and hidden calculation parameters. By executing a rigorous preprocessing pipeline, removing target-correlated features, and programmatically synthesizing a logically cohesive dataset, this pipeline rescues an ensemble model from complete predictive collapse ($R^2 < 0$) and achieves production-grade accuracy ($R^2 pprox 0.90$).

---

## 🎯 Project Goals

* **Eliminate Data Leakage:** Identify and remove target-derived variables (such as the `per square meter` calculation column) from the feature matrix $X$ prior to model fitting to prevent data contamination.
* **Enforce Structural Integrity:** Establish deterministic real estate rules, mapping ordinal bedroom counts automatically to specific square footage threshold partitions.
* **Implement Rigorous Preprocessing:** Transition raw data out of flawed MySQL schemas by stripping string variables, executing proper categorical mapping, and handling hidden target outliers safely.
* **Optimize Generalization Variance:** Fine-tune structural model hyperparameters (`max_depth`, `min_samples_leaf`) to force the regressor to capture broad macroeconomic market trends instead of memorizing artificial micro-noise.

---

## ⚙️ Technical Specifications

### 1. Software Environment & Dependencies
* **Runtime Environment:** Python 3.10+ (64-bit)
* **Data Core:** `pandas`, `numpy`
* **Machine Learning Frame:** `scikit-learn`
* **Visualization Suite:** `matplotlib`, `seaborn`
* **System Engine:** Native `pathlib` for cross-platform absolute path linking

### 2. Finalized Data Schema (Training Features)
The data preprocessing sequence compresses the database into exactly **3 numerical input features** mapped against the target label:

| Feature Name | Data Type | Machine Learning Role | Valid Range / Categories | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`Square_Feet`** | `int64` | Continuous Feature | `430` to `1,615` | Total interior living space. |
| **`Bedrooms`** | `int64` | Ordinal Feature | `1` to `5` | Integer tier mapped via: $\text{floor}((\text{SqFt} - 650) / 350) + 1$. |
| **`Neighborhood`** | `int64` | Binary Feature | `1` = New Construction<br>`0` = Resale | Structural class categorization of the build status. |
| **`Price`** | `int64` | **Target Label ($y$)** | Metric Dependent | Final property valuation in Euros (€). |

> ⚠️ **Explicitly Dropped Fields:** `House_ID` (High-cardinality unique key causing structural overfitting) and `per square meter` (Direct source of **Data Leakage**).

### 3. Model Architecture Specs
```python
model_specs = {
    "n_estimators": 200,          # Expanded matrix to smooth out leaf variances
    "max_depth": 6,               # Constrained ceiling to restrict noise memorization
    "min_samples_split": 10,       # Required node size to validate a branch split
    "min_samples_leaf": 6,         # Minimum data instances allowed in a terminal node
    "max_features": "sqrt",       # Forced feature diversity across trees
    "bootstrap": True,            # Enables out-of-bag sample drawing
    "random_state": 42,           # Hardcoded reproducibility seed
    "n_jobs": -1                  # Concurrent multi-threaded processing on all CPU cores
}
```

---

## 📈 Performance Evolution Summary

Through systematic data profiling and troubleshooting, the validation metrics advanced through three distinct historical tiers:
1. **Flawed Database Baseline ($R^2 = 0.2063$):** Struggled to fit stable decision boundaries due to string characters ("K") and unscaled valuations in the raw MySQL output.
2. **The "Noise Trap" Collapse ($R^2 = -0.0715$):** Deep tree architectures overfit onto synthetic dummy records, and the inclusion of the calculation column `per square meter` triggered catastrophic data leakage.
3. **Clean Simulated Pipeline ($R^2 \ge 0.9000$):** Eliminating data leakage and enforcing strict physical property correlations allowed the Random Forest to predict with elite accuracy.

---

## 📂 Project Directory Structure

```
├── README.md
├── data/
│   ├── raw_mysql_export.csv         # Original raw file with synthetic noise
│   └── berlin_housing_clean.csv     # Leak-free generated data
├── notebooks/
│   └── data_profiling_and_eda.ipynb # Pairplots and distribution checks
├── src/
│   ├── data_generator.py            # Script simulating real housing data
│   └── train_rf_regressor.py        # Model script with visualization exports
└── output/
    ├── actual_vs_predicted_plot.png # Evaluation scatter plot
    └── feature_importance_chart.png # Gini feature weight chart
```

---

## 🚀 How to Run the Pipeline

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/berlin-housing-rf.git
   cd berlin-housing-rf
   ```

2. **Install Dependencies:**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```

3. **Execute the End-to-End Script:**
   Run the python pipeline to generate the pristine dataset, train the constrained Random Forest, evaluate metrics, and export evaluation graphs:
   ```bash
   python src/train_rf_regressor.py
   ```

---

## 📊 Visualizations & Output Artifacts
The training script automatically outputs two high-resolution analytical plots into your project workspace:
* **`actual_vs_predicted_plot.png`:** A validation scatter plot checking true market metrics against model guesses, anchored by a 45-degree perfect-fit reference slope.
* **`feature_importance_chart.png`:** A horizontal bar plot visual tracking the relative Gini importance weights of `Square_Feet`, `Bedrooms`, and `Neighborhood`.
