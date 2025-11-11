# AI4I Predictive Maintenance Analysis

This project explores predictive maintenance using two datasets — the **AI4I 2020 Predictive Maintenance Dataset** and an enhanced irregular version, **AI4I-PMDI**, which introduces realistic industrial irregularities and noise to simulate real-world sensor behavior. The goal is to understand how models trained on clean data perform when exposed to more complex, noisy data and vice versa.

---

## Project Overview

Predictive maintenance leverages sensor and process data to forecast equipment failures before they occur. This project investigates how models behave under two conditions:
- **Clean environment:** stable, well-calibrated sensor data.
- **Unclean environment:** irregular or degraded data including sensor drift, noise, and missing values.

The study compares model performance and generalization when trained and tested across these datasets.

---

## Datasets

### AI4I 2020 Dataset (Clean)
A well-known dataset from the UCI Machine Learning Repository:
- 10,000 records
- Stable process conditions with no missing data
- Machine states categorized by type (L, M, H)
- Labels for machine failure and operational conditions

### AI4I-PMDI Dataset (Irregular)
Derived from the research paper *“AI4I-PMDI: Predictive maintenance datasets with complex industrial settings’ irregularities”*, this dataset extends AI4I 2020 by introducing:
- Sensor irregularities such as drift and noise
- Additional **system** and **control** parameters
- Missing or inconsistent readings
- Complex dependencies between variables

This dataset more accurately represents conditions found in real industrial systems and provides a test of model robustness under imperfect data.

---

## Methodology

1. **Data Preparation**
   - Both datasets are preprocessed with the same feature engineering pipeline.
   - New features include:
     - Temperature difference (`process_temperature - air_temperature`)
     - Angular velocity (`omega_rad_s`)
     - Power (`torque * omega_rad_s`)
     - Wear-and-torque interaction
   - Categorical features (machine type and control) are one-hot encoded for model input consistency.

2. **Model Training**
   - Trained and evaluated models:
     - **Random Forest**
     - **Gradient Boosting**
     - **XGBoost**
   - Each model is trained separately on the clean and irregular datasets to measure in-domain performance.

3. **Cross-Dataset Comparison**
   - Evaluates model generalization across datasets using paired product IDs:
     - **Train on Clean → Test on Unclean**
     - **Train on Unclean → Test on Clean**
   - ROC-AUC is used to measure how well each model distinguishes failures between environments without data leakage.

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/<yourusername>/ai4i-predictive-maintenance.git
cd ai4i-predictive-maintenance
```
### 2 Install dependencies
pip install -r requirements.txt

### 3 Run main script
python main.py

## References

**AI4I 2020 Predictive Maintenance Dataset**  
Bischl, B., et al. *AI4I 2020 Predictive Maintenance Dataset.* UCI Machine Learning Repository, 2020.  
[https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset)

**AI4I-PMDI: Predictive Maintenance Datasets with Complex Industrial Settings’ Irregularities**  
Autran, J.-V., Kuhn, V., Diguet, J.-P., Dubois, M., & Buche, C. (2024). *AI4I-PMDI: Predictive Maintenance Datasets with Complex Industrial Settings’ Irregularities.* *Procedia Computer Science, 234*, 546–553.  
[https://doi.org/10.1016/j.procs.2024.09.546](https://doi.org/10.1016/j.procs.2024.09.546)
