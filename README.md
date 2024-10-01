# Predictive Maintenance using Machine Learning

# Overview

This project focuses on Machine Failure Prediction by analyzing various independent failure modes. The machine failure prediction model helps businesses that rely on machinery to prevent downtime by identifying potential failures before they occur. The project leverages several machine learning models to predict failures based on multiple failure modes.

# Failure Modes

The machine failure is determined based on the following five independent failure modes:

	1.	Tool Wear Failure (TWF): Failure due to tool wear.
	2.	Heat Dissipation Failure (HDF): Failure due to inadequate heat dissipation.
	3.	Power Failure (PWF): Failure due to power loss or fluctuation.
	4.	Overstrain Failure (OSF): Failure caused by excessive strain on the machine.

If any of these failure modes are detected, the machine failure label is set to 1 (i.e., machine failure).

# Objective

The goal of this project is to predict machine failure by analyzing the above failure modes using machine learning models. By implementing predictive analytics, businesses can:

	•	Proactively detect failures before they occur.
	•	Reduce costly downtime due to unexpected machine failures.
	•	Optimize maintenance schedules and improve overall productivity.

## Dataset

The dataset includes the following features:

	•	Failure modes: TWF, HDF, PWF, OSF
	•	Target label: Machine failure (1 for failure, 0 for no failure)

The dataset is used to train and test the machine learning models to predict whether a machine will fail based on the input failure modes.

### Data Format
The training dataset includes the following columns:
- `id`: Unique identifier for each record.
- `Machine failure`: Binary label indicating whether a machine failure occurred (1) or not (0).
- Various features related to machine operations, including product IDs, machine types, and operational metrics.

## Key Features
- **Exploratory Data Analysis (EDA)**: Data cleaning, visualization, and analysis of feature distributions.
- **Feature Engineering**: Normalization of numerical features and one-hot encoding of categorical features.
- **Modeling**: Implementation of multiple machine learning algorithms, including:
  - Random Forest Classifier
  - Decision Tree Classifier
  - Support Vector Classifier (SVC)
  - CatBoost Classifier
- **Model Evaluation**: Evaluation of models using accuracy, F1 score, confusion matrices, and ROC curves.
- **Hyperparameter Tuning**: Use of RandomizedSearchCV to optimize model hyperparameters.

