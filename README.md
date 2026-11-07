# Online Shoppers Purchasing Intention Prediction

**BITS WILP | M.Tech Artificial Intelligence and Machine Learning | Machine Learning Assignment 2 | 2026AJ05675**

## 1. Problem Statement

The objective of this project is to develop and evaluate machine learning classification models to predict whether an online shopping visitor will generate a purchase (Revenue) based on their browsing behaviour and session characteristics.

The project implements five classification algorithms:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

The models are evaluated using multiple classification metrics to understand their predictive performance, particularly considering the class imbalance in the dataset.

Additionally, hyperparameter optimization is performed for Decision Tree and SVM classifiers to investigate opportunities for improving predictive performance.

A Streamlit application is developed to allow users to upload labelled datasets, select a trained model and evaluate its classification performance without retraining.

## 2. Dataset Description

**Dataset:** Online Shoppers Purchasing Intention Dataset  
**Source:** UCI Machine Learning Repository  
**Dataset ID:** 468

The dataset contains information about online shopping sessions and whether a visitor eventually completed a purchase.

| Dataset characteristic | Description |
|---|---|
| Total instances | 12,330 |
| Total input features | 17 |
| Target variable | Revenue |
| Classification type | Binary classification |
| Training samples | 9,864 |
| Testing samples | 2,466 |
| Training-test split | 80:20 |
| Random state | 42 |
| Stratification | Applied |

### Target variable distribution

| Class | Description | Instances | Percentage |
|---|---|---:|---:|
| 0 | No Purchase | 10,422 | 84.53% |
| 1 | Purchase | 1,908 | 15.47% |

The dataset is imbalanced, with non-purchasing sessions representing approximately 84.53% of the observations.

Therefore, accuracy alone is insufficient to evaluate model performance. Precision, Recall, F1-score, AUC and Matthews Correlation Coefficient (MCC) are also considered.

### Feature categories

The 17 input features are grouped into the following categories:

- **Continuous features (10):** Administrative, Administrative_Duration, Informational, Informational_Duration, ProductRelated, ProductRelated_Duration, BounceRates, ExitRates, PageValues and SpecialDay.
- **Categorical features (6):** Month, VisitorType, OperatingSystems, Browser, Region and TrafficType.
- **Boolean feature (1):** Weekend.

## 3. Data Preprocessing

The following preprocessing techniques are implemented:

- Numerical features are standardized using StandardScaler.
- Categorical features are transformed using OneHotEncoder.
- Boolean features are passed through without transformation.
- ColumnTransformer is used to combine the preprocessing operations.
- Missing or previously unseen categorical values are handled through the preprocessing pipeline configuration.

The dataset is divided into training and testing subsets using an 80:20 stratified split.

All preprocessing operations are incorporated into the model pipelines to ensure consistent transformations during training and application-based evaluation.

## 4. Model Implementation and Experimentation

### 4.1 Baseline Classification Models

All five required classification algorithms are initially implemented using their baseline configurations.

| Algorithm | Implementation |
|---|---|
| Logistic Regression | LogisticRegression |
| Decision Tree | DecisionTreeClassifier |
| KNN | KNeighborsClassifier |
| Gaussian Naive Bayes | GaussianNB |
| SVM | SVC with RBF kernel |

### 4.2 SVM Kernel Comparison

Three SVM kernel configurations are evaluated:

- Linear Kernel
- Radial Basis Function (RBF) Kernel
- Polynomial Kernel

The RBF kernel demonstrates the best overall balance among the three configurations based on F1-score and MCC.

### 4.3 SVM Hyperparameter Optimization

RandomizedSearchCV is used with stratified 3-fold cross-validation to identify suitable SVM hyperparameters.

The following parameters are explored:

- C: Regularization parameter.
- Gamma: Kernel coefficient for the RBF kernel.
- Class weight: Comparison between default and balanced class weights.

F1-score is selected as the optimization metric because of the class imbalance.

The optimized SVM improves recall, F1-score, AUC and MCC compared with the baseline SVM.

### 4.4 Decision Tree Hyperparameter Optimization

RandomizedSearchCV with stratified 3-fold cross-validation is also used to optimize the Decision Tree classifier.

The following parameters are explored:

- Maximum tree depth.
- Minimum samples required for splitting.
- Minimum samples required at leaf nodes.
- Class weights.

The tuned Decision Tree demonstrates improved performance across most evaluation metrics compared with its baseline configuration.

### 4.5 SVM Decision Threshold Analysis

An additional experiment investigates the effect of changing the decision threshold of the tuned SVM.

A validation subset is created from the training data, and different thresholds are evaluated using precision, recall and F1-score.

A threshold of 0.30 achieves the best validation F1-score in the tested threshold range. Its performance is subsequently evaluated on the independent test dataset.

The experiment demonstrates that changing the classification threshold can influence the precision-recall trade-off.

## 5. Model Performance Comparison

All reported metrics are calculated using the held-out test dataset.

| Model | Accuracy | AUC | Precision | Recall | F1-score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 88.12% | 88.30% | 74.05% | 35.86% | 48.32% | 46.10% |
| Decision Tree (Baseline) | 85.93% | 73.40% | 54.52% | 55.24% | 54.88% | 46.54% |
| Decision Tree (Tuned) | **88.89%** | 87.83% | 67.65% | 54.19% | 60.17% | 54.26% |
| KNN | 87.15% | 75.72% | 64.19% | 38.48% | 48.12% | 43.07% |
| Gaussian Naive Bayes | 27.33% | 73.28% | 17.27% | 97.38% | 29.34% | 12.92% |
| SVM (Baseline RBF) | 88.73% | 88.17% | 70.63% | 46.60% | 56.15% | 51.42% |
| SVM (Tuned RBF) | 87.10% | 90.28% | 56.37% | **74.08%** | **64.03%** | **57.13%** |
| SVM (Threshold 0.30) | 87.92% | **90.43%** | 59.68% | 67.80% | 63.48% | 56.44% |

**Note:** The threshold SVM experiment uses a model trained on a reduced training subset. Its results should therefore be interpreted separately from the models trained on the full training dataset.

## 6. Model Observations

### Logistic Regression

- Achieves the highest precision among the evaluated configurations at 74.05%.
- Demonstrates good AUC of 88.30%.
- Has relatively low recall of 35.86%.
- Provides a more conservative classification approach, identifying fewer actual purchasing sessions.

### Decision Tree

- The baseline Decision Tree achieves an accuracy of 85.93%.
- Hyperparameter optimization improves accuracy to 88.89%.
- AUC increases from 73.40% to 87.83%.
- Precision improves from 54.52% to 67.65%.
- The tuned model provides better overall classification performance than its baseline configuration.

### K-Nearest Neighbors

- Achieves an accuracy of 87.15%.
- Demonstrates moderate precision but relatively low recall.
- Its AUC and F1-score indicate scope for further improvement.

### Gaussian Naive Bayes

- Achieves very high recall of 97.38%.
- However, precision is only 17.27%.
- The model produces a large number of false positive predictions.
- Its low MCC and F1-score indicate limited overall classification effectiveness for this dataset.

### Support Vector Machine

- The baseline RBF SVM achieves an accuracy of 88.73%.
- Kernel comparison indicates that RBF provides the best overall balance among the tested kernels.
- Hyperparameter optimization improves recall from 46.60% to 74.08%.
- F1-score improves from 56.15% to 64.03%.
- MCC improves from 51.42% to 57.13%.
- The tuned SVM achieves the highest F1-score and MCC among the evaluated configurations.

## 7. Overall Model Selection

Based on the experimental results, the **Tuned RBF SVM** is selected as the preferred model for this classification problem.

The primary reasons are:

- Highest F1-score of 64.03%.
- Highest MCC of 57.13%.
- Strong recall of 74.08%.
- AUC of 90.28%.

The tuned Decision Tree is also retained as an improved alternative, achieving the highest accuracy of 88.89%.

The final application uses the following five models:

1. Logistic Regression – Baseline
2. Decision Tree – Tuned
3. KNN – Baseline
4. Gaussian Naive Bayes – Baseline
5. SVM – Tuned RBF

The selection is based on the balance between precision, recall and overall classification quality rather than accuracy alone.

## 8. Streamlit Application

A Streamlit application is developed to demonstrate the evaluation of the trained classification models.

### Application features

- Upload labelled CSV datasets.
- Select one of the five trained classification models.
- Evaluate uploaded data without retraining.
- Display Accuracy, AUC, Precision, Recall, F1-score and MCC.
- Visualize the confusion matrix.
- Display the classification report.
- Download prediction results as a CSV file.

### Live Application

**Streamlit Community Cloud:** To be added after deployment.

### Running the application locally

Clone the repository:

```bash
git clone https://github.com/2026AJ05675-LEO/online-shoppers-purchase-prediction.git
```

Navigate to the project directory:

```bash
cd online-shoppers-purchase-prediction
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

The uploaded CSV must contain the same 17 input features used during training and the target column `Revenue`, containing binary values 0 and 1.

## 9. Repository Structure

```text
online-shoppers-purchase-prediction/
│
├── model/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── gaussian_naive_bayes.pkl
│   └── svm.pkl
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── final_model_comparison.csv
└── online-shoppers-purchase-prediction.ipynb
```

## 10. Future Scope and Potential Improvements

The current implementation establishes baseline performance for all five required classifiers and performs additional optimization experiments on SVM and Decision Tree.

Further improvements can be explored through:

1. **Hyperparameter Optimization:** Extend systematic tuning to Logistic Regression, KNN and Gaussian Naive Bayes.
2. **Feature Selection:** Apply RFE, SelectKBest and feature importance-based selection to identify relevant predictors.
3. **Feature Engineering:** Develop additional features based on browsing behaviour and customer session characteristics.
4. **Class Imbalance Handling:** Experiment with SMOTE, undersampling and alternative class weighting strategies.
5. **Decision Threshold Optimization:** Investigate different operating thresholds based on business requirements.
6. **Ensemble Learning:** Explore Random Forest, Gradient Boosting and other ensemble algorithms.

Due to the assignment timeline and available time constraints, these extensions are proposed as future work and have not been implemented or evaluated in the current project.

## 11. Conclusion

This project demonstrates the implementation and evaluation of five machine learning classification algorithms for predicting online shopping purchase intention.

The experiments highlight the importance of model evaluation using multiple metrics, particularly when working with imbalanced classification datasets.

Hyperparameter optimization significantly improves the performance of the Decision Tree and SVM classifiers. Among the evaluated configurations, the tuned RBF SVM provides the strongest balance of precision, recall and overall classification quality.

The Streamlit application provides an interactive interface for evaluating the trained models on new labelled datasets without requiring model retraining.

**GitHub Repository:**  
https://github.com/2026AJ05675-LEO/online-shoppers-purchase-prediction

---

**BITS WILP – M.Tech AIML | Machine Learning Assignment 2- 2026AJ05675**
