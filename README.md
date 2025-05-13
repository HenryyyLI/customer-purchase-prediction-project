# 📈 Customer Purchase Prediction Project

A robust machine learning pipeline for predicting customer purchase intent based on historical transaction data — featuring custom KNN imputation for missing values, class imbalance handling, and advanced models like AdaBoost and Neural Network. Designed to uncover customer behavior patterns and support data-driven marketing decisions that identify high-value customers and optimize targeting strategies.

## 🔍 Key Features

- **🧠 ML Pipeline for Purchase Prediction**

  An end-to-end machine learning solution that spans data cleaning, custom KNN-based missing value imputation, feature engineering, model training, and performance evaluation — all tailored to predict customer purchase intent with high accuracy and real-world applicability.

- **🔄 Balanced Dataset Handling**

  Implemented threshold optimization via ROC curve analysis, selecting the point that maximizes the TPR–FPR gap to fine-tune decision boundaries. Combined with class weight tuning and resampling strategies (oversampling/undersampling), this approach significantly improves minority class performance — achieving over a 7% F1 score boost for underrepresented outcomes.

- **🧩 Custom KNN-Based Missing Value Imputation**

  Designed a tailored KNN imputation strategy to fill missing values by leveraging the most similar records based on feature distances. This preserves underlying data structure and reduces noise introduced by traditional imputation methods, improving downstream model stability.

- **🤖 Neural Network Classifier Integration**

  Integrated a lightweight Neural Network (NN) as part of the modeling suite to capture nonlinear relationships in customer behavior. Tuned with dropout and batch normalization to improve generalization, the NN model offers a strong alternative to tree-based methods in scenarios with complex feature interactions.


## 📁 Project Structure

```bash
customer-purchase-prediction-project/
├── data_imputing.py                  # Custom KNN imputation script for handling missing values
├── customer_purchase_prediction.ipynb # Jupyter notebook for EDA, modeling, and evaluation
├── train_dataset.csv                 # Training dataset containing labeled customer behavior data
├── test_dataset.csv                  # Test dataset for evaluating model performance
├── README.md                         # Project documentation with overview, setup, and usage
└── .gitignore                        # Specifies files and folders to be ignored by Git
```

## 🛠 Tech Stack

- **Language**: Python

- **Data Manipulation**: Pandas, NumPy

- **Data Visualization**: Matplotlib, Seaborn

- **Machine Learning Models**: Scikit-learn (Logistic Regression, Random Forest, KNN, etc.)

- **Neural Network Framework**: Torch

