# Movie Revenue Prediction

[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-360/) [![sk-learn](https://img.shields.io/badge/scikit-learn-grey.svg?logo=scikit-learn)](https://scikit-learn.org/stable/whats_new.html)

[arXiv Paper](http://arxiv.org/abs/2405.11651) | [Academia Paper](https://www.academia.edu/119091410/Movie_Revenue_Prediction) | [Streamlit WebApp](https://movie-revenue-prediction.streamlit.app)

A machine learning project to predict movie revenue using CNN, DNN, and XGBoost, with an interactive Streamlit app for filmmakers and data enthusiasts.

## Overview

This project, developed as part of the graduation internship at the Vietnam Aviation Academy (Faculty of Information Technology), focuses on building and evaluating machine learning models to predict movie box office revenue. The project leverages three models—**Convolutional Neural Network (CNN)**, **Deep Neural Network (DNN)**, and **Extreme Gradient Boosting (XGBoost)**—to analyze various movie attributes and forecast revenue.

## Objectives

- Develop a predictive system for movie revenue using features such as budget, genre, cast, release date, IMDb ratings, and more.
- Compare the performance of **CNN**, **DNN**, and **XGBoost** models to identify the most effective approach.
- Create an interactive web application using **Streamlit** to allow users to input movie details and receive revenue predictions.

## Dataset

- **Source**: "The Movies Industry Dataset" from IMDb, hosted on Kaggle, containing 7,669 movies (1980–2020).
  - The datasets have been taken from [Movie Industry](https://www.kaggle.com/datasets/danielgrijalvas/movies) dataset.
- **Features**: Includes movie name, genre, MPAA rating, release year, IMDb score, votes, director, writer, star, country, budget, company, and runtime.
- **Preprocessing**:
  - Handled 2,247 missing values, reducing the dataset to 5,422 records by removing entries with missing budget or revenue.
  - Applied **Label Encoding** for categorical features, **Standard Scaling** for numerical features, and created new features (e.g., budget-year ratio, vote-score ratio).

## Models

Three machine learning models were implemented and evaluated:

1. **Convolutional Neural Network (CNN)**:
   - Adapted for sequential data using 1D convolutions.
   - Architecture: Conv1D layers, MaxPooling, Flatten, Dense layers with ReLU activation.

2. **Deep Neural Network (DNN)**:
   - Built using **Keras** with **TensorFlow**, featuring multiple dense layers, Batch Normalization, LeakyReLU, and Dropout.

3. **XGBoost (Extreme Gradient Boosting)**:
   - Utilized for its strength in handling tabular data, with boosting and regularization.
   - Key components: XGBRegressor, DMatrix, cross-validation, and early stopping.

## Evaluation

- **Metrics**: R² Score, Mean Squared Error (MSE), Mean Squared Logarithmic Error (MSLE), and Mean Absolute Percentage Error (MAPE).
- **Findings**:
  - **XGBoost** outperformed **CNN** and **DNN**, with the highest R² and lowest error metrics.
  - Budget and votes were the most influential features for revenue prediction.
  - **CNN** and **DNN** showed limitations in handling tabular data compared to **XGBoost**.
- **Visualizations**: Loss curves and actual vs. predicted value plots were used to assess model convergence and prediction accuracy.

The table below summarizes the performance of the three models on the test set:

| Model   | R²     | MSE    | MSLE   |
|---------|--------|--------|--------|
| CNN     | 0.6551 | 1.3010 | 0.0049 |
| DNN     | 0.5857 | 1.5629 | 0.0060 |
| XGBoost | 0.7402 | 1.0193 | 0.0041 |

![Model Performance Table]

- **Analysis**:
  - **XGBoost** achieved the highest R² (0.7402) and the lowest MSE (1.0193) and MSLE (0.0041), making it the best performer.
  - **CNN** performed better than **DNN** with an R² of 0.6551 compared to **DNN**'s 0.5857, but both lagged behind **XGBoost**.
  - The high MAPE values across all models (e.g., 199.18% for **XGBoost**) indicate room for improvement, possibly due to outliers or data distribution issues.

## Application

- **Streamlit Web App**:
  - Built an interactive interface allowing users to input movie details (e.g., release date, budget, genre).
  - Displays predicted revenue for selected models (**CNN**, **DNN**, **XGBoost**) and a comparative table.
  - Includes estimated revenue ranges for practical insights.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/NgocMinh126/Movie-Revenue-Prediction.git
   cd Movie-Revenue-Prediction

   python3 -m venv ./env
   source env/bin/activate

   pip install -r requirements.txt
   ```
### Using the CLI

1. Navigate to the project directory.
2. Run the CLI:
   ```bash
   python main.py
   ```
3. Follow the prompts to input the movie features and choose the prediction model.

## Streamlit Web Interface

Additionally a web interface is also developed using Streamlit to allow users to input movie features and get revenue predictions.

### Running the Web Interface

1. Navigate to the project directory.
2. Run the Web Interface:
   ```bash
   streamlit run streamlit_app.py
   ```
3. Follow the prompts to input the movie features and choose the prediction model.
   
### Running the Model
1. Run compare models:
   ```bash
   python models/ compare_models.py
   ```
2. Example Run DNN model:
   ```bash
   python models/ dnn.py
   ```
## Usage
1. Open the Streamlit app in your browser.
2. Enter movie details in the provided fields.
3. Select a model (CNN, DNN, or XGBoost) to generate revenue predictions.
4. View results, including predicted revenue and model comparisons.

## Contributors
  Ngọc Minh (Group Leader) 
  \
  Lan Anh (Member)
  \
  Thanh Minh (Member)
  
## Acknowledgments
Institution: Vietnam Aviation Academy, Faculty of Information Technology
