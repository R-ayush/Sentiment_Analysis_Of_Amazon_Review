##Sentiment Analysis of Amazon Reviews
This project focuses on predicting the sentiment (positive or negative) of Amazon product reviews using machine learning models. The solution is deployed as a real-time sentiment prediction web app built with Streamlit.

#Table of Contents
Overview
Features
Technologies Used
Installation
Usage
Model Performance
Future Enhancements

#Overview
The goal of this project is to analyze Amazon product reviews and predict the sentiment associated with each review. This can help businesses understand customer feedback at scale. The project involves data preprocessing, feature extraction using TF-IDF, model training using machine learning algorithms, and deploying the solution via a Streamlit web app.

#Features
Predicts sentiment (positive/negative) of product reviews.
Machine learning models used: RandomForest, XGBoost, and DecisionTree.
Provides a user-friendly interface for real-time sentiment prediction via Streamlit.
Visualizes sentiment trends and model performance metrics.

#Technologies Used
Python: Core language for building the application.
Pandas, NumPy: For data manipulation and numerical operations.
NLTK: For text preprocessing (tokenization, stopword removal).
Scikit-learn: For machine learning model implementation (RandomForest, DecisionTree) and TF-IDF vectorization.
XGBoost: For gradient boosting model implementation.
Streamlit: For deploying the web application.
Matplotlib, Seaborn: For data visualization.

#Usage
Run the web application using the Streamlit command.
Enter an Amazon product review in the input box provided.
The app will predict whether the review has a positive or negative sentiment.
#Example:
Input: "This product is fantastic! Highly recommended."
Output: Positive Sentiment

#Model Performance
The RandomForestClassifier provided the best performance with an accuracy of 89.5%. The models were evaluated using:

Accuracy: Percentage of correct predictions.
Precision, Recall, F1-Score: To assess the balance between true positives and false positives.
Confusion Matrix: For detailed performance analysis.

#Future Enhancements
Multi-class sentiment analysis: Extend to classify neutral reviews.
Deep learning models: Implement advanced models like LSTM or BERT for better sentiment prediction.
Product-specific analysis: Tailor sentiment predictions for specific product categories.
