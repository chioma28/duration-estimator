# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from textblob import TextBlob
# import numpy as np

# # Load historical data
# data = pd.read_csv('tasks_history.csv')

# # Sentiment Analysis
# data['sentimentScore'] = data['title'].apply(lambda x: TextBlob(x).sentiment.polarity)

# # Convert dueDate to datetime and then to numerical value (days since start of the year)
# data['dueDate'] = pd.to_datetime(data['dueDate'], errors='coerce')
# data['daysUntilDue'] = (data['dueDate'] - pd.to_datetime('2024-01-01')).dt.days

# # Drop rows with NaT in 'daysUntilDue' to handle any issues with date conversion
# data.dropna(subset=['daysUntilDue'], inplace=True)

# # Features and target variable
# X = data[['sentimentScore', 'duration', 'daysUntilDue']]
# y = data['duration']

# # Ensure all features are numeric
# X = X.apply(pd.to_numeric, errors='coerce')
# y = y.apply(pd.to_numeric, errors='coerce')

# # Drop rows with NaN values in features or target variable
# X.dropna(inplace=True)
# y = y[X.index]  # Ensure target variable matches features index

# # Train-test split with a sufficient test size
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train Linear Regression model
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Predictions
# y_pred = model.predict(X_test)

# # Evaluation
# if len(y_test) > 1:
#     mae = mean_absolute_error(y_test, y_pred)
#     mse = mean_squared_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)
#     print(f'Mean Absolute Error (MAE): {mae:.2f}')
#     print(f'Mean Squared Error (MSE): {mse:.2f}')
#     print(f'R-squared (R²): {r2:.2f}')
# else:
#     print("Not enough test samples to calculate R² score.")

# # Function to estimate duration of a new task
# def estimate_duration(title, completionTime, dueDate):
#     sentiment_score = TextBlob(title).sentiment.polarity
#     dueDate = pd.to_datetime(dueDate, errors='coerce')
#     days_until_due = (dueDate - pd.to_datetime('2024-01-01')).days
#     new_task_features = np.array([[sentiment_score, completionTime, days_until_due]])
#     estimated_duration = model.predict(new_task_features)
#     return estimated_duration[0]

# # Example usage
# title = "Attend Meeting"
# completion_time = 3  # Example completion time
# due_date = '2024-08-30'
# estimated_duration = estimate_duration(title, completion_time, due_date)
# print(f"Estimated Duration: {estimated_duration:.2f} minutes")


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load historical data
data = pd.read_csv('tasks_history.csv')

# Create TF-IDF vectors for task titles
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data['title'])

# Function to get similarity score


def get_similarity_score(new_title):
    new_title_vec = vectorizer.transform([new_title])
    similarities = cosine_similarity(new_title_vec, tfidf_matrix)
    return similarities.max()


# Sentiment Analysis
data['sentimentScore'] = data['title'].apply(
    lambda x: TextBlob(x).sentiment.polarity)

# Convert dueDate to datetime and then to numerical value (days since start of the year)
data['dueDate'] = pd.to_datetime(data['dueDate'], errors='coerce')
data['daysUntilDue'] = (data['dueDate'] - pd.to_datetime('2024-01-01')).dt.days

# Drop rows with NaT in 'daysUntilDue' to handle any issues with date conversion
data.dropna(subset=['daysUntilDue'], inplace=True)

# Features and target variable
X = data[['sentimentScore', 'duration', 'daysUntilDue']]
y = data['duration']

# Ensure all features are numeric
X = X.apply(pd.to_numeric, errors='coerce')
y = y.apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values in features or target variable
X.dropna(inplace=True)
y = y[X.index]  # Ensure target variable matches features index

# Train-test split with a sufficient test size
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Check if there are enough samples
if len(X_train) < 2 or len(X_test) < 2:
    raise ValueError(
        "Not enough samples to train and test models. Consider increasing the dataset size.")

# Define models
models = {
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'SVR': SVR()
}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluation
    # if len(y_test) > 1:
    #     mae = mean_absolute_error(y_test, y_pred)
    #     mse = mean_squared_error(y_test, y_pred)
    #     r2 = r2_score(y_test, y_pred)
    #     print(f'{name}:')
    #     print(f'  Mean Absolute Error (MAE): {mae:.2f}')
    #     print(f'  Mean Squared Error (MSE): {mse:.2f}')
    #     print(f'  R-squared (R²): {r2:.2f}')
    # else:
    #     print(f'{name}: Not enough test samples to calculate R² score.')

# Function to estimate duration of a new task


def estimate_duration(title, completionTime, dueDate, model_name='Decision Tree'):
    sentiment_score = TextBlob(title).sentiment.polarity
    similarity_score = get_similarity_score(title)
    print("Sentiment score", sentiment_score)
    print("Similarity score", similarity_score)

    dueDate = pd.to_datetime(dueDate, errors='coerce')
    days_until_due = (dueDate - pd.to_datetime('2024-01-01')).days
    new_task_features = pd.DataFrame({
        'sentimentScore': [sentiment_score],
        'duration': [completionTime],
        'daysUntilDue': [days_until_due]
    })

    # Use the model with the best performance from training
    model = models[model_name]
    estimated_duration = model.predict(new_task_features)
    return estimated_duration[0]


# Example usage
title = "Exercise"
completion_time = 2  # Example completion time
due_date = '2024-08-25'
best_model = 'Decision Tree'  # Replace with the model name that performed the best
estimated_duration = estimate_duration(
    title, completion_time, due_date, model_name=best_model)
print(f"Estimated Duration: {estimated_duration:.2f} minutes")
