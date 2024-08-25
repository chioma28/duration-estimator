from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import pickle

# Load historical data
data = pd.read_csv('tasks_history.csv')


# Convert duration to numeric (handle any potential non-numeric values)
data['duration'] = pd.to_numeric(data['duration'], errors='coerce')

# Initialize SentenceTransformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings for task titles
embeddings = embedding_model.encode(data['title'].tolist(), convert_to_tensor=True)

# Function to get the most similar historical task
def get_most_similar_task(title, embeddings, data):
    # Encode the new title using the SentenceTransformer model
    title_embedding = embedding_model.encode([title], convert_to_tensor=True)
    
    # Calculate cosine similarities
    similarities = util.pytorch_cos_sim(title_embedding, embeddings).numpy()
    
    # Print all similarity scores along with their corresponding titles and durations
    print("All similarity scores:")
    for i, score in enumerate(similarities[0]):
        print(f"Index: {i}, Title: {data['title'].iloc[i]}, Duration: {data['duration'].iloc[i]}, Similarity: {score}")
    
    # Find index of the most similar task
    most_similar_index = np.argmax(similarities)
    print("index: ", most_similar_index)
    
    # Retrieve the most similar task's title and duration
    most_similar_title = data['title'].iloc[most_similar_index]
    similar_task_duration = data['duration'].iloc[most_similar_index]
    print("index duration: ", similar_task_duration)
    
    return most_similar_title, similar_task_duration


# Convert dueDate to datetime and then to numerical value (days since start of the year)
data['dueDate'] = pd.to_datetime(data['dueDate'], errors='coerce')
data['daysUntilDue'] = (data['dueDate'] - pd.Timestamp.now()).dt.days

# Features and target variable
X = data[['duration', 'daysUntilDue']]
y = data['duration']

# Ensure all features are numeric
X = X.apply(pd.to_numeric, errors='coerce')
y = y.apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values in features or target variable
X.dropna(inplace=True)
y = y[X.index]  # Ensure target variable matches features index

# Train-test split with a sufficient test size
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
# models = {
#     'Decision Tree': DecisionTreeRegressor(random_state=42),
#     'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
#     'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
#     'SVR': SVR()
# }
model = DecisionTreeRegressor(random_state=42)
# Train and evaluate each model
# for name, model in models.items():
model.fit(X_train, y_train)

# Save the model to a file
with open('decision_tree_similarity_model.pkl', 'wb') as f:
    pickle.dump(model, f)

np.save('embeddings.npy', embeddings)

    
y_pred = model.predict(X_test)
    
    # # Evaluation
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

# Function to estimate duration of a new task based on semantic similarity
def estimate_duration(title, model):
    try:
        # Get the most similar historical task
        most_similar_title, similar_task_duration = get_most_similar_task(title, embeddings, data)
        
        # Handle potential NaN in duration
        if pd.isna(similar_task_duration) or similar_task_duration <= 0:
            print("Warning: Found NaN or non-positive value in similar task duration, setting to default value of 30 minutes.")
            similar_task_duration = 30.0
        else:
            similar_task_duration = similar_task_duration
        
        # Print debug information
        print(f"Most similar task title: {most_similar_title}")
        print(f"Duration of most similar task: {similar_task_duration:.2f} minutes")
        
        # Check if the task has previous history
        historical_data = data[data['title'] == title]
        
        if not historical_data.empty:
            # If there is previous history, use the most recent duration
            most_recent_duration = historical_data['duration'].max()
            print(f"Most recent duration for the task '{title}' from historical data: {most_recent_duration:.2f} minutes")
            return most_recent_duration
        
        # Use the model with the best performance from training
        chosen_model = model
        
        # Create feature set for prediction
        new_task_features = pd.DataFrame({
            'duration': [similar_task_duration],
            'daysUntilDue': [data['daysUntilDue'].mean()]
        })
        
        # Ensure the features are numeric
        new_task_features = new_task_features.astype(float)
        
        # Predict the duration
        estimated_duration = chosen_model.predict(new_task_features)
        return estimated_duration[0]
    
    except Exception as e:
        print(f"An error occurred: {e}")
        # Return a default duration in case of an error
        return 30.0


# Example usage
title = "task"
best_model = 'Decision Tree' 
estimated_duration = estimate_duration(title, model)
print(f"The estimated duration for the {title} task is {estimated_duration:.2f} minutes")
