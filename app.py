from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.tree import DecisionTreeRegressor
import pickle
from flask_cors import CORS
import os
import nltk
nltk.data.path.append('/nltk_data')


app = Flask(__name__)
CORS(app)

# Global variables
model = None
embedding_model = None
embeddings = None
data = None

def load_resources():
    global model, embedding_model, embeddings, data
    # Load your model and embeddings
    model = pickle.load(open('decision_tree_similarity_model.pkl', 'rb'))
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = np.load('embeddings.npy')
    data = pd.read_csv('tasks_history.csv')
    data['dueDate'] = pd.to_datetime(data['dueDate'], errors='coerce')
    data['daysUntilDue'] = (data['dueDate'] - pd.Timestamp.now()).dt.days
    data['daysUntilDue'] = pd.to_numeric(data['daysUntilDue'], errors='coerce')

def estimate_duration(title):
    title_embedding = embedding_model.encode([title], convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(title_embedding, embeddings).numpy()
    most_similar_index = np.argmax(similarities)
    similar_task_duration = data['duration'].iloc[most_similar_index]

    # Create feature set for prediction
    new_task_features = pd.DataFrame({
        'duration': [similar_task_duration],
        'daysUntilDue': [data['daysUntilDue'].mean()]  # Ensure this is numeric
    })
    
    # Ensure the features are numeric
    new_task_features = new_task_features.apply(pd.to_numeric, errors='coerce')
    
    # Debugging output
    print("New task features for prediction:")
    print(new_task_features)

    # Predict the duration
    estimated_duration = model.predict(new_task_features)
    return estimated_duration[0]

@app.route('/estimate-duration', methods=['POST'])
def estimate():
    try:
        title = request.json.get('title')
        estimated_duration = estimate_duration(title)
        
        # Save new data if the title was not in the history
        global data
        historical_data = data[data['title'] == title]
        if historical_data.empty:
            new_entry = {
                'title': title,
                'duration': estimated_duration,
                'dueDate': pd.Timestamp.now(),  # Or some other default due date
                'daysUntilDue': (pd.Timestamp.now() - pd.Timestamp.now()).days
            }
            new_entry_df = pd.DataFrame([new_entry])
            data = pd.concat([data, new_entry_df], ignore_index=True)
            data.to_csv('tasks_history.csv', index=False)  # Save updated data to CSV
            print(f"Added new task '{title}' with estimated duration {estimated_duration:.2f} minutes to history.")
        
        return jsonify({'estimated_duration': estimated_duration})
    
    except Exception as e:
        print(f"An error occurred during estimation: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/test', methods=['GET'])
def test():
    return "Test endpoint is working!"

if __name__ == '__main__':
    load_resources()  # Load resources before starting the app
    port = int(os.environ.get('PORT', 5300))
    app.run(debug=True, port=port)

