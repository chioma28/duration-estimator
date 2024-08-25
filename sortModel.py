import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib


# Load your data
data = pd.read_csv('tasks.csv')  

#####priority formula
# Priority Score=(Priority Numeric Value×Duration)+Points

# Preprocess your data
le_category = LabelEncoder()
le_priority = LabelEncoder()
data['category'] = le_category.fit_transform(data['category'])
data['priority'] = le_priority.fit_transform(data['priority'])

X = data[['category', 'priority', 'duration']]  # Features
y = data['priorityScore']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
models = {
    'GBM': GradientBoostingClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f'{name} Accuracy: {accuracy_score(y_test, y_pred)}')
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score
# from textblob import TextBlob
# import joblib

# # Load your data
# data = pd.read_csv('tasks.csv')

# # Sentiment Analysis function
# def get_sentiment_score(text):
#     return TextBlob(text).sentiment.polarity

# # Apply sentiment analysis to task titles
# data['sentimentScore'] = data['title'].apply(get_sentiment_score)

# # Preprocess your data
# le_category = LabelEncoder()
# le_priority = LabelEncoder()
# data['category'] = le_category.fit_transform(data['category'])
# data['priority'] = le_priority.fit_transform(data['priority'])

# # Feature engineering: convert dueDate to numerical format
# data['dueDate'] = pd.to_datetime(data['dueDate']).astype(int) / 10**9  # Convert to timestamp

# X = data[['category', 'priority', 'duration', 'dueDate', 'sentimentScore']]  # Features
# y = data['priorityScore']  # Target variable

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train models
# # models = {
# #     'GBM': GradientBoostingClassifier(),
# #     'Random Forest': RandomForestClassifier(),
# #     'SVM': SVC()
# # }
# model = GradientBoostingClassifier()
# model.fit(X_train, y_train)
# print(model)
# print("done")

joblib.dump(model, 'your_model3.pkl') #Save model