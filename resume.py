import os
import zipfile

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("UpdatedResumeDataSet.csv")

# Display the first few rows of the dataset
df.head()

# Display the last few rows of the dataset
df.tail()

# Get the shape of the dataset
df.shape

# Count the number of resumes in each category
df['Category'].value_counts()

# Set the style
sns.set_style("whitegrid")

# Plot the countplot
sns.countplot(data=df, x='Category')

# Set labels and title
plt.xlabel('Category', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Distribution of Categories', fontsize=14)

# Rotate x-axis labels for better readability (if needed)
plt.xticks(rotation=90)

# Show the plot
plt.show()

# Get the count of each category
category_counts = df['Category'].value_counts()

# Plot the pie chart
plt.figure(figsize=(10, 10))
plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=140)

# Draw circle to make it a donut chart
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

# Equal aspect ratio ensures that pie is drawn as a circle
plt.axis('equal')

# Add a title
plt.title('Distribution of Categories')

# Display the pie chart
plt.show()

# Define a function to clean resumes
import re

def clean_resume(text):
    # Remove URLs
    text = re.sub(r"http\S+", "", text)
    # Remove non-alphanumeric characters and extra whitespaces
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    # Remove extra whitespaces
    text = re.sub(r"\s+", " ", text)
    # Remove hashtags (words starting with #)
    text = re.sub(r"#\w+", "", text)
    # Remove mentions (words starting with @)
    text = re.sub(r"@\w+", "", text)
    # Remove punctuation and special characters
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"[\n\t\r]", "", text)
    text = re.sub(r"\n", "", text)
    # Convert to lowercase
    text = text.lower()
    return text

# Clean the resumes
df['Resume'] = df['Resume'].apply(lambda x: clean_resume(x))

# Encode the categories
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(df['Category'])
df['Category_num'] = le.transform(df['Category'])

# Feature Engineering - TF-IDF Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf.fit(df['Resume'])
required_text = tfidf.transform(df['Resume'])

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(required_text, df['Category'], test_size=0.2, random_state=42)

# Model Training and Hyperparameter Tuning - KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

param_grid = {'n_neighbors': [3, 5, 7, 8]}
knn_classifier = KNeighborsClassifier()
grid_search = GridSearchCV(knn_classifier, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_score = grid_search.best_score_
best_k = best_params['n_neighbors']
best_knn_classifier = KNeighborsClassifier(n_neighbors=best_k)
best_knn_classifier.fit(X_train, y_train)

# Evaluate the model
from sklearn.metrics import classification_report
y_pred = best_knn_classifier.predict(X_test)
class_report = classification_report(y_test, y_pred)

# Print best parameters and classification report
print("Best Parameters:", best_params)
print("Best Score (CV Accuracy):", best_score)
print("Classification Report:")
print(class_report)

# Serialize the model and TF-IDF vectorizer
import pickle
pickle.dump(tfidf, open('tfidf.pkl', 'wb'))
pickle.dump(best_knn_classifier, open('best_knn_classifier.pkl', 'wb'))

# Load the model and TF-IDF vectorizer for prediction
clf = pickle.load(open('best_knn_classifier.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

# Sample Resume for Prediction
sample_resume = "sample.csv"

# Clean the input resume
cleaned_resume = clean_resume(sample_resume)

# Transform the cleaned resume using TF-IDF vectorizer
input_features = tfidf.transform([cleaned_resume])

# Make predictions using the loaded classifier
prediction_id = clf.predict(input_features)[0]

# Find the category value based on the prediction ID
filtered_df = df[df['Category_num'] == prediction_id]  # Define filtered_df here
print("Filtered DataFrame:", filtered_df)  # Add this line to print the filtered DataFrame

# Display the domain of the sample resume
print("Domain of the sample resume:", prediction_id)  # Assuming prediction_id corresponds to the domain

