import pandas as pd
import numpy as np
import random

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from sklearn.metrics import classification_report

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# Function to load and preprocess the dataset 
def loadAndPreProcess(path):
  
  df = pd.read_json(path, lines = True) # Read the data

  df["label"] = df.annotation.apply(lambda x: x.get('label'))  # Extract label list
  df["label"] = df.label.apply(lambda x: x[0])  # Get first label from the list
    
  X = df.content.values  # Extract text content
  y = df.label.values  # Extract labels
  
  return X, y


# Load the data
X, y = loadAndPreProcess('/content/Dataset.json')


# Define tokenizer
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(X)  # Fit on your dataset

# Convert text to sequences
X_sequences = tokenizer.texts_to_sequences(X)

# Pad sequences to ensure uniform length
X_padded = pad_sequences(X_sequences, maxlen=100, padding='post', truncating='post')

# Encode labels as integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Convert to one-hot encoding
y_categorical = to_categorical(y_encoded)


model = Sequential([
    Embedding(input_dim=10000, output_dim=100, input_length=100),  # Word Embedding Layer
    
    # Sample LSTM
    #LSTM(128, return_sequences=False),  # LSTM layer with 128 units

    # Bidirectional LSTM
    Bidirectional(LSTM(128, return_sequences=False)),

    Dropout(0.5),  # Regularization
    Dense(64, activation='relu'),  # Fully connected layer
    Dense(y_categorical.shape[1], activation='softmax')  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_padded, y_categorical, epochs=30, batch_size=32, validation_split=0.2)


loss, accuracy = model.evaluate(X_padded, y_categorical)
print(f"Test Accuracy: {accuracy:.4f}")


///


from sklearn.metrics import classification_report


y_pred_prob = model.predict(X_padded)  # Get predicted probabilities
y_pred = y_pred_prob.argmax(axis=1)  # Convert probabilities to class labels


y_true = y_categorical.argmax(axis=1)  # Convert one-hot encoded labels to class indices


# Print classification report
print(classification_report(y_true, y_pred, digits=4))




///


import numpy as np

def predict_text(model, tokenizer, text, label_encoder, max_length=100):

    # Convert text into a sequence of integers
    sequence = tokenizer.texts_to_sequences([text])

    # Pad sequence to match model input size
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')

    # Make prediction
    y_pred_prob = model.predict(padded_sequence)
    y_pred = np.argmax(y_pred_prob, axis=1)  # Get predicted class index

    # Convert numerical label back to text label
    predicted_label = label_encoder.inverse_transform(y_pred)[0]

    return predicted_label


Text = ""
predicted_category = predict_text(model, tokenizer, Text, label_encoder)

print(f"Predicted Category: {predicted_category}")
