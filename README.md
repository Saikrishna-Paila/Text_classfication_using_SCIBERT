# Text Classification Using SciBERT

## Project Overview
This project implements a multi-label text classification model using SciBERT, a pre-trained transformer-based model specifically trained on scientific texts. The primary goal is to classify abstracts into multiple categories such as "Computer Science," "Physics," "Mathematics," etc., using a custom deep learning model with multiple heads (Fully Connected, LSTM, and Convolutional).

## Features and Functionalities
- **Multi-Head Classification Model:**
  - Fully connected layer for direct classification.
  - LSTM layer for sequential feature extraction.
  - Convolutional layer for local feature extraction.
  - Final output is an ensemble of all three heads.

- **Preprocessing:**
  - Removes stopwords, URLs, non-alphabetic characters, and excessive whitespace.
  - Tokenization using SciBERT tokenizer with padding and truncation.

- **Dataset Handling:**
  - Custom `AbstractDataset` class to handle tokenized inputs and labels.
  - Data split into training, validation, and testing sets.

- **Model Training and Validation:**
  - Training loop with early stopping based on validation performance.
  - Metrics calculated include Accuracy and Quadratic Weighted Kappa.
  - Model checkpointing to save the best-performing model.

- **Testing and Submission:**
  - Generates predictions on the test dataset.
  - Outputs a CSV file containing predictions for submission.

## Project Files
- `SciBert.py`: The main script containing all the code.
- `Train_ML.csv`: Training dataset file.
- `Test_submission_netid.csv`: Test dataset file.
- `checkpoints/`: Directory to save model checkpoints.

## Setup Instructions
### 1. Environment Setup
- Install Python 3.8 or later.
- Install necessary libraries using:
  ```bash
  pip install torch transformers pandas numpy nltk scikit-learn tqdm
  ```
- Download NLTK stopwords data:
  ```python
  import nltk
  nltk.download('stopwords')
  ```

### 2. Directory Structure
- Place the training dataset (`Train_ML.csv`) and test dataset (`Test_submission_netid.csv`) in the same directory as the script.
- Create a directory named `checkpoints/` for saving model checkpoints.

### 3. Run the Script
- Execute the script using:
  ```bash
  python SciBert.py
  ```

## Detailed Explanation of Code Components

### 1. Preprocessing
The `preprocess_text` function:
- Converts text to lowercase.
- Removes URLs, non-alphabetic characters, and extra spaces.
- Removes repeated characters (e.g., "heyyy" becomes "hey").
- Removes stopwords using NLTK.

### 2. Dataset Class
The `AbstractDataset` class:
- Takes tokenized inputs and labels as input.
- Provides data in a format compatible with PyTorch DataLoader.

### 3. Model Architecture
The `MultiHeadClassifier` model:
- Combines outputs from three heads (fully connected, LSTM, and convolutional).
- Uses SciBERT for feature extraction from input text.

### 4. Training Loop
- Uses Binary Cross Entropy with Logits Loss (`BCEWithLogitsLoss`) as the loss function.
- Implements early stopping to terminate training if the validation performance does not improve for 3 consecutive epochs.

### 5. Validation and Metrics
- Validation loop evaluates the model on the validation dataset.
- Calculates accuracy and Quadratic Weighted Kappa to measure agreement.

### 6. Testing and Submission
- Generates predictions on the test dataset.
- Converts probabilities to binary predictions using a threshold of 0.5.
- Saves predictions in the test file with new columns for each category.

## Key Metrics
- **Accuracy:** Measures the proportion of correct predictions.
- **Quadratic Weighted Kappa:** Measures the agreement between predicted and true labels, accounting for the possibility of agreement by chance.

## Outputs
- **Model Checkpoints:** Saved in `checkpoints/best_model.pth`.
- **Submission File:** Test predictions are saved in `Test_submission_netid.csv`.



