## **IMDB Sentiment Analysis using BiLSTM**  
A deep learning-based sentiment analysis system for classifying IMDB movie reviews as **positive or negative** using a **BiLSTM (Bidirectional LSTM) model** with trainable word embeddings.  The project includes efficient data preprocessing (tokenization, padding, tf.data pipeline) and evaluates performance using accuracy, precision, recall, F1-score, and confusion matrix.


## **Table of Contents**
- [Introduction](#introduction)  
- [Dataset](#dataset)  
- [Model Architecture](#model-architecture)  
- [Preprocessing](#preprocessing)  
- [Training & Evaluation](#training--evaluation)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Results](#results)  
- [Future Improvements](#future-improvements)  
- [License](#license)  

## **Introduction**  
This project implements a **BiLSTM-based sentiment analysis model** to classify IMDB movie reviews. The dataset is preprocessed using **tokenization, padding, and a `tf.data` pipeline** to optimize training. The model is evaluated using **accuracy, precision, recall, F1-score, and a confusion matrix**.  

## **Dataset**  
- **Source:** IMDB movie reviews dataset (50K reviews)  
- **Split:**  
  - Training: **30,000 reviews**  
  - Validation: **10,000 reviews**  
  - Testing: **10,000 reviews**  
- **Format:** Each review is labeled as **positive (1)** or **negative (0)**.  

## **Model Architecture**  
The model follows a **BiLSTM-based architecture** for sentiment classification:  
1. **Embedding Layer** (Trainable word embeddings)  
2. **Bidirectional LSTM (BiLSTM)** for contextual learning  
3. **Dropout Layer** for regularization  
4. **Fully Connected Layers** with ReLU activation  
5. **Output Layer** (Sigmoid activation for binary classification)  

## **Preprocessing**  
- **Tokenization**: Converting words to integer sequences  
- **Padding**: Standardizing sequence lengths  
- **tf.data pipeline**: Efficient data batching and prefetching  

## **Training & Evaluation**  
- **Loss Function:** Binary Cross-Entropy  
- **Optimizer:** Adam  
- **Metrics:** Accuracy, Precision, Recall, F1-score  
- **Visualization:** Confusion matrix, word cloud analysis  

## **Installation**  
### **Requirements**  
Ensure you have **Python 3.8+** and the required libraries installed:  
```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn
