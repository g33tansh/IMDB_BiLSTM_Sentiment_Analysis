## **IMDB Sentiment Analysis using BiLSTM**  
A deep learning-based sentiment analysis system for classifying IMDB movie reviews as **positive or negative** using a **BiLSTM (Bidirectional LSTM) model** with trainable word embeddings. The project includes efficient data preprocessing (tokenization, padding, tf.data pipeline) and evaluates performance using accuracy, precision, recall, F1-score, confusion matrix, and **TensorBoard for real-time training visualization**.

## **Table of Contents**
- [Introduction](#introduction)  
- [Dataset](#dataset)  
- [Model Architecture](#model-architecture)  
- [Preprocessing](#preprocessing)  
- [Training & Evaluation](#training--evaluation)  
- [TensorBoard Visualization](#tensorboard-visualization)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Results](#results)  
- [Future Improvements](#future-improvements)  
- [License](#license)  

## **Introduction**  
This project implements a **BiLSTM-based sentiment analysis model** to classify IMDB movie reviews. The dataset is preprocessed using **tokenization, padding, and a `tf.data` pipeline** to optimize training. The model is evaluated using **accuracy, precision, recall, F1-score, a confusion matrix, and TensorBoard for training visualization**.

## **Dataset**  
- **Source:** IMDB movie reviews dataset (50K reviews)  
- **Split:**  
  - Training: **25,000 reviews**  
  - Testing: **25,000 reviews**  
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
- **Visualization:**  
  - **Confusion Matrix** for classification performance  
  - **Word Cloud** to analyze frequent words  
  - **TensorBoard** for real-time training visualization  

## **TensorBoard Visualization**  
TensorBoard is integrated to monitor training metrics such as **loss, accuracy, learning rate changes, and histograms**.  

### **Using TensorBoard**  
To track training metrics, TensorBoard is added as a callback:  
```python
import datetime  
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)  

history = model.fit(
    train_dataset,  
    validation_data=val_dataset,  
    epochs=10,  
    callbacks=[tensorboard_callback]  
)
```

After training, start TensorBoard:  
```bash
tensorboard --logdir logs/fit --port 6006
```
Then open your browser and visit:  
[http://localhost:6006](http://localhost:6006)  

## **Installation**  
### **Requirements**  
Ensure you have **Python 3.8+** and the required libraries installed:  
```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn nltk wordcloud
```

## **Usage**  
1. **Prepare Dataset**: Load and preprocess IMDB reviews.  
2. **Train Model**: Run the BiLSTM model on preprocessed data.  
3. **Monitor Training**: Use TensorBoard for real-time visualization.  
4. **Evaluate Model**: Compute accuracy, precision, recall, and F1-score.  
5. **Visualize Results**: Generate confusion matrix and word cloud.  

## **Results**  
- **Training Loss & Accuracy Plots** via TensorBoard.  
- **Confusion Matrix** showcasing classification performance.  
- **Word Cloud** highlighting key words in positive and negative reviews.  

## **Future Improvements**  
- Implement **pretrained embeddings (GloVe, Word2Vec, or BERT)**.  
- Explore **CNN+BiLSTM hybrid architectures** for better accuracy.  
- Optimize **hyperparameters** using grid search or Bayesian optimization.  

## **License**  
This project is released under the **MIT License**.

