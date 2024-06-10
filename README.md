# Sentiment Analysis using PySpark and Logistic Regression

## Introduction

Sentiment analysis is a natural language processing (NLP) technique used to determine the sentiment conveyed in textual data, such as social media posts, customer reviews, or news articles. This project presents an analysis of sentiment using PySpark and Logistic Regression. The dataset used for analysis contains textual data from social media, along with sentiment labels.

## Data Loading and Cleaning

The sentiment dataset is loaded into a PySpark DataFrame, where each row represents a tweet along with its associated sentiment label. The following preprocessing steps are applied to ensure the quality of the data:
- **Removal of retweets**: Tweets starting with "RT" are removed to eliminate duplicate content.
- **Removal of duplicates**: Duplicate tweets are removed to ensure the uniqueness of the dataset.
- **Removal of irrelevant columns**: Columns such as "candidate", "sentiment_confidence", and others are dropped as they are not relevant for sentiment analysis.

## Text Preprocessing

Text preprocessing is a crucial step in sentiment analysis to convert raw text data into a format suitable for modeling. The following preprocessing steps are applied:
- **Tokenization**: The text data is tokenized into individual words to facilitate further analysis.
- **Stop words removal**: Common stop words are removed from the tokenized text to reduce noise in the data.
- **Handling emojis, URLs, and special characters**: Regular expressions are used to handle emojis, URLs, and special characters. URLs and special characters are removed from the text.

## Sentiment Labeling

Sentiment labeling is performed using the TextBlob library, which assigns sentiment polarity scores to each tweet. Based on these scores, tweets are labeled as positive, negative, or neutral. The sentiment labels are converted into numerical indices for model training and evaluation.

## Logistic Regression Model

Logistic Regression is applied to perform sentiment analysis on the preprocessed text data. The process involves the following steps:
- **Feature vectorization**: Textual features are converted into numerical vectors using techniques such as TF-IDF (Term Frequency-Inverse Document Frequency) or one-hot encoding.
- **Model training**: A Logistic Regression model is trained on the training data to learn the relationship between the features and sentiment labels.
- **Model evaluation**: The trained model is evaluated on the test data using metrics such as accuracy, precision, recall, and F1-score.

## Model Evaluation and Results

The trained Logistic Regression model is evaluated using the MulticlassClassificationEvaluator. The evaluation metrics provide insights into the performance of the model in classifying sentiment. In this analysis, the metrics are as follows:
- **Accuracy**: 0.686
- **Precision**: 0.685
- **Recall**: 0.686
- **F1-Score**: 0.683

These metrics indicate a reasonable performance of the Logistic Regression model in classifying sentiment.

## Linear Regression Model

In addition to logistic regression, linear regression was applied to predict sentiment labels based on the features extracted from textual data. The model's performance was evaluated using key metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (Coefficient of Determination):
- **Mean Squared Error (MSE)**: 0.00097
- **Root Mean Squared Error (RMSE)**: 0.031
- **R-squared**: 0.998

The low MSE and RMSE values indicate that the model's predictions are close to the actual sentiment labels, demonstrating its accuracy in capturing the variability in the data. The high R-squared value indicates that approximately 99.83% of the variance in the sentiment labels can be explained by the features.

## K-Means Clustering

The K-Means clustering algorithm was used to group similar data points. The clustering results indicate distinct groupings based on the provided features. Each data point was assigned to a cluster, allowing for the identification of similarities and differences among the observations. Further analysis of each cluster can provide valuable insights into the underlying structure of the data and potential patterns or trends.

## Techniques of Visualization (Graphs)

- The graph shows that the most common sentiment is neutral (0.0).
- The term "gopdebate" is the most frequently occurring, with over 4000 instances, while "Donald" appears around 200 times.

## Conclusion and Future Work

This project demonstrates the application of Logistic Regression for sentiment analysis using PySpark. The analysis provides valuable insights into the sentiment conveyed in textual data, which can be leveraged for various applications such as customer feedback analysis, brand monitoring, and market research. Future work may involve further optimization of the model, exploration of different feature engineering techniques, and incorporation of deep learning models for improved sentiment analysis performance.

## Repository Structure

- `data/`: Directory containing the dataset.
- `scripts/`: Directory containing the data loading, cleaning, and preprocessing scripts.
- `models/`: Directory containing the model training and evaluation scripts.
- `notebooks/`: Directory containing Jupyter notebooks for exploratory data analysis and model development.
- `visualizations/`: Directory containing the visualization scripts and graphs.
- `README.md`: This file.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/sentiment-analysis-pyspark.git
    ```
2. Navigate to the project directory:
    ```bash
    cd sentiment-analysis-pyspark
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```


## Contributing

This was a group project done by students of FAST NUCES Lahore.
- Abdullah Naeem
- Ibrahim Zia
- Amad Mateen
- Abdur Rehman Saeed 


---

Feel free to reach out if you have any questions or need further assistance. Happy coding!
