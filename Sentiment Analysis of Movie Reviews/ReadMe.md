# Sentiment Analysis using TF-IDF and Machine Learning

## Project Overview
This project involves training machine learning models for sentiment analysis using the "Sentiment Analysis on Movie Reviews" dataset from the Kaggle competition. The primary focus is on implementing the TF-IDF (Term Frequency-Inverse Document Frequency) technique and using it to train and evaluate various machine learning models for sentiment classification.

## Dataset
- The dataset is available on Kaggle and consists of movie reviews with associated sentiment labels.
- It is split into training and testing sets, with different sentiment classes.

## Implementation Steps

### 1. Download and Explore Dataset
- The dataset is downloaded from Kaggle.
- It is loaded into Pandas DataFrames for exploration.

### 2. Implement the TF-IDF Technique
- The TF-IDF technique is applied to the text data to transform it into numerical features.
- Custom tokenization and stop words are used in this process.

### 3. Train Baseline Model
- A baseline machine learning model (Logistic Regression) is trained using the TF-IDF features.
- Model performance is evaluated on the training and validation sets.

### 4. Train & Fine-Tune Different ML Models
- Two additional machine learning models (Decision Trees and Naive Bayes) are trained.
- Model performance is compared to the baseline model.

### 5. Document & Submit
- The project results and insights are documented in this `readme.md` file.
- The final models are used to make predictions on the test set and a submission file is created for Kaggle.

## Usage
- The Jupyter Notebook or Python script containing the code can be executed step by step to reproduce the results.
- Make sure to install the required libraries and have the dataset in the specified location.

## Requirements
- Python 3.x
- Jupyter Notebook (optional)
- Required Python libraries (NumPy, Pandas, Scikit-Learn, NLTK, etc.)


Happy coding!
