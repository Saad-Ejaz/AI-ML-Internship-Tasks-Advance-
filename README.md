🚀 Machine Learning Internship Projects
This repository contains three comprehensive machine learning tasks completed during my internship. Each task focuses on a unique domain of ML — from NLP classification to ML pipelines and multimodal learning — showcasing skills from data preprocessing to model deployment readiness.

📂 Task 1: News Classification with BERT
🎯 Objective
Classify news headlines into predefined categories (e.g., politics, sports, sci/tech) using BERT, a state-of-the-art NLP transformer model.

📊 Dataset
A labeled dataset of news headlines (text + category).

Preprocessed and tokenized using Hugging Face’s transformers library.

⚙️ Methodology
✅ Tokenized headlines using BERT tokenizer
✅ Fine-tuned bert-base-uncased for multi-class classification
✅ Built a Gradio interface for interactive headline classification
✅ Evaluated with accuracy and confidence scores

📈 Key Results
✅ Achieved ~80–85% accuracy on validation set

✅ Produced a web-based Gradio demo for real-time headline predictions

📂 Task 2: End-to-End ML Pipeline for Customer Churn
🎯 Objective
Build a production-ready ML pipeline to predict Telco customer churn using Scikit-learn Pipeline API.

📊 Dataset
Telco Churn Dataset (7,043 customers, 21 features)

⚙️ Methodology
✅ Preprocessing with ColumnTransformer (scaling numeric + encoding categorical)
✅ Implemented two models:

Logistic Regression

Random Forest

✅ Used GridSearchCV for hyperparameter tuning
✅ Exported trained pipeline as churn_pipeline.pkl for reusability

📈 Key Results
✅ Best model: Logistic Regression (C=10, solver=liblinear)

✅ Accuracy: ~80% on test set

✅ Fully serialized pipeline ready for production

📂 Task 3: Multimodal ML – Housing Price Prediction
🎯 Objective
Predict housing prices using tabular data + property images.

📊 Dataset
Housing dataset (Airbnb-style listings: 48k entries)

~15k property images

⚙️ Methodology
✅ Used ResNet50 CNN (pre-trained on ImageNet) to extract image features
✅ Merged extracted image features with tabular dataset
✅ Trained a regression model (Random Forest Regressor) on combined data
✅ Evaluated with MAE and RMSE

📈 Key Results
✅ MAE: 35.7 (average price error)

✅ RMSE: 43.1

✅ Built a pipeline to handle multi-modal data fusion

📊 Skills Gained
✔ Natural Language Processing (NLP) using Transformers
✔ End-to-End ML Pipeline construction & export
✔ Hyperparameter tuning with GridSearchCV
✔ Multimodal learning (CNN + tabular fusion)
✔ Model evaluation with metrics (Accuracy, MAE, RMSE)
✔ Deployment-ready practices (Joblib, Gradio)
