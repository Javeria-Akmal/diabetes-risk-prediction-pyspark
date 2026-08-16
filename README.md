🩺 Diabetes Risk Prediction — Big Data & Machine Learning Pipeline

An end-to-end machine learning system that predicts an individual's diabetes risk using demographic, lifestyle, and clinical data. The project covers the full pipeline — from big data processing with PySpark, through feature engineering and model training, to a live, interactive Streamlit web application.

🚀 Live Demo

👉 https://diabetes-risk-prediction-pyspark-mzvhgqercuq4uhqi53mxfe.streamlit.app/

Enter health and lifestyle details (age, BMI, glucose levels, activity, etc.) and get an instant diabetes risk prediction with probability score.

📊 Project Overview

This project analyzes a dataset of 100,000 patient records to build a classification model that predicts whether an individual is likely to have diabetes. It demonstrates a complete data science workflow at scale, combining distributed data processing with traditional ML techniques.

Key steps:

Large-scale data processing and exploration using PySpark (RDDs, DataFrames, Spark SQL, window functions)
Data cleaning, outlier removal (IQR method), and missing value imputation
Feature engineering, encoding, and scaling
Handling class imbalance with SMOTE
Model training and comparison (Random Forest, SVM, XGBoost)
Hyperparameter tuning with GridSearchCV
Careful removal of data leakage features to ensure realistic, generalizable performance
Deployment as an interactive web app for real-time predictions
🛠️ Tech Stack
Category	Tools
Big Data Processing	PySpark, Dask
Data Analysis	Pandas, NumPy
Machine Learning	Scikit-learn, XGBoost, imbalanced-learn (SMOTE)
Visualization	Matplotlib, Seaborn
Cloud & Public Data	Google BigQuery, Google Colab
Deployment	Streamlit, Streamlit Community Cloud
Version Control	Git, GitHub
📈 Model Performance

Multiple models were trained and evaluated (Random Forest, SVM, XGBoost) using accuracy, precision, recall, F1-score, and AUC. XGBoost was selected as the final model based on overall performance.

Note: An earlier version of this model included features derived from the target label (e.g., diabetes stage, row-level ranking), which artificially inflated accuracy to ~99.9%. These were identified and removed to prevent data leakage, resulting in a model that reflects realistic, real-world predictive performance on genuinely unseen inputs.

🧩 Features Used

The model takes 29 real-world features across four categories:

Demographics: age, gender, ethnicity, education level, income level, employment status
Lifestyle: smoking status, alcohol consumption, physical activity, diet score, sleep hours, screen time
Medical History: family history of diabetes, hypertension history, cardiovascular history
Clinical Measurements: BMI, waist-to-hip ratio, blood pressure, heart rate, cholesterol panel, triglycerides, glucose (fasting/postprandial), insulin level, HbA1c
📁 Repository Structure
├── Diabetes_Health_Indicators_Dataset.ipynb   # Full analysis & model training notebook
├── app.py                                     # Streamlit web application
├── diabetes_model.pkl                         # Trained XGBoost model
├── scaler.pkl                                 # Fitted StandardScaler
├── encoders.pkl                               # LabelEncoders for categorical features
├── feature_columns.json                       # Expected feature order for inference
├── requirements.txt                           # Python dependencies
└── README.md
⚙️ Running Locally
bash
# Clone the repository
git clone https://github.com/Javeria-Akmal/diabetes-risk-prediction-pyspark.git
cd diabetes-risk-prediction-pyspark

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
⚠️ Disclaimer

This tool is built for educational and demonstration purposes only. It is not a medical diagnostic tool. Predictions should not be used as a substitute for professional medical advice — please consult a healthcare provider for any health concerns.

👩‍💻 Author

Javeria Akmal BS Data Science, KFUEIT GitHub
