HR Workforce Intelligence System
Project Overview

The HR Workforce Intelligence System is a machine learning–based predictive analytics project designed to identify employee attrition risk and provide actionable insights for HR decision-making.

It uses the IBM HR Analytics Employee Attrition & Performance dataset (selected features) to build multiple machine learning models and analyze key drivers of employee turnover.

Author

Fatima Azeemi

Domain

Human Resources · Machine Learning · Predictive Analytics · Data Science

Problem Statement

Employee attrition is one of the most critical challenges for organizations. The cost of replacing an employee can range from 50% to 200% of annual salary, including recruitment, training, and productivity loss.

This project addresses the following questions:

Which employees are at risk of leaving?
What are the key factors influencing attrition?
How can HR teams take proactive action to reduce turnover?
Objectives
Build a machine learning pipeline to predict employee attrition
Perform exploratory data analysis to identify behavioral patterns
Compare multiple classification models
Identify the most important features affecting attrition
Generate HR-focused business insights and recommendations
Save a production-ready model for deployment
Dataset
Source: IBM HR Analytics Employee Attrition & Performance Dataset
Version: Selected 10-feature subset
Target Variable: Attrition (Yes / No)
Technologies Used
Programming
Python 3.x
Libraries
Pandas, NumPy
Matplotlib, Seaborn
Scikit-learn
XGBoost
Imbalanced-learn (SMOTE)
Joblib
Data Preprocessing
Handled missing values (none found)
Removed duplicate records
Dropped non-informative columns:
EmployeeNumber
EmployeeCount
Feature Engineering:
Age Group
Salary Band
Distance Category
Education Level mapping
Travel Risk Flag
Label Encoding for categorical variables
SMOTE applied to balance class distribution
StandardScaler used for Logistic Regression
Machine Learning Models

The following models were trained and evaluated:
Logistic Regression
Decision Tree Classifier
Random Forest Classifier
Gradient Boosting Classifier
XGBoost Classifier

Model Evaluation Results
Model	            Accuracy	F1 Score	ROC-AUC
Logistic Regression	0.6361	    0.2190	    0.5033
Decision Tree	    0.7347	    0.2500	    0.5420
Random Forest	    0.7381	    0.2524	    0.5379
Gradient Boosting	0.8095	    0.2222	    0.5627
XGBoost         	0.8129	    0.2254	    0.5522

Best Model
Gradient Boosting Classifier (based on ROC-AUC and balanced performance)

Key Insights
Overall attrition rate is approximately 16%
Frequent business travelers have the highest attrition risk
Young employees (18–30) show higher turnover rates
Low salary bands are strongly associated with attrition
Employees living farther from work are more likely to leave
Age and Daily Rate are the most influential predictive features
Business Recommendations
Improve compensation structure for low-salary employees
Introduce structured mentorship programs for young employees
Reduce unnecessary business travel load
Offer hybrid/flexible working options for distant employees
Run targeted retention programs in high-risk departments
Feature Importance

Top predictive factors include:

Age
Daily Rate
Distance From Home
Business Travel
Education Field
Model Deployment

The final model pipeline includes:

Trained ML model (Gradient Boosting)
StandardScaler (for numeric scaling)
Label Encoders
Feature schema
Saved using Joblib
Saved Model Path
models/hr_attrition_model.pkl

How to Run the Project

1. Install dependencies
pip install -r requirements.txt
2. Run notebook or script
python app.py
3. Launch Streamlit app (if applicable)
streamlit run app.py

Project Structure

HR Workforce Intelligence System/

│

├── app.py

├── models/

│   └── hr_attrition_model.pkl

├── datasets/

│   └── 
WA_Fn-UseC_-HR-Employee-Attrition-selected-columns.csv

├── outputs/

│   ├── plots/

│   └── evaluation charts

├── README.md

Conclusion

This project demonstrates how machine learning can be used in HR analytics to predict employee attrition and support data-driven decision-making. The system provides both predictive capability and actionable business insights for improving employee retention.