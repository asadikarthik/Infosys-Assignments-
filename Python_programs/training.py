#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from transformers import pipeline
nltk.download('stopwords')


# In[3]:


# Specify the folder path where your datasets are located
folder_path = 'recruitment_data'  # Update this to the folder where the datasets are located

# List all Excel files in the folder
files = [file for file in os.listdir(folder_path) if file.endswith('.xlsx')]


# In[4]:


# Loop through each file and print only the column names
for file in files:
    df = pd.read_excel(folder_path + '/' + file)
    print(f"Columns in {file}:")
    print([col for col in df.columns])  # Ensure the output only contains column names
    print()  # Add a blank line for better readability


# In[5]:


# Code for Merging all the datasets and creating the full dataset for traing and testing
dataframes = []

for file in files:
    file_path = os.path.join(folder_path, file)
    combined_df = pd.read_excel(file_path)
    dataframes.append(combined_df)

# Combine all dataframes into a single dataframe
combined_df = pd.concat(dataframes, ignore_index=True)



# # Loading and Inspecting Datasets
# Point: Specify the folder path where datasets are located and list all Excel files.
# Code reads all .xlsx files in the given folder path and lists the column names of each dataset for inspection.
# Point: Merge all datasets into a single DataFrame for analysis.
# Combines individual dataframes into one comprehensive dataset for training and testing.
# 
# 

# In[7]:


# Save the combined dataframe to a CSV file
combined_df.to_csv('combined_dataset.csv', index=False)
print("Combined dataframe saved to 'combined_dataset.csv'.")


# In[8]:


combined_df.shape


# In[9]:


combined_df.head()


# In[10]:


combined_df.info()


# In[11]:


# # Drop the last two columns: 'Unnamed: 0' and 'num_words_in_transcript'
# columns_to_drop = ['Unnamed: 0', 'num_words_in_transcript']
# combined_df = combined_df.drop(columns=columns_to_drop, errors='ignore')


# In[12]:


# Function to clean text columns
def clean_text_column(column):
    """
    Cleans a text column by:
    1. Removing all non-alphabetic characters.
    2. Removing extra spaces.
    3. Converting text to lowercase.
    """
    if column.dtype == 'object':  # Only apply to text columns
        return column.str.replace(r"[^a-zA-Z\s]", "", regex=True).str.strip().str.lower()
    return column

# Clean specific text columns
columns_to_clean = ['Transcript', 'Resume', 'Job Description', 'Reason for decision']
for col in columns_to_clean:
    if col in combined_df.columns:
        combined_df[col] = clean_text_column(combined_df[col])

# Display cleaned data for verification
print("Cleaned Data Sample:")
combined_df.head()


# # Data Overview and Cleaning 
# Point: Display shape, column information, and preview the combined dataset.
# 
# Provides basic information (shape, head, info) for an overview of the dataset's structure and size.
# Point: Clean text columns by removing special characters, trimming whitespace, and converting text to lowercase.
# 
# Applies cleaning functions to specified text-based columns like 'Transcript', 'Resume', 'Job Description', and 'Reason for decision' to standardize data.
# Point: Handle decision column and unify labels ('select' and 'selected').
# 
# Maps variations of decision outcomes into standardized labels ('select', 'reject') for consistency.

# In[14]:


combined_df['decision'].unique()


# In[15]:


def process_decision(text):
    if text in ['select','selected']:
        return 'select'
    else :
        return 'reject'


# In[16]:


combined_df['Role'].unique()


# In[17]:


unique_count = combined_df.groupby('Role')['ID'].count()
unique_count


# In[18]:


# Generate summary insights
insights = {}

# Total Candidates
insights['Total Candidates'] = len(combined_df)

# Selected and Rejected Candidates
if 'decision' in combined_df.columns:
    decision_counts = combined_df['decision'].str.strip().value_counts()
    insights['Select Candidates'] = decision_counts.get('select', 0)
    insights['Reject Candidates'] = decision_counts.get('reject', 0)
    insights['Selected Candidates'] = decision_counts.get('selected', 0)
    insights['Rejected Candidates'] = decision_counts.get('rejected', 0)
else:
    insights['Selected Candidates'] = "Column 'Decision' not found"
    insights['Rejected Candidates'] = "Column 'Decision' not found"

# Most Common Reason for Decision
if 'Reason for decision' in combined_df.columns:
    insights['Most Common Reason for Decision'] = combined_df['Reason for decision'].mode()[0]
else:
    insights['Most Common Reason for Decision'] = "Column 'Reason for decision' not found"

# Print insights
print("Insights:")
for key, value in insights.items():
    print(f"{key}: {value}")


# In[19]:


# Check for null values
print("Null values in combined dataset:")
print(combined_df.isnull().sum())


# In[20]:


# Basic statistics for numeric columns
print("Basic statistics for numeric columns:")
combined_df.describe()


# In[21]:


# Add length columns for Transcript, Resume, and Job Description
combined_df['Transcript_length'] = combined_df['Transcript'].apply(lambda x: len(str(x)))
combined_df['Resume_length'] = combined_df['Resume'].apply(lambda x: len(str(x)))
combined_df['Job_Description_length'] = combined_df['Job Description'].apply(lambda x: len(str(x)))


# In[22]:


print(combined_df.columns.tolist())


# In[23]:


combined_df.head()


# In[24]:


# Clean the target column
combined_df['decision'] = combined_df['decision'].replace({'reject': 'rejected', 'select': 'selected'})

# Verify the cleaned column
print("Unique target values after cleaning:", df['decision'].unique())


# In[25]:


# # Save the combined dataframe to a CSV file
# combined_df.to_csv('cleaned_dataset.csv', index=False)
# print("Combined dataframe saved to 'combined_dataset.csv'.")


# # Data Analysis and Insights
# Point: Generate summary insights, including:
# Total candidates, number of selected/rejected candidates, and most common reasons for decisions.
# Point: Add text length columns for selected features.
# Computes and appends new columns indicating the length of textual data in 'Transcript', 'Resume', and 'Job Description'.

# In[27]:


combined_df.groupby(['Role','decision'])['Transcript_length'].mean()


# In[28]:


combined_df.groupby(['Role','decision'])['Transcript_length'].median()


# In[29]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Ensure the necessary columns exist in the dataset
required_columns = ['Transcript', 'Resume', 'Job Description']
for col in required_columns:
    if col not in combined_df.columns:
        raise ValueError(f"Missing required column: {col}")

# Fill missing values with empty strings to avoid errors
for col in required_columns:
    combined_df[col] = combined_df[col].fillna("")

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Define a function to calculate similarity score
def calculate_similarity(text1, text2):
    tfidf_matrix = tfidf_vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return similarity

# Calculate scores
combined_df['Transcript_Resume_Score'] = combined_df.apply(
    lambda row: calculate_similarity(row['Transcript'], row['Resume']), axis=1
)
combined_df['Transcript_JobDescription_Score'] = combined_df.apply(
    lambda row: calculate_similarity(row['Transcript'], row['Job Description']), axis=1
)
combined_df['Resume_JobDescription_Score'] = combined_df.apply(
    lambda row: calculate_similarity(row['Resume'], row['Job Description']), axis=1
)

# Calculate the final score as an average of the pairwise scores
combined_df['Final_Match_Score'] = combined_df[
    ['Transcript_Resume_Score', 'Transcript_JobDescription_Score', 'Resume_JobDescription_Score']
].mean(axis=1)

# # Save the dataframe with the scores to a file
# combined_df.to_csv('combined_dataset_with_scores.csv', index=False)
# print("Combined dataset with match scores saved to 'combined_dataset_with_scores.csv'.")

# Display a sample of the data with the calculated scores
combined_df[['Transcript', 'Resume', 'Job Description', 'Transcript_Resume_Score',
                   'Transcript_JobDescription_Score', 'Resume_JobDescription_Score', 'Final_Match_Score']].head()


# In[30]:


# Save the dataframe with the scores to a file
combined_df.to_csv('combined_dataset_with_scores.csv', index=False)
print("Combined dataset with match scores saved to 'combined_dataset_with_scores.csv'.")


# # TF-IDF Vectorizer:
# 
# Converts text data into numerical vectors while giving importance to unique terms in each document.
# Cosine Similarity:
# 
# Measures the similarity between two vectors. It outputs a value between 0 (no similarity) and 1 (identical).
# Pairwise Similarities:
# 
# Transcript vs. Resume
# Transcript vs. Job Description
# Resume vs. Job Description
# Final Match Score:
# 
# The average of the three similarity scores to give an overall measure.

# In[32]:


# Configure visualizations
sns.set(style="whitegrid")

# Plot the most common 'Reason for Decision' if the column exists
if 'Reason for decision' in combined_df.columns:
    plt.figure(figsize=(12, 6))
    
    # Get the top 10 most common reasons
    reason_counts = combined_df['Reason for decision'].value_counts().head(10)
    
    # Create the bar plot
    sns.barplot(x=reason_counts.values, y=reason_counts.index, palette='coolwarm')
    
    # Add plot title and axis labels
    plt.title('Top 10 Most Common Reasons for Decision', fontsize=16)
    plt.xlabel('Count', fontsize=12)
    plt.ylabel('Reason', fontsize=12)
    plt.tight_layout()
    plt.show()


# In[33]:


# Check for missing values or completely empty rows
print("Number of missing values in Transcript:", combined_df['Transcript'].isnull().sum())
print("Number of missing values in Resume:", combined_df['Resume'].isnull().sum())
print("Number of missing values in Transcript:", combined_df['Reason for decision'].isnull().sum())
print("Number of missing values in Resume:", combined_df['Job Description'].isnull().sum())


# In[34]:


# Apply cleaning to all object-type columns in the dataset
combined_df = combined_df.apply(lambda col: clean_text_column(col) if col.dtype == 'object' else col)


# In[35]:


# Check for remaining unwanted characters
for col in columns_to_clean:
    if col in combined_df.columns:
        print(f"Sample cleaned data from column '{col}':")
        combined_df[col].head()


# In[36]:


plt.figure(figsize=(8, 6))
sns.histplot(combined_df['Transcript_length'], kde=True, bins=30, color='skyblue')
plt.title("Distribution of Number of Words in Transcript", fontsize=16)
plt.xlabel("Number of Words", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.show()


# In[37]:


plt.figure(figsize=(8, 6))
sns.boxplot(x=combined_df['Transcript_length'], color='orange')
plt.title("Boxplot for Number of Words in Transcript", fontsize=16)
plt.xlabel("Number of Words", fontsize=12)
plt.show()


# In[38]:


plt.figure(figsize=(8, 6))
sns.countplot(data=combined_df, x='decision', palette='viridis')
plt.title("Distribution of Decisions", fontsize=16)
plt.xlabel("Decision", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.show()


# In[39]:


plt.figure(figsize=(10, 6))
reason_counts = combined_df['Reason for decision'].value_counts().head(10)
sns.barplot(x=reason_counts.index, y=reason_counts.values, palette='coolwarm')
plt.title("Top 10 Reasons for Decision", fontsize=16)
plt.xlabel("Reason", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.show()


# # Data Visualization
# Point: Visualize the distribution of decisions.
# 
# Creates a bar plot for the decision column to show the counts of 'select' and 'reject' labels.
# Point: Analyze common reasons for decision.
# 
# Generates a bar chart for the top 10 most common reasons for candidate decisions.
# Point: Analyze word distribution in 'Transcript' column.
# 
# Plots a histogram and boxplot to display the distribution and variability of word counts in transcripts.

# In[41]:


# # Save the combined dataframe to a CSV file
# combined_df.to_csv('combined_cleaned_dataset.csv', index=False)
# print("Combined dataframe saved to 'combined_cleaned_dataset.csv'.")


# In[42]:


get_ipython().system('pip install xgboost')


# In[43]:


# Import necessary libraries
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import xgboost as xgb
import matplotlib.pyplot as plt


# # Import Necessary Libraries
# Use Python libraries for machine learning and data processing:
# os for file operations.
# numpy and pandas for numerical and data manipulation.
# sklearn modules for preprocessing, model building, and evaluation.
# xgboost for gradient boosting algorithms.
# matplotlib for plotting results.

# In[45]:


# Define folder path
folder_path = 'recruitment_data'

# Load all Excel files
files = [file for file in os.listdir(folder_path) if file.endswith('.xlsx')]
dataframes = [pd.read_excel(os.path.join(folder_path, file)) for file in files]
combined_df = pd.concat(dataframes, ignore_index=True)

# Clean text columns
def clean_text_column(column):
    if column.dtype == 'object':
        return column.str.replace(r"[^a-zA-Z\s]", "", regex=True).str.strip().str.lower()
    return column

columns_to_clean = ['Transcript', 'Resume', 'Job Description', 'Reason for decision']
for col in columns_to_clean:
    if col in combined_df.columns:
        combined_df[col] = clean_text_column(combined_df[col])

# Process decision column
combined_df['decision'] = combined_df['decision'].apply(lambda x: 'select' if x in ['select', 'selected'] else 'reject')


# # Define Folder Path and Load Data
# Specify the folder containing recruitment data.
# Load all Excel files from the folder using os and pandas.
# Combine data from all files into a single DataFrame using pd.concat().
# 
# Clean Text Columns
# Create a function to clean text columns:
# Remove non-alphabetic characters using regex.
# Convert text to lowercase and strip whitespace.
# Apply the function to specific columns, e.g., Transcript, Resume, etc.
# 

# In[47]:


# Separate features and target
X = combined_df.drop(['decision', 'ID'], axis=1, errors='ignore')
y = combined_df['decision']

# Identify categorical and numeric columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(include=['number']).columns.tolist()

# Define preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# Transform the data
X_preprocessed = preprocessor.fit_transform(X)

# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Encode target labels


# # Process Decision Column
# 
# Normalize decision values:
# Map all forms of "select" (e.g., "selected") to a single label, select.
# Map other values to reject.
# 
# # Feature and Target Separation
# Separate features (X) and target (y).
# Drop unnecessary columns like ID and decision from X.
# Identify:
# Categorical columns for encoding.
# Numeric columns for scaling.
# Preprocessing Pipeline
# 
# # Use ColumnTransformer to:
# Apply StandardScaler to numeric columns.
# Apply OneHotEncoder to categorical columns.
# Preprocess the feature set using this pipeline.
# Encode Target Variable
# Use LabelEncoder to convert decision labels (select, reject) into numerical values (0, 1).

# In[49]:


# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_preprocessed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"Train set size: {X_train.shape}, Test set size: {X_test.shape}")


# # Train-Test Split
# Split data into training (80%) and testing (20%) sets using train_test_split.
# 
# Use stratification to maintain the class distribution.

# # 1. Decision Tree

# In[52]:


# Decision Tree with hyperparameter tuning
dt = DecisionTreeClassifier(random_state=42)
param_grid_dt = {'max_depth': [3, 5, 10], 'min_samples_split': [2, 5, 10]}
grid_dt = GridSearchCV(dt, param_grid_dt, cv=5, scoring='roc_auc')
grid_dt.fit(X_train, y_train)

# Best model and evaluation
best_dt = grid_dt.best_estimator_
y_pred_dt = best_dt.predict(X_test)
y_prob_dt = best_dt.predict_proba(X_test)[:, 1]
acc_dt = accuracy_score(y_test, y_pred_dt)
roc_dt = roc_auc_score(y_test, y_prob_dt)

print("Decision Tree - Accuracy:", acc_dt)
print("Decision Tree - ROC AUC Score:", roc_dt)


# # Decision Tree Model
# Perform hyperparameter tuning for max_depth and min_samples_split.
# 
# Evaluate the best model:
# Calculate accuracy and ROC AUC scores.
# 
# Achieved accuracy: 61.89%; ROC AUC: 65.57%.
# # How It Works:
# Splits data based on feature values to create a tree structure.
# 
# Each node represents a feature condition, and leaves represent the decision (class).

# # 2. Random Forest

# In[55]:


# Random Forest with hyperparameter tuning
rf = RandomForestClassifier(random_state=42)
param_grid_rf = {'n_estimators': [100, 200], 'max_depth': [5, 10], 'min_samples_split': [2, 5]}
grid_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='roc_auc')
grid_rf.fit(X_train, y_train)

# Best model and evaluation
best_rf = grid_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)
y_prob_rf = best_rf.predict_proba(X_test)[:, 1]
acc_rf = accuracy_score(y_test, y_pred_rf)
roc_rf = roc_auc_score(y_test, y_prob_rf)

print("Random Forest - Accuracy:", acc_rf)
print("Random Forest - ROC AUC Score:", roc_rf)


# # Random Forest Model
# Perform hyperparameter tuning for n_estimators, max_depth, and min_samples_split.
# 
# Evaluate the best model:
# Calculate accuracy and ROC AUC scores.
# 
# Achieved accuracy: 72.28%; ROC AUC: 94.58%.
# # How It Works:
# Combines multiple decision trees trained on random subsets of data and features.
# 
# Aggregates the predictions (majority vote for classification).

# # 3. XGBoost

# In[58]:


import xgboost as xgb

# XGBoost with hyperparameter tuning
xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
param_grid_xgb = {'n_estimators': [100, 200], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.1]}
grid_xgb = GridSearchCV(xgb_clf, param_grid_xgb, cv=5, scoring='roc_auc')
grid_xgb.fit(X_train, y_train)

# Best model and evaluation
best_xgb = grid_xgb.best_estimator_
y_pred_xgb = best_xgb.predict(X_test)
y_prob_xgb = best_xgb.predict_proba(X_test)[:, 1]
acc_xgb = accuracy_score(y_test, y_pred_xgb)
roc_xgb = roc_auc_score(y_test, y_prob_xgb)

print("XGBoost - Accuracy:", acc_xgb)
print("XGBoost - ROC AUC Score:", roc_xgb)


# # XGBoost Model
# Perform hyperparameter tuning for n_estimators, max_depth, and learning_rate.
# 
# Evaluate the best model:
# Calculate accuracy and ROC AUC scores.
# Achieved accuracy: 80.63%; ROC AUC: 93.23%.
# # How It Works:
# Builds sequential trees where each tree corrects the errors of the previous ones.
# 
# Optimizes performance using gradient descent.
# 
# Implements regularization to reduce overfitting.
# 

# # 4. Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score

# Logistic Regression with hyperparameter tuning
lr = LogisticRegression(max_iter=1000, random_state=42)
param_grid_lr = {'C': [0.01, 0.1, 1, 10]}
grid_lr = GridSearchCV(lr, param_grid_lr, cv=5, scoring='roc_auc')
grid_lr.fit(X_train, y_train)

# Best model and evaluation
best_lr = grid_lr.best_estimator_
y_pred_lr = best_lr.predict(X_test)
y_prob_lr = best_lr.predict_proba(X_test)[:, 1]
acc_lr = accuracy_score(y_test, y_pred_lr)
roc_lr = roc_auc_score(y_test, y_prob_lr)

print("Logistic Regression - Accuracy:", acc_lr)
print("Logistic Regression - ROC AUC Score:", roc_lr)


# In[ ]:


# # Bootstrapping to estimate standard errors
# n_iterations = 100
# coefficients_bootstrap = []

# for i in range(n_iterations):
#     X_resampled, y_resampled = resample(X_train, y_train, random_state=i)
#     bootstrap_lr = LogisticRegression(max_iter=1000, random_state=42).fit(X_resampled, y_resampled)
#     coefficients_bootstrap.append(bootstrap_lr.coef_.flatten())

# # Calculate standard errors from bootstrap samples
# coefficients_bootstrap = np.array(coefficients_bootstrap)
# std_errors = coefficients_bootstrap.std(axis=0)

# # Extract coefficients and intercept
# coefficients = best_lr.coef_.flatten()
# intercept = best_lr.intercept_[0]

# # Calculate z-scores and p-values
# z_scores = coefficients / std_errors
# p_values = 2 * (1 - norm.cdf(np.abs(z_scores)))  # Two-tailed p-value

# # Confidence intervals
# conf_intervals = np.array([
#     [coef - 1.96 * std_err, coef + 1.96 * std_err]
#     for coef, std_err in zip(coefficients, std_errors)
# ])

# # Create summary table
# summary_table = pd.DataFrame({
#     # 'Variable': ['Intercept'] + list(X_train.columns),
#     'Coefficient': [intercept] + coefficients.tolist(),
#     'Std. Error': [np.nan] + std_errors.tolist(),
#     'z': [np.nan] + z_scores.tolist(),
#     'P>|z|': [np.nan] + p_values.tolist(),
#     '[0.025': [np.nan] + conf_intervals[:, 0].tolist(),
#     '0.975]': [np.nan] + conf_intervals[:, 1].tolist()
# })

# # Printing summary table
# print(summary_table)


# # Logistic Regression Model
# Perform hyperparameter tuning using GridSearchCV for C values.
# 
# Evaluate the best model:
# Calculate accuracy and ROC AUC scores.
# 
# Achieved accuracy: 83.62%; ROC AUC: 94.39%.
# # How It Works:
# Predicts the probability of a class using the logistic function (sigmoid curve).
# 
# Output probabilities are converted into binary classifications based on a threshold (e.g., 0.5).

# In[ ]:


# Summarize results
results = {
    'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest'],
    'Accuracy': [acc_lr, acc_dt, acc_rf],
    'ROC AUC': [roc_lr, roc_dt, roc_rf]
}

results_df = pd.DataFrame(results)
print(results_df)

# Identify the best model
best_model_index = results_df['ROC AUC'].idxmax()
best_model_name = results_df.loc[best_model_index, 'Model']
print("Best Model:", best_model_name)


# # Results Summary
# Compile results into a DataFrame with accuracy and ROC AUC for all models.
# 
# Identify the best model based on ROC AUC:
# Best Model: Random Forest (Accuracy: 72.28%, ROC AUC: 94.58%).

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Define a function to plot results for each model
def plot_model_results(model_name, model, X_train, X_test, y_train, y_test):
    # Predict
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Print accuracy
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()

    # Plot Feature Importance if the model supports it
    if hasattr(model, "feature_importances_"):
        feature_importances = model.feature_importances_
        if feature_importances is not None and len(feature_importances) > 0:
            # Plot feature importance
            plt.figure(figsize=(8, 6))
            plt.barh(range(len(feature_importances)), feature_importances)
            plt.title(f"{model_name} - Feature Importance")
            plt.xlabel("Importance")
            plt.ylabel("Features")
            plt.show()

# Models to evaluate
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "XGBoost": xgb.XGBClassifier(eval_metric='logloss', random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
}

# Train and plot for each model
for model_name, model in models.items():
    model.fit(X_train, y_train)
    plot_model_results(model_name, model, X_train, X_test, y_train, y_test)


# # Plotting Model Results
# Create a function to:
# Predict and evaluate using metrics like accuracy and classification report.
# 
# Visualize confusion matrix using a heatmap.
# Plot feature importance (for models that support it).
# Apply the function to all trained models.

# In[ ]:


get_ipython().system('pip install shap')


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import PartialDependenceDisplay
from sklearn.preprocessing import StandardScaler  # Import StandardScaler


# In[ ]:


# Load your data (replace with your actual file path)
data = pd.read_csv("combined_cleaned_dataset.csv")


# In[ ]:


data.head()


# In[ ]:


# Assuming 'data' is your pandas DataFrame
unique_decisions = data['decision'].unique()
print("Unique values in 'decision' column:", unique_decisions)


# In[ ]:


# Assuming 'decision' is the target variable
X = data[['Transcript_length', 'Resume_length', 'Job_Description_length']] 
y = data['decision']


# In[ ]:


# Scale the features (important for SHAP)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[ ]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42) # Split the data


# In[ ]:


# Train a RandomForest model
model = RandomForestClassifier()  # Replace with your desired hyperparameters
model.fit(X_scaled, y)


# -*- coding: utf-8 -*-
"""Training.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1OTVbLPji8DTiE0iV8zTbkD5x9RXkzgcM
"""

import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import PartialDependenceDisplay
from sklearn.preprocessing import StandardScaler  # Import StandardScaler

# Load your data (replace with your actual file path)
data = pd.read_csv("/content/combined_cleaned_dataset.csv")

data.head()

# Assuming 'data' is your pandas DataFrame
unique_decisions = data['decision'].unique()
print("Unique values in 'decision' column:", unique_decisions)

# Assuming 'data' is your pandas DataFrame
data['decision'] = data['decision'].replace({'select': 'selected', 'reject': 'rejected'})

# Assuming 'decision' is the target variable
X = data[['Transcript_length', 'Resume_length', 'Job_Description_length']]
y = data['decision']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42) # Split the data

# Scale the features (important for SHAP)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train a RandomForest model
model = RandomForestClassifier()  # Replace with your desired hyperparameters
model.fit(X_scaled, y)

# Initialize and compute SHAP values
explainer = shap.Explainer(model.predict_proba, X_scaled)
shap_values = explainer(X_scaled)

# Assuming 'decision' is the target variable and has 2 classes ('selected' (1) and 'rejected'(0))
# To focus on 'selected' (1), set class_index to 1
# To focus on 'rejected' (0), set class_index to 0

class_index = 1  # For 'selected' class

# Initialize and compute SHAP values for the specific class
# model.predict_proba returns probabilities for all classes,
# we select the column corresponding to the class_index
explainer = shap.Explainer(lambda X: model.predict_proba(X)[:, class_index], X_scaled)
shap_values = explainer(X_scaled)

# Now you can create the beeswarm plot
shap.plots.beeswarm(shap_values)

# Assuming 'decision' is the target variable and has 2 classes ('selected' (1) and 'rejected'(0))
# To focus on 'selected' (1), set class_index to 1
# To focus on 'rejected' (0), set class_index to 0

class_index = 0  # For 'rejected' class

# Initialize and compute SHAP values for the specific class
# model.predict_proba returns probabilities for all classes,
# we select the column corresponding to the class_index
explainer = shap.Explainer(lambda X: model.predict_proba(X)[:, class_index], X_scaled)
shap_values = explainer(X_scaled)

# Now you can create the beeswarm plot
shap.plots.beeswarm(shap_values)

"""#Plot Type:
Beeswarm plot.
#Purpose:
This plot shows the SHAP values for each feature and how they impact the model's predictions. Each dot represents a data instance.
The color of the dot indicates the feature value (red for high, blue for low).
The position on the horizontal axis shows the SHAP value, which represents the impact on the model's prediction. Dots further to the right increase the prediction (e.g., probability of being 'selected'), dots to the left decrease the prediction.
The vertical axis groups data points by feature.
The plot is automatically sorted by feature importance in descending order by default.
#Insights:
Feature Importance: The features with the widest spread of dots and the largest SHAP values (both positive and negative) are the most important for the model's predictions.
Feature Impact: The direction (left or right) and color of the dots tell you how a feature value influences the prediction.
Interactions: Clustering of dots might indicate interactions between features.
#Example:
If the 'Transcript_length' feature has a wide spread of dots, it's an important predictor. Red dots on the right side for this feature would mean longer transcripts are associated with higher predicted probabilities for the 'selected' class.
"""

import numpy as np

# Get predicted probabilities for the 'selected' class (assuming class 1)
predictions = model.predict_proba(X_scaled)[:, 1]

# Define thresholds for low, medium, and high predictions
low_threshold = 0.3  # Adjust as needed
high_threshold = 0.7  # Adjust as needed

# Find indices of instances belonging to each category
low_indices = np.where(predictions < low_threshold)[0]
medium_indices = np.where((predictions >= low_threshold) & (predictions <= high_threshold))[0]
high_indices = np.where(predictions > high_threshold)[0]

# Print the number of instances in each category
print("Number of low prediction instances:", len(low_indices))
print("Number of medium prediction instances:", len(medium_indices))
print("Number of high prediction instances:", len(high_indices))

# Print the indices for each category
print("Low prediction indices:", low_indices)
print("Medium prediction indices:", medium_indices)
print("High prediction indices:", high_indices)

# Waterfall plot for a low prediction
shap.plots.waterfall(shap_values[0])  # Replace 0 with the index of a low prediction instance

# Waterfall plot for a medium prediction
shap.plots.waterfall(shap_values[184])  # Replace 5 with the index of a medium prediction instance

# Waterfall plot for a high prediction
shap.plots.waterfall(shap_values[1]) # Replace 10 with the index of a high prediction instance

"""#Plot Type:
 Waterfall plot.
#Purpose:
This plot provides a detailed breakdown of how SHAP values contribute to the model's prediction for a single instance (specified by the index).
The starting point (usually E[f(x)]) is the average model prediction.
Each bar represents a feature, with its length proportional to the SHAP value.
Red bars increase the prediction; blue bars decrease it.
#Insights:
Prediction Explanation: The plot helps understand why a specific instance received a particular prediction.
Feature Contributions: You can see which features had the biggest positive and negative impact on the prediction for this specific instance.
#Example:
A waterfall plot for a specific instance might show that a low 'Resume_length' had a large negative impact, leading to a lower predicted probability of being 'selected.'
"""

# Get feature names from the original data
feature_names = ['Transcript_length', 'Resume_length', 'Job_Description_length']

# Assuming shap_values.values is a NumPy array or a list of arrays
for i, feature in enumerate(feature_names):
    shap.plots.scatter(shap_values[:, i], color=shap_values) # Use the index 'i' instead of the feature name

"""#Plot Type:
 Scatter plot.
#Purpose:
Shows the relationship between a feature's value and its SHAP value.
Each dot represents a data point.
The x-axis is the feature value.
The y-axis is the SHAP value for that feature.
The color of the dot represents the value of another feature (indicated by color=shap_values).
#Insights:
Feature Impact: The slope and direction of the relationship tell you how the feature impacts the prediction.
Interactions: Color variations can reveal interactions, showing how the impact of one feature changes based on the value of another feature.
#Example:
A scatter plot for 'Transcript_length' might show a positive slope, indicating that longer transcripts are associated with higher predictions. Color variations could reveal that this relationship is stronger when 'Resume_length' is also high.
"""

from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

# Get feature names from the original data (before scaling)
feature_names = ['Transcript_length', 'Resume_length', 'Job_Description_length']

# 1D Partial Dependence Plots
fig, ax = plt.subplots(figsize=(15, 5))
PartialDependenceDisplay.from_estimator(
    model,
    X_scaled,  # Pass the scaled data
    features=[0, 1, 2],  # Refer to features by their index in X_scaled
    feature_names=feature_names,  # Provide the original feature names for labeling
    ax=ax
)
plt.show()

"""#Plot Type:
 1D Partial Dependence Plot.

#Purpose:
 This plot shows the relationship between a single feature and the model's prediction, while holding other features constant. It visualizes the marginal effect of a feature on the predicted outcome.

#Insights:

Feature Effect: Observe the trend of the line to understand how changing the feature value affects the prediction. An upward trend indicates a positive relationship (increasing the feature value increases the prediction), while a downward trend indicates a negative relationship.
Feature Importance: Features with steeper slopes generally have a stronger influence on the model's prediction. Flat lines suggest that the feature has little impact in that range of values.
Non-linearity: Curves or non-linear shapes in the line indicate that the feature's effect is not simply additive, and its impact changes depending on its value.
#Example:
 If the line for 'Transcript_length' shows an upward trend, it suggests that longer transcripts are associated with a higher probability of being 'selected', all else being equal. A steeper slope would imply a stronger impact.
"""

import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

# Get feature names from the original data (before scaling)
feature_names = ['Transcript_length', 'Resume_length', 'Job_Description_length']

# 2D Partial Dependence Plot
fig, ax = plt.subplots(figsize=(10, 8))  # Adjust figure size as needed

# Assuming 'Transcript_length' (index 0) and 'Resume_length' (index 1) are the top 2 features
display = PartialDependenceDisplay.from_estimator(
    model,
    X_scaled,
    features=[(0, 1)],  # Specify the indices of the two features for interaction
    feature_names=feature_names,  # Provide original feature names for labeling
    kind='average',  # Use 'average' for the 2D PDP
    contour_kw={'cmap': 'viridis'},  # Customize contour plot colors
    ax=ax,
)

plt.colorbar(display.deciles_vlines_[0][0], label='Average Prediction')  # Add a colorbar, referencing the PDP
plt.show()

"""#Plot Type:
2D Partial Dependence Plot.

#Purpose:
This plot visualizes the interaction effect of two features on the model's prediction. It shows how the relationship between one feature and the prediction changes depending on the value of another feature.

#Insights:

Interaction Effects: Look for areas of different colors or contour lines that are not parallel to the axes. This indicates an interaction effect, where the impact of one feature depends on the value of the other.
Feature Importance: Regions with denser contour lines or larger color gradients indicate areas where the interaction effect is stronger.
Non-linear Interactions: Curved or complex contour patterns suggest a non-linear interaction between the features.
#Example:
In the 2D PDP for 'Transcript_length' and 'Resume_length', if the contour lines are closer together in the area where both features have high values, it suggests that the positive impact of a longer transcript is even stronger when the resume length is also longer, indicating an interaction between these two features.
"""

import numpy as np

# Assuming 'shap_values' is your SHAP values object
base_value = shap_values.base_values[0]  # Get the base value for the first instance
transcript_length_impact = shap_values.values[0, 0]  # Get the impact of 'Transcript_length' for the first instance

# Convert base value to probability
base_probability = np.exp(base_value) / (1 + np.exp(base_value))

# Convert feature impact to probability
transcript_length_probability = np.exp(base_value + transcript_length_impact) / (1 + np.exp(base_value + transcript_length_impact))

print("Base Probability:", base_probability)
print("Transcript Length Probability:", transcript_length_probability)

"""# Overall Feature Importance

Transcript Length and Resume Length appear to be the most important features influencing the model's predictions for whether a candidate is selected or rejected. They have the widest spread of SHAP values in the beeswarm plot and the steepest slopes in the 1D Partial Dependence Plots (PDP).
Job Description Length has a relatively smaller impact compared to the other two features.
# Specific Feature Impacts

Transcript Length: Longer transcripts generally lead to a higher probability of being selected (positive SHAP values, upward trend in PDP).
Resume Length: Longer resumes are generally associated with a higher probability of being selected (positive SHAP values, upward trend in PDP), but the impact might be less pronounced than transcript length.
Job Description Length: The relationship is less clear-cut. There might be an optimal range for job description length where the chances of selection are higher, and very long or very short descriptions might have a negative impact (non-linearity in PDP).
# Interactions

The scatter plots and 2D PDP suggest a potential interaction between Transcript Length and Resume Length. The impact of a longer transcript might be amplified when the resume length is also longer.
There could be other interactions, but they are not as clearly visible in the visualizations.
# Additional Insights from Waterfall Plots

Waterfall plots provide instance-specific explanations. You looked at examples of low, medium, and high prediction instances.
You can see how individual features contributed to those predictions, highlighting the key factors driving the model's decisions.
# In essence

The model indicates that candidates with longer transcripts and resumes have a higher likelihood of being selected. The job description length might play a role, but its relationship is more complex. Additionally, there's evidence of an interaction effect between transcript and resume length.
"""



# -*- coding: utf-8 -*-
"""Embeddings_&_Ensemble_Technique.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1MZkMsXK5et2tdCcrPvCcEYqGbJJS__zc

#Import Libraries and Load Dataset
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from transformers import BertTokenizer, BertModel
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import warnings
warnings.filterwarnings("ignore")

"""#Purpose:
 Imports necessary libraries for data manipulation, model building, and evaluation. Libraries include pandas, numpy, sklearn, transformers, torch, and tensorflow.
#Explanation:
pandas is used for data handling and manipulation.

numpy is used for numerical operations.

sklearn is used for machine learning tasks like model training and evaluation.

transformers is used for BERT embeddings.

torch is the backend for the transformers library.

tensorflow is used for building and training the ANN.
"""

# Load the cleaned dataset
file_path = "combined_cleaned_dataset.csv"  # Update with your file name
df = pd.read_csv(file_path)

df.head()

df.info()

"""#Generate BERT Embeddings"""

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

"""#Purpose:
 Initializes the BERT tokenizer and model for generating embeddings.
#Explanation:
BertTokenizer is used to tokenize the text data.

BertModel is the pre-trained BERT model that will be used to generate embeddings.
"""

def get_bert_embeddings(text_column):
    """
    Generate BERT embeddings for a given text column.
    """
    embeddings = []
    for text in text_column:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        outputs = bert_model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].detach().numpy()
        embeddings.append(cls_embedding.flatten())
    return np.array(embeddings)

"""#Purpose:
Defines a function to generate BERT embeddings for a given text column.
#Explanation:
Takes a text column as input.

Tokenizes the text using the BERT tokenizer.

Generates embeddings using the BERT model.

Returns the embeddings as a NumPy array.

#Bert Embeddings for Transcript
"""

df['Transcript_embeddings'] = list(get_bert_embeddings(df['Transcript']))

"""#Bert Embeddings for Resume"""

df['Resume_embeddings'] = list(get_bert_embeddings(df['Resume']))

"""#Bert Embeddings for Job_Description"""

df['Job_Description_embeddings'] = list(get_bert_embeddings(df['Job Description']))

"""#Purpose:
Applies the get_bert_embeddings function to generate embeddings for the 'Transcript', 'Resume', and 'Job Description' columns and stores them as new columns in the DataFrame.
"""

df.head()

df.info()

unique_decisions = df['decision'].unique()
print("Unique values in 'decision' column:", unique_decisions)

# Assuming 'data' is your pandas DataFrame
df['decision'] = df['decision'].replace({'select': 'selected', 'reject': 'rejected'})

# Assuming 'data' is your pandas DataFrame
unique_decisions = df['decision'].unique()
print("Unique values in 'decision' column:", unique_decisions)

"""#Combine Features and Train-Test Split"""

# Combine embeddings and features
handcrafted_features = ['Transcript_length', 'Resume_length', 'Job_Description_length']
X_embeddings = np.hstack([
    np.vstack(df['Transcript_embeddings']),
    np.vstack(df['Resume_embeddings']),
    np.vstack(df['Job_Description_embeddings']),
    df[handcrafted_features].values
])

# Encode target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['decision'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_embeddings, y, test_size=0.2, random_state=42, stratify=y)

"""#Purpose:
These cells perform data preprocessing steps, including:

Checking and correcting the values in the 'decision' column.

Combining embeddings and other features.

Encoding the target variable
('decision') using LabelEncoder.

Splitting the data into training and testing sets.
"""

print(np.unique(y_test))

"""#Random Forest Model"""

rf = RandomForestClassifier(random_state=42, n_estimators=100)
rf.fit(X_train, y_train)

# Evaluate Random Forest
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

print("Random Forest - Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Random Forest - ROC AUC Score:", roc_auc_score(y_test, y_prob_rf))

"""#Purpose:
Trains and evaluates a Random Forest classifier.
#Explanation:
Initializes the Random Forest model.

Trains the model using the training data.

Evaluates the model's performance using accuracy and ROC AUC score.

#Artificial Neural Network (ANN)
"""

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

# Define ANN model
ann = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# Compile the ANN
ann.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

history = ann.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the ANN
y_prob_ann = ann.predict(X_test).flatten()
y_pred_ann = (y_prob_ann > 0.5).astype(int)

print("ANN - Accuracy:", accuracy_score(y_test, y_pred_ann))
print("ANN - ROC AUC Score:", roc_auc_score(y_test, y_prob_ann))

"""#Purpose:
 Builds, trains, and evaluates an Artificial Neural Network (ANN) model.
#Explanation:
Defines the ANN architecture.
Compiles the model.
Trains the model using the training data.

Evaluates the model's performance using accuracy and ROC AUC score.

#Combine Random Forest and ANN Predictions
"""

combined_predictions = (y_prob_rf + y_prob_ann) / 2
combined_pred_labels = (combined_predictions > 0.5).astype(int)

# Evaluate combined model
print("Combined Model - Accuracy:", accuracy_score(y_test, combined_pred_labels))
print("Combined Model - ROC AUC Score:", roc_auc_score(y_test, combined_predictions))

"""#Purpose:
Combines the predictions from the Random Forest and ANN models and evaluates the combined model's performance.
#Explanation:
Averages the prediction probabilities from both models.

Evaluates the combined model using accuracy and ROC AUC score.
"""

# Save the combined dataset with embeddings for future use
df.to_csv("combined_dataset_with_embeddings.csv", index=False)
print("Dataset with embeddings saved as 'combined_dataset_with_embeddings.xlsx'")