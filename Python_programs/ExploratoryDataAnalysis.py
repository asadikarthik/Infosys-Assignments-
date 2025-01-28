import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from transformers import pipeline
nltk.download('stopwords')

folder_path = 'recruitment_data'

cleaned_dataframes = []

files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]

text_columns = ['Transcript', 'Resume', 'Job Description', 'Reason for decision']

for col in text_columns:
    if col in df.columns:
        df[col] = df[col].str.lower()

print("Dataset after converting text to lowercase:")
print(df[text_columns].head())

df.drop_duplicates(inplace=True)
df.fillna('Not Specified', inplace=True)

print("Null values in the dataset:")
print(df.isnull().sum())

stop_words = set(stopwords.words('english'))

for col in text_columns:
    if col in df.columns:
        df[col] = df[col].apply(lambda x: ' '.join(word for word in str(x).split() if word not in stop_words))

print("Dataset after removing stop words:")
print(df[text_columns].head())

from nltk.tokenize import word_tokenize
nltk.download('punkt')

for col in text_columns:
    if col in df.columns:
        df[f'{col}_tokens'] = df[col].apply(word_tokenize)

print("Sample tokenized text:")
print(df[[f'{col}_tokens' for col in text_columns]].head())

bow_vectorizer = CountVectorizer(max_features=100)
bow_matrix = bow_vectorizer.fit_transform(df['Transcript'])

bow_df = pd.DataFrame(bow_matrix.toarray(), columns=bow_vectorizer.get_feature_names_out())
print("Bag of Words Representation (First 5 Rows):")
print(bow_df.head())

tfidf_vectorizer = TfidfVectorizer(max_features=100)
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Transcript'])

tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
print("TF-IDF Representation (First 5 Rows):")
print(tfidf_df.head())

df['Transcript_Length'] = df['Transcript'].apply(lambda x: len(str(x)))
df['Resume_Length'] = df['Resume'].apply(lambda x: len(str(x)))

print("Mean Transcript Length:", df['Transcript_Length'].mean())
print("Median Transcript Length:", df['Transcript_Length'].median())
print("Std Transcript Length:", df['Transcript_Length'].std())

print("Mean Resume Length:", df['Resume_Length'].mean())
print("Median Resume Length:", df['Resume_Length'].median())
print("Std Resume Length:", df['Resume_Length'].std())

if 'Role' in df.columns:
    roles = df['Role'].unique()
    print(f"Unique Roles: {roles}")
    for role in roles:
        role_df = df[df['Role'] == role]
        print(f"Role: {role}, Data Points: {len(role_df)}")
else:
    print("Role column not found in the dataset.")

sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))

if 'Decision' in df.columns:
    sns.countplot(data=df, x='Decision', palette='viridis')
    plt.title('Decision Distribution', fontsize=16)
    plt.xlabel('Decision', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.show()
