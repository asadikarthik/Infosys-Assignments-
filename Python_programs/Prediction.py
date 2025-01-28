
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

predict_df = pd.read_excel('prediction_data.xlsx')

predict_df.head()

text_columns = ['Transcript', 'Resume', 'Job Description', 'Reason for decision']

for col in text_columns:
    if col in predict_df.columns:
        predict_df[col] = predict_df[col].str.lower()

predict_df.drop_duplicates(inplace=True)
predict_df.fillna('Not Specified', inplace=True)

print("Null values in combined dataset:")
predict_df.isnull().sum()

prediction_df['Role'].unique()

unique_count = prediction_df.groupby('Role')['ID'].count()
unique_count

tfidf_vectorizer = TfidfVectorizer()
all_text = pd.concat([predict_df[col] for col in text_columns if col in predict_df.columns])
tfidf_vectorizer.fit(all_text)

tfidf_transcript = tfidf_vectorizer.transform(predict_df['Transcript'])
tfidf_resume = tfidf_vectorizer.transform(predict_df['Resume'])
tfidf_job_desc = tfidf_vectorizer.transform(predict_df['Job Description'])

predict_df['resume_job_similarity'] = [cosine_similarity(tfidf_resume[i], tfidf_job_desc[i])[0][0] for i in range(len(predict_df))]
predict_df['transcript_job_similarity'] = [cosine_similarity(tfidf_transcript[i], tfidf_job_desc[i])[0][0] for i in range(len(predict_df))]

features = ['resume_job_similarity', 'transcript_job_similarity']
predict_df['decision'] = np.random.choice(['selected', 'rejected'], size=len(predict_df))

X = predict_df[features]
y = predict_df['decision']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(X_train, y_train)

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(random_forest, f)

X_new = predict_df[features]
predictions = random_forest.predict(X_new)
predict_df['predicted_decision'] = predictions

predict_df[['ID', 'predicted_decision']]

selected_count = predict_df['predicted_decision'].value_counts()['selected']
rejected_count = predict_df['predicted_decision'].value_counts()['rejected']

print(f"Selected: {selected_count}")
print(f"Rejected: {rejected_count}")

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

sender_email = "uppariupendra@gmail.com"
sender_password = "eseuxrzxutsyjwse"

def send_email(receiver_email, subject, body):
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            text = msg.as_string()
            server.sendmail(sender_email, receiver_email, text)
            print(f"Email sent to {receiver_email}")
    except Exception as e:
        print(f"Error sending email to {receiver_email}: {e}")

if __name__ == "__main__":
    receiver_email = "21r21a66k0@mlrinstitutions.ac.in"
    subject = "Congratulations! You're selected for the next round."
    body = """Dear Candidate,

We are pleased to inform you that you have been selected for the next round of the interview process for the Software Engineer role.

Further instructions will be provided shortly.

Sincerely,
[Your Name/Company]
"""
    send_email(receiver_email, subject, body)

import smtplib
import re
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

sender_email = "uppariupendra@gmail.com"
sender_password = "eseuxrzxutsyjwse"

def send_email(receiver_email, subject, body):
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        text = msg.as_string()
        server.sendmail(sender_email, receiver_email, text)
        print(f"Email sent to {receiver_email}")
    except Exception as e:
        print(f"Error sending email: {e}")
    finally:
        pass  

for index, row in predict_df.iterrows():
    resume_text = row['Resume']

    email_match = re.search(r'[\w\.-]+@[\w\.-]+', resume_text)
    if email_match:
        candidate_email = email_match.group(0)
    else:
        print(f"Email not found for ID: {row['ID']}, skipping...")
        continue

    decision = row['predicted_decision']

    if decision == 'selected':
        subject = "Congratulations! You're selected for the next round."
        body = f"Dear {row['Name']},\n\nWe are pleased to inform you that you have been selected for the next round of the interview process for the {row['Role']} role. [Provide further instructions here].\n\nSincerely,\n[Your Name/Company]"
    elif decision == 'rejected':
        subject = "Update on your application"
        body = f"Dear {row['Name']},\n\nThank you for your interest in the {row['Role']} role. We appreciate you taking the time to apply. While your qualifications were impressive, we have decided to move forward with other candidates. We wish you the best in your job search.\n\nSincerely,\n[Your Name/Company]"
    else:
        continue

    send_email(candidate_email, subject, body)
