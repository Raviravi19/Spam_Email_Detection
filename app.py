import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the data
raw_mail_data = pd.read_csv('C:\\Users\\ravi shankar\\OneDrive\\Documents\\Spam_Email_Detection\\mail_data (1).csv')

# Preprocess the data
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)), '')
mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1

# Separate the data as texts and labels
X = mail_data['Message']
Y = mail_data['Category']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# Transform the text data to feature vectors
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

# Train the model
model = LogisticRegression()
model.fit(X_train_features, Y_train)

# Streamlit app interface
st.header("Spam Email Detection")

uploaded_file = st.file_uploader("C:\\Users\\ravi shankar\\OneDrive\\Documents\\Spam_Email_Detection\\mail_data (1).csv")
if uploaded_file is not None:
    raw_mail_data = pd.read_csv(uploaded_file)
    st.write(raw_mail_data.head())
else:
    st.error("Please upload the CSV file.")

# Input text box for the user to enter an email message
input_mail = st.text_area("Enter the email message:")

if st.button('Predict'):
    if input_mail:
        # Convert text to feature vectors
        input_data_features = feature_extraction.transform([input_mail])

        # Make prediction
        prediction = model.predict(input_data_features)

        if prediction[0] == 1:
            st.success("Ham mail")
        else:
            st.error("Spam mail")
    else:
        st.warning("Please enter an email message.")
