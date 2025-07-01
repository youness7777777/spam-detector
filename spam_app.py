import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --- Custom Styling ---
st.markdown(
    """
    <style>
        body {
            background: linear-gradient(135deg, #f8fafc 0%, #e0e7ff 100%) !important;
        }
        .main {
            background-color: rgba(255,255,255,0.85) !important;
            border-radius: 16px;
            padding: 2rem 2rem 1rem 2rem;
            box-shadow: 0 4px 24px rgba(0,0,0,0.07);
        }
        h1, .stTitle {
            color: #111 !important;
            font-family: 'Segoe UI', 'Roboto', sans-serif;
            font-weight: 700;
        }
        .stTextArea textarea {
            background: #f1f5f9 !important;
            border-radius: 8px !important;
            border: 1.5px solid #a5b4fc !important;
            font-size: 1.1rem !important;
            color: #111 !important;
        }
        .stButton button {
            background: linear-gradient(90deg, #e5e7eb 0%, #f3f4f6 100%) !important;
            color: #111 !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            font-size: 1.1rem !important;
            padding: 0.5rem 2rem !important;
            margin-top: 0.5rem;
        }
        .stButton button:hover {
            background: linear-gradient(90deg, #d1d5db 0%, #e5e7eb 100%) !important;
        }
        .result-box {
            border-radius: 10px;
            padding: 1rem;
            margin-top: 1.5rem;
            font-size: 1.2rem;
            font-weight: 600;
            text-align: center;
        }
        .spam {
            background: #fee2e2;
            color: #b91c1c;
            border: 2px solid #f87171;
        }
        .not-spam {
            background: #dcfce7;
            color: #166534;
            border: 2px solid #4ade80;
        }
        .logo {
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 150px;
            margin-bottom: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Logo/Header ---
st.markdown(
    '<img src="https://tse3.mm.bing.net/th/id/OIP.bxae4HqXp4__1YfpzFsaXgHaEN?r=0&cb=thvnext&rs=1&pid=ImgDetMain&o=7&rm=3" class="logo" alt="Spam Logo"/>',
    unsafe_allow_html=True
)

# Title of the app
st.title("Spam Email Detection App")

# Cache loading data for performance
@st.cache_data
def load_data():
    df = pd.read_csv("mail_data.csv")
    df = df.where(pd.notnull(df), '')  # Replace NaN with empty string
    df['Category'] = df['Category'].replace({'spam': 0, 'ham': 1}).astype(int)
    return df

data = load_data()

# Prepare features and labels
X = data['Message']
Y = data['Category']

# Split into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=42
)

# Vectorize text data using TF-IDF
vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train_features, Y_train)


# User input section
st.subheader("Test Your Email")

user_input = st.text_area("Enter the email content here:")

if st.button("Detect"):
    if user_input.strip() == "":
        st.warning("Please enter some email content to analyze.")
    else:
        input_features = vectorizer.transform([user_input])
        prediction = model.predict(input_features)
        if prediction[0] == 0:
            st.markdown(
                '<div class="result-box spam">🚫 This is <b>SPAM</b>!</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="result-box not-spam">✅ This is <b>NOT spam</b>.</div>',
                unsafe_allow_html=True
            )
