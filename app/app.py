import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# --- Load saved model and vectorizer ---
with open("../models/logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("../vectorizer/count_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# --- Preprocessing function (same as training) ---
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'rt\s+', '', text)                 # remove 'RT'
    text = re.sub(r'http\S+|www\S+', ' url ', text)   # replace urls
    text = re.sub(r'@\w+', ' user ', text)            # replace mentions
    text = re.sub(r'[^a-z\s]', '', text)              # keep only letters and spaces
    tokens = [word for word in text.split() if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)

# --- Class label mapping ---
label_map = {
    0: ("🟥 Hate Speech", "This comment contains hateful or discriminatory language."),
    1: ("🟧 Offensive", "This comment is offensive but not necessarily hate speech."),
    2: ("🟩 Neutral", "This comment looks safe and neutral.")
}


# --- Streamlit UI ---
st.title("📝 Tweets/Comments Classifier")
st.write("Classify tweets or comments into **Hate Speech, Offensive, or Neutral**.")

# User input
user_input = st.text_area("Enter a tweet or comment:")

if st.button("Classify"):
    if user_input.strip():
        clean_text = preprocess(user_input)
        X_new = vectorizer.transform([clean_text])
        prediction = model.predict(X_new)[0]

        # Friendly output
        label, description = label_map.get(prediction, ("Unknown", "No description available."))

        st.subheader("Prediction")
        st.markdown(f"**Category:** {label}")
        st.info(description)
    else:
        st.warning("Please enter some text before classifying.")