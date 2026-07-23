import streamlit as st
from src.predict import predict

st.title("SMS Spam Detector")
with st.sidebar:
    st.header("About")
    st.write("This app detects SMS spam using a machine learning model trained on the SMS Spam Collection dataset.")
    st.write("Built by Awurabena Affainie and Rawuf Adebayo.")
st.write("Enter a message below to check if it's spam or ham.")

user_input = st.text_area("Message", height=100)

if st.button("Check"):
    cleaned = user_input.strip()

    if cleaned == "":
        st.warning("Please enter a message before checking.")
    elif len(cleaned) < 3:
        st.warning("That message seems too short to check meaningfully.")
    else:
        result = predict(cleaned)
        if result == "spam":
            st.error(f"🚨 This looks like: **SPAM**")
        else:
            st.success(f"✅ This looks like: **HAM (not spam)**")

        with st.expander("See message analyzed"):
            st.write(cleaned)