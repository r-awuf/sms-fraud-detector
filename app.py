import streamlit as st
from src.predict import predict

st.title("SMS Spam Detector")
st.write("Enter a message below to check if it's spam or ham.")

user_input = st.text_area("Message")

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        result = predict(user_input)
        if result == "spam":
            st.error(f"This looks like: **{result.upper()}**")
        else:
            st.success(f"This looks like: **{result.upper()}**")
