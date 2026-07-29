import streamlit as st

from src.predict import log_report, predict


st.set_page_config(page_title="SMS Fraud Detector", page_icon="🛡️", layout="centered")

if "sms_input" not in st.session_state:
    st.session_state.sms_input = ""
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None


st.title("SMS Fraud & Spam Detector")
st.caption("Detect suspicious SMS messages with a trained model and quick explainability cues.")

with st.sidebar:
    st.header("About")
    st.write(
        "This app combines a machine learning model with keyword signals to flag SMS spam and mobile money scams."
    )
    st.write("Built by Awurabena Affainie and Rawuf Adebayo.")

st.write("Enter a message below to check whether it looks like spam, fraud, or a normal message.")
st.markdown("**Try an example:**")

example_fraud = (
    "Congratulations! Your MTN MoMo account has won GHS 500. Send your PIN to 024XXXXXXX to claim your reward."
)
example_legit = "Hey, are we still meeting at 3pm today? Let me know if anything changes."
example_momo = (
    "ALERT: Your Vodafone Cash account will be suspended. Call 055XXXXXXX and provide your PIN to verify."
)

col1, col2, col3 = st.columns(3)
if col1.button("🚨 MoMo Scam"):
    st.session_state.sms_input = example_fraud
if col2.button("✅ Legit SMS"):
    st.session_state.sms_input = example_legit
if col3.button("⚠️ Vodafone Scam"):
    st.session_state.sms_input = example_momo

sms_input = st.text_area(
    "Message",
    height=140,
    placeholder="e.g. You have won a MoMo prize! Send your PIN to claim...",
    key="sms_input",
)

if st.button("Analyse Message", type="primary", use_container_width=True):
    cleaned = sms_input.strip()

    if cleaned == "":
        st.warning("Please enter a message before checking.")
        st.session_state.analysis_result = None
    elif len(cleaned) < 3:
        st.warning("That message seems too short to check meaningfully.")
        st.session_state.analysis_result = None
    else:
        prediction, confidence, keywords = predict(cleaned)
        st.session_state.analysis_result = {
            "message": cleaned,
            "prediction": prediction,
            "confidence": confidence,
            "keywords": keywords,
        }

result = st.session_state.analysis_result
if result and result["message"] == sms_input.strip():
    score = result["confidence"] * 100

    st.divider()
    if score >= 70 or result["prediction"] == "spam":
        st.error(f"Likely Fraud / Spam - Risk Score: {score:.1f}%")
    elif score >= 40:
        st.warning(f"Suspicious - Risk Score: {score:.1f}%")
    else:
        st.success(f"Looks Legitimate - Risk Score: {score:.1f}%")

    st.progress(int(score))

    if result["keywords"]:
        st.markdown("**Fraud indicators found in this message:**")
        st.markdown(" ".join([f"`{keyword}`" for keyword in result["keywords"]]))
    else:
        st.markdown("**No common fraud keywords detected.**")

    with st.expander("See message analyzed"):
        st.write(result["message"])

    st.divider()
    st.markdown("**Was this result wrong? Help improve the model:**")
    if st.button("Report this message as fraud"):
        log_report(result["message"], result["prediction"])
        st.success("Thank you. This message has been logged for review.")
elif result:
    st.info("The current text differs from the last analysis. Click Analyse Message again to refresh the result.")