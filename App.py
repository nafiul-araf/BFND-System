import streamlit as st 
import altair as alt
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Load pipeline
pipeline = joblib.load("fake_news_classifier_pipeline.pkl")

def predict_fakenews(docx):
    results = pipeline.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipeline.predict_proba([docx])
    return results

def get_shap_explanation(docx):
    explainer = shap.Explainer(pipeline)
    shap_values = explainer([docx])
    return shap_values

news_emoji_dict = {"Fake": "🛛", "Real": "✅"}

def main():
    st.title("Bangla Fake News Classifier")
    st.subheader("Input Text")
    
    with st.form(key='fakenews_clf_form'):
        raw_text = st.text_area("Type Here")
        submit_text = st.form_submit_button(label='Submit')
    
    if submit_text:
        col1, col2 = st.columns(2)
        # Apply functions
        prediction = predict_fakenews(raw_text)
        probability = get_prediction_proba(raw_text)
        shap_values = get_shap_explanation(raw_text)
        
        with col1:
            st.success("Original Text")
            st.write(raw_text)
            st.success("Prediction")
            news_icon = news_emoji_dict[prediction]
            st.write(f"{prediction}: {news_icon}")
            st.write(f"Confidence: {np.max(probability)}")
        
        with col2:
            st.success("Prediction Probability")
            proba_df = pd.DataFrame(probability, columns=pipeline.classes_)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["news type", "probability"]

            fig = alt.Chart(proba_df_clean).mark_bar().encode(
                x='news type',
                y='probability',
                color='news type'
            )
            st.altair_chart(fig, use_container_width=True)

        # Feature importance section
        st.subheader("Feature Importance")
        st.set_option('deprecation.showPyplotGlobalUse', False)
        shap.plots.bar(shap_values[0], show=False)
        st.pyplot()

if __name__ == '__main__':
    main()
