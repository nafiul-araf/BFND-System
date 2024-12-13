import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import joblib
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load pipeline
pipeline = joblib.load("fake_news_classifier_pipeline.pkl")

def predict_fakenews(docx):
    results = pipeline.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipeline.predict_proba([docx])
    return results

def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    return wordcloud

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

        with col1:
            st.success("Original Text")
            st.write(raw_text)
            st.success("Prediction")
            st.write(f"{prediction}")
            st.write(f"Confidence: {np.max(probability):.2f}")

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

        # Word Cloud Visualization
        if raw_text:
            st.subheader("Word Cloud from Your Text")
            wordcloud = generate_wordcloud(raw_text)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            st.pyplot(plt)

if __name__ == '__main__':
    main()
