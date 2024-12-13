import streamlit as st 
import altair as alt
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import CountVectorizer

# Load pipeline
pipeline = joblib.load("fake_news_classifier_pipeline.pkl")

# Simulated vocabulary for HashingVectorizer
def get_hashing_vocabulary():
    # Replace this with a precomputed vocabulary matching your HashingVectorizer
    count_vectorizer = CountVectorizer()  # Use same settings as HashingVectorizer
    dummy_texts = ["dummy text for creating vocabulary"]
    count_vectorizer.fit(dummy_texts)
    return count_vectorizer.get_feature_names_out()

def predict_fakenews(docx):
    results = pipeline.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipeline.predict_proba([docx])
    return results

def get_feature_importance_with_hashing():
    try:
        classifier = pipeline.named_steps.get('classifier', None)
        if classifier and hasattr(classifier, 'feature_importances_'):
            feature_importances = classifier.feature_importances_
            # Retrieve approximate feature names from CountVectorizer
            feature_names = get_hashing_vocabulary()
            return feature_names, feature_importances
    except Exception as e:
        print(f"Error retrieving feature importances: {e}")
    return None, None

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
        feature_names, feature_importances = get_feature_importance_with_hashing()
        
        with col1:
            st.success("Original Text")
            st.write(raw_text)
            st.success("Prediction")
            st.write(f"{prediction}")
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
        if feature_names is not None and feature_importances is not None:
            st.subheader("Top Contributing Features")
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importances
            }).sort_values(by='Importance', ascending=False).head(10)

            fig = alt.Chart(importance_df).mark_bar().encode(
                x=alt.X('Importance', title='Feature Importance'),
                y=alt.Y('Feature', sort='-x', title='Top Features'),
                color='Feature'
            )
            st.altair_chart(fig, use_container_width=True)
        else:
            st.warning("Feature importance could not be retrieved.")

if __name__ == '__main__':
    main()
