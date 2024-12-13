import streamlit as st 
import altair as alt
import pandas as pd
import numpy as np
import joblib

# Load pipeline
pipeline = joblib.load("fake_news_classifier_pipeline.pkl")

def predict_fakenews(docx):
    results = pipeline.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipeline.predict_proba([docx])
    return results

def get_feature_importance():
    if hasattr(pipeline, 'named_steps') and 'classifier' in pipeline.named_steps:
        model = pipeline.named_steps['classifier']
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
    return None

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
        feature_importances = get_feature_importance()
        
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
        if feature_importances is not None:
            st.subheader("Feature Importance")
            feature_names = pipeline.named_steps['vectorizer'].get_feature_names_out()
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

if __name__ == '__main__':
    main()
