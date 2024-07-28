import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Load the pipeline and MLFlow model
pipeline_grid_xgb = joblib.load("model/pipeline_grid_xgb.pkl")
model = joblib.load("model/model.pkl")


# Data Preparation
def data_preparation(df):
    # Removing unnecessary columns
    df = df.drop(['mailchimp_id', 'user_full_name', 'user_email', 'optin_time', 'email_provider'], axis=1)

    categorical_features=['country_code']

    ordinal_features = {
        'member_rating': ["1", "2", "3", "4", "5"]
    }

    numeric_features = ['tag_count',
                        'tag_count_by_optin_day',
                        'tag_aws_webinar',
                        'tag_learning_lab',
                        'tag_learning_lab_05',
                        'tag_learning_lab_09',
                        'tag_learning_lab_11',
                        'tag_learning_lab_12',
                        'tag_learning_lab_13',
                        'tag_learning_lab_14',
                        'tag_learning_lab_15',
                        'tag_learning_lab_16',
                        'tag_learning_lab_17',
                        'tag_learning_lab_18',
                        'tag_learning_lab_19',
                        'tag_learning_lab_20',
                        'tag_learning_lab_21',
                        'tag_learning_lab_22',
                        'tag_learning_lab_23',
                        'tag_learning_lab_24',
                        'tag_learning_lab_25',
                        'tag_learning_lab_26',
                        'tag_learning_lab_27',
                        'tag_learning_lab_28',
                        'tag_learning_lab_29',
                        'tag_learning_lab_30',
                        'tag_learning_lab_31',
                        'tag_learning_lab_32',
                        'tag_learning_lab_33',
                        'tag_learning_lab_34',
                        'tag_learning_lab_35',
                        'tag_learning_lab_36',
                        'tag_learning_lab_37',
                        'tag_learning_lab_38',
                        'tag_learning_lab_39',
                        'tag_learning_lab_40',
                        'tag_learning_lab_41',
                        'tag_learning_lab_42',
                        'tag_learning_lab_43',
                        'tag_learning_lab_44',
                        'tag_learning_lab_45',
                        'tag_learning_lab_46',
                        'tag_learning_lab_47',
                        'tag_time_series_webinar',
                        'tag_webinar',
                        'tag_webinar_01',
                        'tag_webinar_no_degree',
                        'tag_webinar_no_degree_02',
                        'optin_days']
    updated_df = df[[*categorical_features, *ordinal_features.keys(), *numeric_features]]
    return updated_df

# Streamlit interface
def main():
    st.title("Lead Scoring App")
    
    # App introduction
    st.markdown("Welcome to the Lead Scoring App! Upload a CSV file and adjust the threshold to classify leads into Hot and Cold categories.")

    # Add a button
    st.button("Lead Scoring")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        # Read the uploaded file
        original_df = pd.read_csv(uploaded_file)
        # Process uploaded file
        df = data_preparation(original_df)

        # Process the data using the pipeline's transformers
        X_processed = pipeline_grid_xgb[:-1].transform(df)


        def get_predictions(df):
            predictions = model.predict_proba(df)[:, 1]
            return predictions

        predictions = get_predictions(df)

        # Add predictions to the DataFrame
        df['probability'] = predictions

        # Define threshold and apply labels
        threshold = st.slider("Threshold", min_value=0.0, max_value=1.0, step=0.01, value=0.8)
        df['Status'] = df['probability'].apply(lambda x: 'hot-lead' if x > threshold else 'cold-lead')

        # Line chart for hot and cold leads
        hot_leads_count = df[df['Status'] == 'hot-lead'].shape[0]
        cold_leads_count = df[df['Status'] == 'cold-lead'].shape[0]
        lead_counts = pd.DataFrame({'Leads': ['Hot Leads', 'Cold Leads'], 'Count': [hot_leads_count, cold_leads_count]})
        fig = px.bar(lead_counts, x='Leads', y='Count', color='Leads', labels={'Count': 'Number of Leads'})
        st.plotly_chart(fig)

        # Add Probability and Status from prediction to original_df
        original_df['probability'] = df['probability']
        original_df['Status'] = df['Status']

        # Radio button for user to select hot or cold leads
        selection = st.radio("Select Leads Type", ["Hot Leads", "Cold Leads"])

        # Show selected columns from original_df based on user's selection
        if selection == "Hot Leads":
            selected_df = original_df[original_df['Status'] == 'hot-lead'][['mailchimp_id', 'user_full_name', 'user_email', 'country_code', 'probability', 'Status']]
            st.write(selected_df)

            # Show statistics for hot leads
            st.subheader("Statistics for Hot Leads")
            st.write(f"Total Hot Leads: {hot_leads_count}")
            # Graph grouped by country code for hot leads
            hot_leads_by_country = original_df[original_df['Status'] == 'hot-lead'].groupby('country_code').size().reset_index(name='Count')
            fig_hot = px.bar(hot_leads_by_country, x='country_code', y='Count', color='country_code', labels={'Count': 'Number of Leads'}, title='Hot Leads by Country')
            st.plotly_chart(fig_hot)

        elif selection == "Cold Leads":
            selected_df = original_df[original_df['Status'] == 'cold-lead'][['mailchimp_id', 'user_full_name', 'country_code', 'probability', 'Status']]
            st.write(selected_df)

            # Show statistics for cold leads
            st.subheader("Statistics for Cold Leads")
            st.write(f"Total Cold Leads: {cold_leads_count}")

            # Graph grouped by country code for cold leads
            cold_leads_by_country = original_df[original_df['Status'] == 'cold-lead'].groupby('country_code').size().reset_index(name='Count')
            fig_cold = px.bar(cold_leads_by_country, x='country_code', y='Count', color='country_code', labels={'Count': 'Number of Leads'}, title='Cold Leads by Country')
            st.plotly_chart(fig_cold)

        # Download button for predictions
        csv = original_df.to_csv(index=False)
        st.download_button("Download Predictions", csv, "Lead_Scoring.csv", "text/csv")

if __name__ == "__main__":
    main()
