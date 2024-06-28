import streamlit as st
import pandas as pd
from sklearn.impute import SimpleImputer
from streamlit import session_state as ss


# Function to clean data
def clean_data(data):
    st.write("---")

    # Display original dataset
    st.write("## Original Dataset:")
    st.write(data.head())

    st.write("---")

    if data.isnull().sum().any():
        st.write("## Missing Values Detected!")
        st.write(data.isnull().sum())

        st.write("---")

        # Ask user how to handle missing values
        option = st.radio(
            "Choose how to handle missing values:", # Button Label
            (
                "Remove rows with missing values", # Option 1
                "Impute missing values" # Option 2
            ),
            index=None # Make sure nothing is selected as of now
        )

        if option == "Impute missing values":
            imputation_choice = st.radio(
                "Choose how to impute missing values for numerical columns:", # Radio Label
                (
                    "Mean", # Radio Button 1
                    "Median", # Radio Button 2
                    "Maximum" # Radio Button 3
                ),
                index=None # Make sure nothing is selected as of now
            )

        if st.button("Start Imputation"):
            if option == "Remove rows with missing values":
                data_cleaned = data.dropna()
                st.write("Dataset after removing rows with missing values:")
                st.write(data_cleaned.head())
                ss.cleaned_data = data

            elif option == "Impute missing values":
                numerical_columns = data.select_dtypes(include=['number']).columns

                if imputation_choice == "Mean":
                    imputer = SimpleImputer(strategy='mean')
                elif imputation_choice == "Median":
                    imputer = SimpleImputer(strategy='median')
                elif imputation_choice == "Maximum":
                    imputer = SimpleImputer(
                        strategy='constant',
                        fill_value=data[numerical_columns].max()
                    )

                data[numerical_columns] = imputer.fit_transform(data[numerical_columns])

                st.write("Dataset after imputing missing values:")
                st.write(data.head())
                ss.cleaned_data = data

    else:
        st.write("No missing values found!")
        ss.cleaned_data = data

# Main function to render the clean data page
def main(data):
    # Check if the data has been cleaned already
    if 'cleaned_data' in ss:
        if ss.data_change == "Yes":
            ss.data_change = "No"
            clean_data(data)
        else:
            st.write("The data has already been cleansed.")
            st.write("## Cleaned Dataset:")
            st.write(ss.cleaned_data.head())
    else:
        clean_data(data)

    st.page_link("pages/2_🔍_Preprocess_Data.py", label="Proceed to Preprocess the data", icon="🔍")

if __name__ == '__main__':
    st.title("Clean Data")

    if "uploaded_data" not in ss:
        st.warning(f"Go to homepage and upload csv there to start...")
        st.write("---")
        st.write("Link to Homepage or Dataset Page:")
        st.page_link("1_🏠_Home.py", label="Homepage", icon="🏠")
        st.page_link("pages/5_🗄️_sklearn_datasets.py", label="sklearn Datasets", icon="🗄️")
        st.stop()
    else:

        uploaded_data = ss.uploaded_data

        if uploaded_data is not None:
            main(uploaded_data)

    st.markdown("---")
    st.write("Developed by Madhavan Namboothiri")