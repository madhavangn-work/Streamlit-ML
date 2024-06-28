import streamlit as st
import pandas as pd
from streamlit import session_state as ss



# Function to configure Streamlit page settings
def configure_page() -> None:
    st.set_page_config(
        page_title="ML App with Streamlit",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

# Main function to render the home page
def main() -> None:
    configure_page()

    st.title("ML Application with Streamlit")

    st.markdown("""
                I am creating this app to make ML more accessible for people. While this is not a 
                ground breaking app by any means
                The objective of this app is to enable people to do ML more easily on data 
                that is available to them. I will explain 
                every part of this process in lay person terms to guide people along the way.

                Let's break down classification with a simple real-world example, and then we’ll 
                talk about the target variable in both regression and classification problems.
                            
                ## Classification:
                Classification is a type of machine learning task where the goal is to categorize 
                data into different classes or groups. 
                For example, imagine you have a bunch of emails, and you want to automatically sort 
                them into two categories:
                "Spam" and "Not Spam." This is a classification problem because you’re assigning each 
                email to one of two specific classes.

                ### Real-World Example: Classifying Emails

                - **Data:** You have a dataset of emails.
                - **Features:** These are the details about each email, 
                like the subject line, the sender, and the content.
                - **Target Variable:** This is what you’re trying to predict. 
                In this case, the target variable is whether the email is "Spam" or "Not Spam."

                A Classification model's aim is to learn from the data and the features to classify new 
                emails correctly.

                ### Target Variable:
                A target variable is the specific piece of information that you want your machine learning model 
                to predict. It's the "answer" or "outcome" you’re looking to find based on the data you have.

                ### Target Variable in Regression and Classification

                #### Classification:    
                - Target Variable: Categorical (like "Spam" or "Not Spam").
                - Example: If you’re classifying fruit based on their features (color, size, shape), your 
                target variable might be the type of fruit (like "Apple," "Banana," "Orange").

                #### Regression:
                - Target Variable: Continuous (a number that can take any value).
                - Example: If you’re predicting house prices based on features (number of bedrooms, square 
                footage, location), your target variable is the price of the house, which is a continuous number.


                          
    """)

    uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])

    if uploaded_file is not None:
        # Read the data and store it in session state
        data = pd.read_csv(uploaded_file)
        ss.uploaded_data = data
        ss.data_change = None

        # Display summary of uploaded data
        st.subheader("Summary of Uploaded Data:")
        st.write(data.head())
        st.success('File successfully uploaded!')
        st.page_link("pages/1_🧹_Clean_Data.py", label="Continue to Clean Data", icon="🧹")
    st.markdown("---")
    st.write("Developed by Madhavan Namboothiri")
    

if __name__ == '__main__':
    main()