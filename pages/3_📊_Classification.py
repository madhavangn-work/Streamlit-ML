from streamlit import session_state as ss
import streamlit as st
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
from src.Train_Helper import *
from sklearn.metrics import accuracy_score as acc
import pickle
import os

model_path = 'models/model.pkl'

st.set_page_config(
    layout="wide"
)

st.title("Classification")
st.write("---")

def train_helper(name, tune = False) -> KNeighborsClassifier | DecisionTreeClassifier | SVC | MLPClassifier | RandomForestClassifier:
    """
    Helper function to train the specified classifier with optional hyperparameter tuning.

    Parameters:
    name (str): The name of the classifier to train.
    tune (bool): Whether to perform hyperparameter tuning. Default is False.

    Returns:
    KNeighborsClassifier | DecisionTreeClassifier | SVC | MLPClassifier | RandomForestClassifier: The trained classifier.
    """
    X_train = ss.X_train_pca
    X_train = X_train.select_dtypes(include=[np.number])

    y_train = ss.y_train

    with st.spinner("Training..."):
        if name == "K-Nearest Neighbors":
            clf = do_KNN(X = X_train, y = y_train, tune = tune)
        elif name == "Decision Tree":
            clf = do_DT(X = X_train, y = y_train, tune = tune)
        elif name == "Support Vector Machine":
            clf = do_SVM(X = X_train, y = y_train, tune = tune)
        elif name == "Multi-Layer Perceptron":
            clf = do_MLP(X = X_train, y = y_train, tune = tune)
        elif name == "Random Forest Classifier":
            clf = do_RFC(X = X_train, y = y_train, tune = tune)

        X_test = ss.X_test_pca.select_dtypes(include=[np.number])
        y_test = ss.y_test

        y_pred = clf.predict(X_test)

        accuracy_score =  acc(y_test, y_pred)

        if "accuracy_score" not in ss:
            ss.accuracy_score = accuracy_score

        ss.accuracy_score = accuracy_score
    
    st.write(f"The trained model was able to predict the test dataset with an accuracy of {ss.accuracy_score * 100}%")

    return clf


def main():

    """
    Main function to run the Streamlit app.
    Allows the user to select a classification model and choose whether to perform hyperparameter tuning.
    """

    st.write("## Select the model you want to use to classify")

    model_name = st.selectbox(
        label="Select",
        options=[
            "K-Nearest Neighbors",
            "Decision Tree",
            "Support Vector Machine",
            "Multi-Layer Perceptron",
            "Random Forest Classifier"
        ]
    )

    if "model_name" not in ss:
        ss.model_name = model_name
        
    ss.model_name = model_name

    hyper_param_tuning = st.radio("Hyperparameter tune the model? This will take a significant amount of time", ["Yes", "No"], index=1)

    if hyper_param_tuning not in ss:
        ss.hyper_param_tuning = hyper_param_tuning

    ss.hyper_param_tuning = hyper_param_tuning

    if ss.hyper_param_tuning == "Yes":
        fin_conf = st.radio("Are you Sure?", ["Yes","No"], index=1)
        if "fin_conf" not in ss:
            ss.fin_conf = fin_conf

        ss.fin_conf = fin_conf

        if fin_conf == "Yes":
            clf = train_helper(name=ss.model_name, tune=True)
        else:
            clf = train_helper(name=ss.model_name)


    else:
        clf = train_helper(name=ss.model_name)

    if "clf" not in ss:
        ss.clf = clf
    
    ss.clf = clf

    with open(model_path, "wb") as f:
        pickle.dump(ss.clf, f)

    # Check if the model file exists
    if os.path.exists(model_path):
        # Provide a download button
        with open(model_path, 'rb') as model_file:
            model_data = model_file.read()
            st.download_button(
                label="Download model.pkl",
                data=model_data,
                file_name="model.pkl",
                mime="application/octet-stream"
            )
    

if __name__ == '__main__':
    main()