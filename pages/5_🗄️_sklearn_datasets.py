import streamlit as st
from streamlit import session_state as ss
from sklearn import datasets
import pandas as pd


def mkdata(df):
    # Create a DataFrame
    data = pd.DataFrame(data=df.data, columns=df.feature_names)
    data['target'] = df.target
    return data


def get_data(name):

    if name == "Iris":
        data = datasets.load_iris()
    if name == "Digits":
        data = datasets.load_digits()
    if name == "Wine":
        data = datasets.load_wine()
    if name == "Breast Cancer":
        data = datasets.load_breast_cancer()

    data = mkdata(data)

    if "uploaded_data" in ss:
        st.warning("Replacing uploaded data with this one...")
        ss.data_change = "Yes"
        ss.uploaded_data = data
        st.write(ss.uploaded_data)
    else:
        ss.data_change = "No"
        ss.uploaded_data = data
        st.write(ss.uploaded_data)

    st.success("Data Loaded")



def main():
    list = [
        "Iris",
        "Digits",
        "Wine",
        "Breast Cancer",
    ]

    dataset_name = st.selectbox("Which Dataset do you want to import?", options=list, index=0)

    if "dataset_name" not in ss:
        ss.dataset_name = dataset_name

    ss.dataset_name = dataset_name

    get_data(ss.dataset_name)


if __name__=="__main__":
    main()

    st.write("---")
    st.write("All datasets here are sources from Scikit-Learn's Datasets module")
    st.write("For more information refer to their [website](https://scikit-learn.org/stable/datasets/toy_dataset.html)")