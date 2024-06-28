import streamlit as st
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
import numpy as np
from sklearn.decomposition import PCA
from streamlit import session_state as ss



def traintestsplit(data) -> None:
    target_variable = st.selectbox("Select target Column Name from the above list", data.columns)
    X_variables = [col for col in data.columns if col != target_variable]
    X = data[X_variables]
    y = data[target_variable]
    pct = st.slider("Percentage of training dataset", 1, 99, 80, step=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=pct / 100.0, random_state=42)

    if 'target_variable' not in ss:
        ss.target_variable = target_variable
    if 'X_train' not in ss:
        ss.X_train = X_train
    if 'X_test' not in ss:
        ss.X_test = X_test
    if 'y_train' not in ss:
        ss.y_train = y_train
    if 'y_test' not in ss:
        ss.y_test = y_test

    # Save to session state dynamically
    ss.target_variable = target_variable
    ss.X_train = X_train
    ss.X_test = X_test
    ss.y_train = y_train
    ss.y_test = y_test


def scaler(scaler_type) -> None:
    # Identify numerical columns
    numerical_columns = ss.X_train.select_dtypes(include=[np.number]).columns.tolist()

    if scaler_type == 'StandardScaler':
        scaler = StandardScaler()
    elif scaler_type == 'MinMaxScaler':
        scaler = MinMaxScaler()
    elif scaler_type == 'Normalizer':
        scaler = Normalizer()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")

    # Fit the scaler on numerical columns of X_train and transform X_train
    X_train_scaled = ss.X_train.copy()
    X_train_scaled[numerical_columns] = scaler.fit_transform(X_train_scaled[numerical_columns])

    # Transform numerical columns of X_test using the fitted scaler
    X_test_scaled = ss.X_test.copy()
    X_test_scaled[numerical_columns] = scaler.transform(X_test_scaled[numerical_columns])

    # Store scaled data in session state
    if "X_train_scaled" not in ss:
        ss.X_train_scaled = X_train_scaled
    if "X_test_scaled" not in ss:
        ss.X_test_scaled = X_test_scaled

    # Update session state with scaled data
    ss.X_train_scaled = X_train_scaled
    ss.X_test_scaled = X_test_scaled


def PriComAna(num_comp):
    numerical_columns = ss.X_train_scaled.select_dtypes(include=[np.number]).columns.tolist()
    non_numerical_columns = ss.X_train_scaled.select_dtypes(exclude=[np.number]).columns.tolist()

    pc = PCA(num_comp)

    X_train_pca = ss.X_train_scaled.copy()
    pcdf_train = pc.fit_transform(X_train_pca[numerical_columns])

    X_test_pca = ss.X_test_scaled.copy()
    pcdf_test = pc.fit_transform(X_test_pca[numerical_columns])

    # Store scaled data in session state
    if "X_train_pca" not in ss:
        ss.X_train_pca = pcdf_train
    if "X_test_pca" not in ss:
        ss.X_test_pca = pcdf_test

    # Update session state with scaled data
    ss.X_train_pca = pd.concat([pd.DataFrame(pcdf_train), pd.DataFrame(ss.X_train_scaled[non_numerical_columns]).reset_index(drop=True)], axis=1)
    ss.X_test_pca = pd.concat([pd.DataFrame(pcdf_test), pd.DataFrame(ss.X_test_scaled[non_numerical_columns]).reset_index(drop=True)], axis=1)

    st.write(f'{(np.cumsum(pc.explained_variance_ratio_)[-1]) * 100: .2f}% variance of data explained')



def preprocess_data(data):

    st.write("## Original Data:")
    st.write(data.head())

    st.write("## Train Test Split")

    tts = st.checkbox("Let's Start with Train Test Split")
    if tts:
        traintestsplit(data)
        st.success("Dataset Split successfully")

        st.write("## Scaling")

        scaling = st.radio("Want Scale or Normalise Your Data?", ["Yes", "No"], index=1)

        if "scaling" not in ss:
            ss.scaling = scaling

        ss.scaling = scaling

        if ss.scaling == "No":
            ss.X_train_scaled = ss.X_train
            ss.X_test_scaled = ss.X_test
            st.warning("Proceeding without Scaling")
        else:
            method = st.radio("Which method?", ['StandardScaler', 'MinMaxScaler', 'Normalizer'], index=0)

            if 'method' not in ss:
                ss.method = method

            ss.method = method

            scaler(ss.method)
            st.success("Data Scaled")

        st.write("Dimensionality Reduction using PCA")

        pca = st.radio("Want to reduce dimensionality of your Data?", ["Yes", "No"], index=1)

        if "pca" not in ss:
            ss.pca = pca

        ss.pca = pca

        if ss.pca == "No":
            st.warning("Proceeding without reducing Dimensions")
            ss.X_train_pca = ss.X_train_scaled
            ss.X_test_pca = ss.X_test_scaled
        else:
            n_comp = st.slider("How many components to choose?", 1, (len(data.columns) - 1))

            if n_comp not in ss:
                ss.n_comp = n_comp

            ss.n_comp = n_comp

            PriComAna(ss.n_comp)

        st.write(" ## PCA Data")
        st.write("### X_train")
        st.write(ss.X_train_pca)
        st.write("### y_train")
        st.write(ss.y_train)

# Main function to render the preprocess data page
def main():
    st.title("Preprocess Data")

    # Access the cleaned data from session state
    if 'cleaned_data' in ss:
        cleaned_data = ss.cleaned_data
        preprocess_data(cleaned_data)
    else:
        st.write("No cleaned data found. Please clean the data first.")

if __name__ == '__main__':
    main()
    if "X_train_pca" in ss:
        st.page_link(page="pages/3_📊_Classification.py", label="Proceed to Classify Data", icon="📊")