Streamlit ML Application
========================

Overview
--------

This is a basic Streamlit application that provides an interactive interface for machine learning tasks. Users can upload their own datasets or choose a popular dataset from scikit-learn's database to perform data cleaning, preprocessing, and classification. The app also includes experimental features for generating data reports.

Features
--------

### Homepage

-   **CSV Upload**: Users can upload a CSV file to apply machine learning tasks on their own data.

### 1_🧹_Clean_Data

-   **Data Cleaning Options**:
    -   Remove rows with null values.
    -   Impute missing values using mean, median, or mode.
-   **Output**: The cleaned dataframe is passed on to the next step for further processing.

### 2_🔍_Preprocess_Data

-   **Train-Test Split**: Split the data into training and testing sets based on user-defined parameters.
-   **Scaling**:
    -   Option to scale the data using MinMaxScaler or StandardScaler.
-   **Dimensionality Reduction**:
    -   Option to apply Principal Component Analysis (PCA) to reduce the number of dimensions to a user-defined value.
-   **Output**: The processed dataframe is prepared for classification.

### 3_📊_Classification

-   **Model Selection**:
    -   Choose from a selection of classification models to apply to the dataset.
-   **Hyperparameter Tuning**:
    -   Option to perform hyperparameter tuning using grid search. *Note: This is a computationally intensive process and may take a significant amount of time.*

### 5_🗄️_sklearn_datasets

-   **Dataset Selection**: Users can select a popular dataset from the scikit-learn library if they do not have their own data to upload.

### 6_🧪_[Experimental]_Data_Report

-   **Data Report**: Generates a comprehensive report on the uploaded dataset using `ydata_profiling`, providing insights into the data structure, distribution, and potential issues.

Installation
------------

To run this Streamlit application locally, follow these steps:

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/madhavangn-work/Streamlit-ML.git
    cd Streamlit-ML
    ```
2.  **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the Streamlit application**:
    ```bash
    streamlit run 1_🏠_Home.py
    ```

Usage
-----

-   **Upload your dataset**: Navigate to the homepage and upload a CSV file.
-   **Clean the data**: Use the `1_🧹_Clean_Data` page to clean and preprocess your data.
-   **Preprocess the data**: Go to the `2_🔍_Preprocess_Data` page to split, scale, and reduce the dimensions of your data.
-   **Classification**: Choose a model from the `3_📊_Classification` page and run the classification task. Optionally, you can enable hyperparameter tuning for better results.
-   **Use a predefined dataset**: If you don't have your own data, head over to the `5_🗄️_sklearn_datasets` page to select a dataset from scikit-learn.
-   **Generate a data report**: Visit the `6_🧪_[Experimental]_Data_Report` page to generate a detailed report on your dataset.

Contributing
------------

Contributions are welcome! Please open an issue or submit a pull request for any enhancements, bug fixes, or new features.

License
-------

This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
---------------

-   [Streamlit](https://streamlit.io/) for providing an easy-to-use framework for building web apps.
-   [scikit-learn](https://scikit-learn.org/) for the machine learning models and datasets.
-   [ydata-profiling](https://github.com/ydataai/ydata-profiling) for the data report generation tool.
