import pandas as pd
from ydata_profiling import ProfileReport
import streamlit as st
import numpy as np


from streamlit import session_state as ss

st.set_page_config(layout="wide")

from streamlit_pandas_profiling import st_profile_report


if "X_train_pca" in ss:
    df = ss.X_train_pca
else:
    if "uploaded_data" in ss:
        df = ss.uploaded_data
    else:
        df = pd.DataFrame(np.random.rand(100, 5), columns=["a", "b", "c", "d", "e"])


pr = ProfileReport(df, title="Profiling Report")


st_profile_report(pr)