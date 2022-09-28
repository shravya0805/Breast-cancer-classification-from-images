import app1
import app2
import app3
import app4
import streamlit as st

PAGES = {
    "DataSet": app1,
    "Preprocessing": app2,
    "Classifier":app3,
    "Prediction":app4
}
st.sidebar.title('Navigation')
selection = st.sidebar.selectbox("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()
