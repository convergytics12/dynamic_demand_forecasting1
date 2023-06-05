import streamlit as st
import pandas as pd

uploaded_files = st.file_uploader("Upload Files", accept_multiple_files=True)

filelist = []

for file in uploaded_files:

    filelist.append(file.name)

selected_file = st.selectbox("Select a file:", filelist)


for i in uploaded_files:
    if(i.name==selected_file):
        df = pd.read_excel(i)
        st.dataframe(df)

        
        
        

