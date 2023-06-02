import streamlit as st
import pandas as pd

uploaded_files = st.file_uploader("Upload Files", accept_multiple_files=True)

filelist = []

for file in uploaded_files:
    filelist.append(file.name)
    df1 = pd.read_excel(file)
    st.write(df1)
selected_file = st.selectbox("Select a file:", filelist)
st.write(selected_file)
    
        
        
        

