import streamlit as st
import pandas as pd

uploaded_files = st.file_uploader("Upload Files", accept_multiple_files=True)

filelist = []

for file in uploaded_files:
    filelist.append(file.name)
selected_file = st.selectbox("Select a file:", filelist)
df1 = pd.read_excel(str(selected_file))
st.write(df1)
    
        
        
        

