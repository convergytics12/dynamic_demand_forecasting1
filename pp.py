import streamlit as st
import pandas as pd

uploaded_files = st.file_uploader("Upload Files", accept_multiple_files=True)

filelist = []
dict={}
for i in uploaded_files:
    dict[i]=i.name
    
st.write(dict)
    

for file in uploaded_files:
    st.write(file)
    filelist.append(file.name)
    df1 = pd.read_excel(file)
    st.write(df1)
selected_file = st.selectbox("Select a file:", filelist)
st.write(selected_file)
df2 = pd.read_excel(selected_file)
st.write(df2)
    
        
        
        

