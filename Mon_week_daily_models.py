import os
import streamlit as st
import pandas as pd
import pandas                             as      pd
import numpy                              as      np
import matplotlib.pyplot                  as      plt
import seaborn                            as      sns
from   datetime                           import  datetime, timedelta
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.stattools            import  adfuller
from statsmodels.tsa.stattools            import  pacf
from statsmodels.tsa.stattools            import  acf
from statsmodels.graphics.tsaplots        import  plot_pacf
from statsmodels.graphics.tsaplots        import  plot_acf
from statsmodels.graphics.gofplots        import  qqplot
from statsmodels.tsa.seasonal             import  seasonal_decompose
from statsmodels.tsa.arima.model          import ARIMA
from statsmodels.tsa.statespace.sarimax   import  SARIMAX
import warnings
import time
import statsmodels.api as sm
import itertools
import plotly.graph_objects as go
st.set_option('deprecation.showPyplotGlobalUse', False)
np.seterr(divide='ignore', invalid='ignore')

def mape(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def adf_test(series):
    result = adfuller(series)
    print('ADF Statistics: {}'.format(result[0]))
    print('p-value: {}'.format(result[1]))
    if result[1] <=0.05:
        print("Strong evidence against the null hypothesis, reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis, time series has no unit root, indicating it is non-stationary")
        
        
rad = st.sidebar.radio('**Navigation**',['Select an option','Weekly','Monthly','Daily'])

if rad=='Select an option':
    pass

if rad=='Weekly':
    
    try:

        uploaded_files = st.file_uploader("Upload Files", accept_multiple_files=True)
        
        filelist = []
        
        for file in uploaded_files:
        
            filelist.append(file.name)
        
        selected_file = st.selectbox("Select a file:", filelist)
        
        
        for i in uploaded_files:
            if(i.name==selected_file):
                df1 = pd.read_excel(i)
                c_box=st.checkbox('Exog_Variable')
                if(c_box==True):
                    date=st.selectbox('**Select the date column**',['Select an option']+list(df1.columns))
                    target=st.selectbox('**Select the target column**',['Select an option']+list(df1.columns))   
                    exog=st.multiselect('**Select the exog column**',['Select an option']+list(df1.columns))
                    
                    fore_period=st.selectbox("Select the number of forecasting weeks:", [' ',1,2])
                    fore_period=int(fore_period)
                    
                    
                    exog_var=[]
        
                    for i in range(len(exog)):
                        exog_var.append(exog[i])
                        
                    file1 = st.file_uploader('Upload a file for Exog variables',key='f1')
    
                    if file1 is not None:
                        df_exog=pd.read_excel(file1,engine="openpyxl")
                
                    try:
                        df1.rename(columns={date:'dt',target:'Quantity'},inplace=True)
                        df1['dt']=pd.to_datetime(df1['dt'])
                        df1 = df1.set_index('dt')
                    except:
                        pass
                
                    fw = df1['dt'] + '-1'
                    dt = pd.to_datetime(fw, format='%Y-%W-%w', errors='coerce')
            
                    df = pd.DataFrame()
                    df['dt'] = dt
                    df['Quantity'] = df1['Quantity']
            
                    
                    df[exog_var]=df1[exog_var]
                    df = df.set_index('dt')
                    
                    p = range(0,5)
                    q = range(0,5)
                    result = adfuller(df['Quantity'])
                    if(result[1]<0.05):
                        d = range(0,1)
                    elif(result[1]>0.05):
                        d = range(1,2)
            
                    pdq = list(itertools.product(p, d, q))
            
                    a = [[(x[0],x[1],x[2],m)             for m in range(2,6,2)         if(m>x[0]  and m>x[2]) ]       for x in list(pdq)]
            
                    model_pdq = [item for sublist in a for item in sublist]
            
                    dfobj = pd.DataFrame(columns=['param', 'mape'])
                    dfobj1 = pd.DataFrame(columns=['param','seasonal', 'mape'])
                    dfobj2 = pd.DataFrame(columns=['sp','mape'])
                    dfobj3=pd.DataFrame(columns=['param','seasonal', 'mape'])
                    dfobj4=pd.DataFrame(columns=['param', 'mape'])
            
            
                    #ARIMA
            
            
                    for param in pdq:
                        try:
                            arima_mod = ARIMA(df['Quantity'],order=param)
                            best_model =arima_mod.fit()  
                            if(d==range(0,1)):
                                pred = best_model.fittedvalues
                                res=mape(df['Quantity'],pred)
                            elif(d==range(1,2)):
                                pred = best_model.fittedvalues
                                #pred1 = pred.shift(periods=-1)
                                res=mape(df['Quantity'][1:],pred[1:])
                            #dfobj=dfobj.append({'param':param,'mape':res},ignore_index=True)
                            dfobj=pd.concat([dfobj,pd.DataFrame({'param':[param],'mape':[res]})]).reset_index(drop=True)
                        except:
                            continue
                    #SARIMA
            
                    for param in pdq:
                        for param_seasonal in model_pdq:
                            try:
                                mod = sm.tsa.statespace.SARIMAX(df['Quantity'],
                                                                    order=param,
                                                                    seasonal_order=param_seasonal,
                                                                    enforce_stationarity=False,
                                                                    enforce_invertibility=False)
                                best_model =mod.fit()   
                               
                                pred = best_model.fittedvalues
                                #pred1 = pred.shift(periods=-1)
                                res=mape(df['Quantity'][1:],pred[1:])
                            #dfobj1 = dfobj1.append({'param':param,'seasonal':param_seasonal ,'mape': res}, ignore_index=True)
                                dfobj1=pd.concat([dfobj1,pd.DataFrame({'param':[param],'seasonal':[param_seasonal] ,'mape': [res]})]).reset_index(drop=True)
                            except:
                                continue
                    #TES
            
                    for sp in range(2,11,2):
                        tes = ExponentialSmoothing(df['Quantity'],trend='additive',seasonal='additive',seasonal_periods=sp,initialization_method='estimated')
                        tes_model = tes.fit()
                        pred = tes_model.fittedvalues
                        res=mape(df['Quantity'],pred)
                        #dfobj2 = dfobj2.append({'sp':sp,'mape': res}, ignore_index=True)
                        dfobj2=pd.concat([dfobj2,pd.DataFrame({'sp':[sp],'mape':[res]})]).reset_index(drop=True)
                    
                    #SARIMAX
                    
                    for param in pdq:
                        for param_seasonal in model_pdq:
                            try:
                                mod = sm.tsa.statespace.SARIMAX(df['Quantity'],
                                                                    order=param,
                                                                    seasonal_order=param_seasonal,
                                                                    exog=df[exog_var],
                                                                    enforce_stationarity=False,
                                                                    enforce_invertibility=False)
                                best_model =mod.fit()   
                                
                                pred = best_model.fittedvalues
                                #pred1 = pred.shift(periods=-1)
                                res=mape(df['Quantity'][1:],pred[1:])
                                #dfobj1 = dfobj1.append({'param':param,'seasonal':param_seasonal ,'mape': res}, ignore_index=True)
                                dfobj3=pd.concat([dfobj3,pd.DataFrame({'param':[param],'seasonal':[param_seasonal] ,'mape': [res]})]).reset_index(drop=True)
                            except:
                                continue
                    
                    #ARIMAX
      
            
                    for param in pdq:
                       try:
                           arima_mod = ARIMA(df['Quantity'],order=param,exog=df[exog_var])
                           best_model =arima_mod.fit()  
                           if(d==range(0,1)):
                               pred = best_model.fittedvalues
                               res=mape(df['Quantity'],pred)
                           elif(d==range(1,2)):
                               pred = best_model.fittedvalues
                               #pred1 = pred.shift(periods=-1)
                               res=mape(df['Quantity'][1:],pred[1:])
                           #dfobj=dfobj.append({'param':param,'mape':res},ignore_index=True)
                           dfobj4=pd.concat([dfobj4,pd.DataFrame({'param':[param],'mape':[res]})]).reset_index(drop=True)
                       except:
                           continue
            
            
                    arima_results=dfobj.sort_values(by=['mape'])[:3].reset_index(drop=True)
                    arima_mape=arima_results[:1]['mape'].to_list()[0]
                    arima_order=arima_results[:1]['param'].to_list()[0]
                    
                    
                    sarima_results=dfobj1.sort_values(by=['mape'])[:3].reset_index(drop=True)
                    sarima_mape=sarima_results[:1]['mape'].to_list()[0]
                    sarima_order=sarima_results[:1]['param'].to_list()[0]
                    sarima_seasonal_order=sarima_results[:1]['seasonal'].to_list()[0]
                    
                    tes_results=dfobj2.sort_values(by=['mape'])[:3].reset_index(drop=True)
                    tes_mape=tes_results[:1]['mape'].to_list()[0]
                    tes_sp=tes_results[:1]['sp'].to_list()[0]
                
                    sarimax_results=dfobj3.sort_values(by=['mape'])[:3].reset_index(drop=True)
                    sarimax_mape=sarimax_results[:1]['mape'].to_list()[0]
                    sarimax_order=sarimax_results[:1]['param'].to_list()[0]
                    sarimax_seasonal_order=sarimax_results[:1]['seasonal'].to_list()[0]
                    
                    arimax_results=dfobj4.sort_values(by=['mape'])[:3].reset_index(drop=True)
                    arimax_mape=arimax_results[:1]['mape'].to_list()[0]
                    arimax_order=arimax_results[:1]['param'].to_list()[0]
                 
                    variables = {'x': arima_mape, 'y': sarima_mape, 'z': tes_mape,'a':sarimax_mape, 'u':arimax_mape}
                    largest_variable = min(variables, key=variables.get)
                    
                    
            
            
                    if(largest_variable=='x'):
                        arima = ARIMA(df['Quantity'],order=arima_order)
                        best_model=arima.fit()
                        st.write('Arima_MAPE: ',round(arima_mape,2))
                        if(d==range(0,1)):
                           pred = best_model.fittedvalues
                           res=mape(df['Quantity'],pred)
                           
            
                           dfa=df.reset_index()
                        #dfa.rename(columns={'dt':'Date','Quantity':'Actual Production'},inplace=True)
            
                           dfp=pd.DataFrame(pred)
                           dfp=dfp.reset_index()
                           dfp.rename(columns={0:'Quantity'},inplace=True)
            
                           fore=best_model.predict(start=(pd.Series(df.index[-1])[0])+ pd.DateOffset(weeks=1),end=(pd.Series(df.index[-1])[0])+ pd.DateOffset(weeks=fore_period))
                           fore_df=pd.DataFrame(data=fore.reset_index())
                           fore_df=fore_df.rename(columns={'index':'dt','predicted_mean':'Quantity'})
                           a=pd.DataFrame(pred.reset_index().iloc[-1]).T.rename(columns={0:'Quantity'}).reset_index(drop=True)
                           dff=pd.concat([a,fore_df],ignore_index=True)
                           dff['dt']=pd.to_datetime(dff['dt'])
                        #dff=dff.set_index(dff['Date'])
                        #dff=dff.drop(['Date'],axis=1)
            
                           fig = go.Figure()
                           fig.add_trace(go.Scatter(x=dfa['dt'], y=dfa['Quantity'], name='Actual Quantity', line=dict(color='blue')))
                           fig.add_trace(go.Scatter(x=dfp['dt'], y=dfp['Quantity'], name='Prediction', line=dict(color='orange')))
                           fig.add_trace(go.Scatter(x=dff['dt'], y=dff['Quantity'], name='Forecast', line=dict(color='green')))
                           st.plotly_chart(fig)
                           
                        elif(d==range(1,2)):
                           pred = best_model.fittedvalues
                           #pred = pred.shift(periods=-1)
                           res=mape(df['Quantity'][1:],pred[1:])
                           
            
                           dfa=df.reset_index()
                        #dfa.rename(columns={'dt':'Date','Quantity':'Actual Production'},inplace=True)
            
                           dfp=pd.DataFrame(pred)
                           dfp=dfp.reset_index()
                           dfp.rename(columns={0:'Quantity'},inplace=True)
            
                           fore=best_model.predict(start=(pd.Series(df.index[-1])[0])+ pd.DateOffset(weeks=1),end=(pd.Series(df.index[-1])[0])+ pd.DateOffset(weeks=fore_period))
                           fore_df=pd.DataFrame(data=fore.reset_index())
                           fore_df=fore_df.rename(columns={'index':'dt','predicted_mean':'Quantity'})
                           a=pd.DataFrame(pred.reset_index().iloc[-1]).T.rename(columns={0:'Quantity'}).reset_index(drop=True)
                           dff=pd.concat([a,fore_df],ignore_index=True)
                           dff['dt']=pd.to_datetime(dff['dt'])
                        #dff=dff.set_index(dff['Date'])
                        #dff=dff.drop(['Date'],axis=1)
            
                           fig = go.Figure()
                           fig.add_trace(go.Scatter(x=dfa['dt'][1:], y=dfa['Quantity'][1:], name='Actual Quantity', line=dict(color='blue')))
                           fig.add_trace(go.Scatter(x=dfp['dt'][1:], y=dfp['Quantity'][1:], name='Prediction', line=dict(color='orange')))
                           fig.add_trace(go.Scatter(x=dff['dt'], y=dff['Quantity'], name='Forecast', line=dict(color='green')))
                           st.plotly_chart(fig) 
            
                    if(largest_variable=='y'):
                        mod = sm.tsa.statespace.SARIMAX(df['Quantity'],order=sarima_order,seasonal_order=sarima_seasonal_order,
                                                        enforce_stationarity=False,enforce_invertibility=False)
                        best_model =mod.fit()
                        st.write('Sarima_MAPE: ',round(sarima_mape,2))
            
                        
                        pred = best_model.fittedvalues
                        #pred = pred.shift(periods=-1)
                        res=mape(df['Quantity'][1:],pred[1:])
                       
    
                        dfa=df.reset_index()
                    #dfa.rename(columns={'dt':'Date','Quantity':'Actual Production'},inplace=True)
    
                        dfp=pd.DataFrame(pred)
                        dfp=dfp.reset_index()
                        dfp.rename(columns={0:'Quantity'},inplace=True)
    
                        fore=best_model.predict(start=(pd.Series(df.index[-1])[0])+ pd.DateOffset(weeks=1),end=(pd.Series(df.index[-1])[0])+ pd.DateOffset(weeks=fore_period))
                        fore_df=pd.DataFrame(data=fore.reset_index())
                        fore_df=fore_df.rename(columns={'index':'dt','predicted_mean':'Quantity'})
                        a=pd.DataFrame(pred.reset_index().iloc[-1]).T.rename(columns={0:'Quantity'}).reset_index(drop=True)
                        dff=pd.concat([a,fore_df],ignore_index=True)
                        dff['dt']=pd.to_datetime(dff['dt'])
                    #dff=dff.set_index(dff['Date'])
                    #dff=dff.drop(['Date'],axis=1)
    
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=dfa['dt'][1:], y=dfa['Quantity'][1:], name='Actual Quantity', line=dict(color='blue')))
                        fig.add_trace(go.Scatter(x=dfp['dt'][1:], y=dfp['Quantity'][1:], name='Prediction', line=dict(color='orange')))
                        fig.add_trace(go.Scatter(x=dff['dt'], y=dff['Quantity'], name='Forecast', line=dict(color='green')))
                        st.plotly_chart(fig) 
            
            
                    if(largest_variable=='z'):
                        tes = ExponentialSmoothing(df['Quantity'],trend='additive',seasonal='additive',seasonal_periods=tes_sp,
                                              initialization_method='estimated')
                        best_model =tes.fit()  
                        st.write('TES_MAPE: ',round(tes_mape,2))
                        pred = best_model.fittedvalues
                        #pred = pred.shift(periods=-1)
            
                        dfa=df.reset_index()
                        dfa.rename(columns={'dt':'Date','Quantity':'Actual Production'},inplace=True)
            
                        dfp=pd.DataFrame(pred)
                        dfp=dfp.reset_index()
                        dfp.rename(columns={'dt':'Date',0:'Predicted Production'},inplace=True)
            
                        fore=best_model.predict(start=(pd.Series(df.index[-1])[0])+ pd.DateOffset(weeks=1),end=(pd.Series(df.index[-1])[0])+ pd.DateOffset(weeks=fore_period))
                        fore_df=pd.DataFrame(data=fore.reset_index())
                        fore_df=fore_df.rename(columns={'index':'Date',0:'Forecasted Production'})
                        a=pd.DataFrame(pred.reset_index().iloc[-1]).T.rename(columns={'dt':'Date',0:'Forecasted Production'}).reset_index(drop=True)
                        dff=pd.concat([a,fore_df],ignore_index=True)
                        dff['Date']=pd.to_datetime(dff['Date'])
                        #dff=dff.set_index(dff['Date'])
                        #dff=dff.drop(['Date'],axis=1)
            
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=dfa['Date'], y=dfa['Actual Production'], name='Actual Production', line=dict(color='blue')))
                        fig.add_trace(go.Scatter(x=dfp['Date'], y=dfp['Predicted Production'], name='Predicted Production', line=dict(color='orange')))
                        fig.add_trace(go.Scatter(x=dff['Date'], y=dff['Forecasted Production'], name='Forecasted Production', line=dict(color='green')))
                        st.plotly_chart(fig)
                        
                    
                    if(largest_variable=='a'):
                       mod = sm.tsa.statespace.SARIMAX(df['Quantity'],order=sarimax_order,seasonal_order=sarimax_seasonal_order,exog=df[exog_var],
                                                       enforce_stationarity=False,enforce_invertibility=False)
                       best_model =mod.fit()
                       st.write('Sarimax_MAPE: ',round(sarimax_mape,2))
           
                       
                       pred = best_model.fittedvalues
                       #pred = pred.shift(periods=-1)
                       res=mape(df['Quantity'][1:],pred[1:])
                      
    
                       dfa=df.reset_index()
                         #dfa.rename(columns={'dt':'Date','Quantity':'Actual Production'},inplace=True)
    
                       dfp=pd.DataFrame(pred)
                       dfp=dfp.reset_index()
                       dfp.rename(columns={0:'Quantity'},inplace=True)
    
                       fore=best_model.predict(start=(pd.Series(df.index[-1])[0])+ pd.DateOffset(weeks=1),end=(pd.Series(df.index[-1])[0])+ pd.DateOffset(weeks=fore_period),exog=df_exog[exog_var])
                       fore_df=pd.DataFrame(data=fore.reset_index())
                       fore_df=fore_df.rename(columns={'index':'dt','predicted_mean':'Quantity'})
                       a=pd.DataFrame(pred.reset_index().iloc[-1]).T.rename(columns={0:'Quantity'}).reset_index(drop=True)
                       dff=pd.concat([a,fore_df],ignore_index=True)
                       dff['dt']=pd.to_datetime(dff['dt'])
              
                       fig = go.Figure()
                       fig.add_trace(go.Scatter(x=dfa['dt'][1:], y=dfa['Quantity'][1:], name='Actual Quantity', line=dict(color='blue')))
                       fig.add_trace(go.Scatter(x=dfp['dt'][1:], y=dfp['Quantity'][1:], name='Prediction', line=dict(color='orange')))
                       fig.add_trace(go.Scatter(x=dff['dt'], y=dff['Quantity'], name='Forecast', line=dict(color='green')))
                       st.plotly_chart(fig)
                       
                    
                    if(largest_variable=='u'):
                        
                        
                        arimax = sm.tsa.statespace.SARIMAX(df['Quantity'],order=arimax_order,seasonal_order=(0,0,0,0),exog = df[exog_var])
                        best_model=arimax.fit()
                        st.write('Arimax_MAPE: ',round(arimax_mape,2))
            
                        pred = best_model.fittedvalues
                        #pred = pred.shift(periods=-1)
                        res=mape(df['Quantity'][1:],pred[1:])
                       
    
                        dfa=df.reset_index()
                    #dfa.rename(columns={'dt':'Date','Quantity':'Actual Production'},inplace=True)
    
                        dfp=pd.DataFrame(pred)
                        dfp=dfp.reset_index()
                        dfp.rename(columns={0:'Quantity'},inplace=True)
    
                        fore=best_model.predict(start=(pd.Series(df.index[-1])[0])+ pd.DateOffset(weeks=1),end=(pd.Series(df.index[-1])[0])+ pd.DateOffset(weeks=fore_period),exog = df_exog[exog_var])
                        fore_df=pd.DataFrame(data=fore.reset_index())
                        fore_df=fore_df.rename(columns={'index':'dt','predicted_mean':'Quantity'})
                        a=pd.DataFrame(pred.reset_index().iloc[-1]).T.rename(columns={0:'Quantity'}).reset_index(drop=True)
                        dff=pd.concat([a,fore_df],ignore_index=True)
                        dff['dt']=pd.to_datetime(dff['dt'])
                    #dff=dff.set_index(dff['Date'])
                    #dff=dff.drop(['Date'],axis=1)
    
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=dfa['dt'][1:], y=dfa['Quantity'][1:], name='Actual Quantity', line=dict(color='blue')))
                        fig.add_trace(go.Scatter(x=dfp['dt'][1:], y=dfp['Quantity'][1:], name='Prediction', line=dict(color='orange')))
                        fig.add_trace(go.Scatter(x=dff['dt'], y=dff['Quantity'], name='Forecast', line=dict(color='green')))
                        st.plotly_chart(fig)
                        
                    arima_results['Model']='ARIMA'
                    sarima_results['Model']='SARIMA'
                    tes_results['Model']='TES'
                    sarimax_results['Model']='SARIMAX'
                    arimax_results['Model']='ARIMAX'
                    
                    ff=pd.concat([arima_results,sarima_results])
                    ff=pd.concat([ff,tes_results])
                    ff=pd.concat([ff,sarimax_results])
                    ff=pd.concat([ff,arimax_results])
                    
                    g = ff.sort_values(by='mape').reset_index(drop=True)[:3]
                    st.subheader('Top Models')
                    st.dataframe(g)
                    
                    st.subheader('SARIMAX Results:')
                    st.table(sarimax_results)
                    
                    writer = pd.ExcelWriter('data.xlsx', engine='openpyxl')
                    fore_df.to_excel(writer, sheet_name='Sheet1', index=False)
                    writer.close()
                    with open('data.xlsx', 'rb') as f:
                        excel_data = f.read()
                        st.download_button(label='Click here to download', data=excel_data, file_name='data.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
                        st.stop()
                    
                else:
                    date=st.selectbox('**Select the date column**',['Select an option']+list(df1.columns))
                    target=st.selectbox('**Select the target column**',['Select an option']+list(df1.columns))
                    fore_period=st.selectbox("Select the number of forecasting weeks:", ['',1,2])
                    fore_period=int(fore_period)
                    
                    try:
                        df1.rename(columns={date:'dt',target:'Quantity'},inplace=True)
                        df1['dt']=pd.to_datetime(df1['dt'])
                        df1 = df1.set_index('dt')
                    except:
                        pass
                
                    fw = df1['dt'] + '-1'
                    dt = pd.to_datetime(fw, format='%Y-%W-%w', errors='coerce')
            
                    df = pd.DataFrame()
                    df['dt'] = dt
                    df['Quantity'] = df1['Quantity']
                  
                    df = df.set_index('dt')
                    
               
                    p = range(0,5)
                    q = range(0,5)
                    result = adfuller(df['Quantity'])
                    if(result[1]<0.05):
                        d = range(0,1)
                    elif(result[1]>0.05):
                        d = range(1,2)
            
                    pdq = list(itertools.product(p, d, q))
            
                    a = [[(x[0],x[1],x[2],m)             for m in range(2,6,2)         if(m>x[0]  and m>x[2]) ]       for x in list(pdq)]
            
                    model_pdq = [item for sublist in a for item in sublist]
            
                    dfobj = pd.DataFrame(columns=['param', 'mape'])
                    dfobj1 = pd.DataFrame(columns=['param','seasonal', 'mape'])
                    dfobj2 = pd.DataFrame(columns=['sp','mape'])
            
            
                    #ARIMA
            
            
                    for param in pdq:
                        try:
                            arima_mod = ARIMA(df['Quantity'],order=param)
                            best_model =arima_mod.fit()  
                            if(d==range(0,1)):
                                pred = best_model.fittedvalues
                                res=mape(df['Quantity'],pred)
                            elif(d==range(1,2)):
                                pred = best_model.fittedvalues
                                #pred1 = pred.shift(periods=-1)
                                res=mape(df['Quantity'][1:],pred[1:])
                            #dfobj=dfobj.append({'param':param,'mape':res},ignore_index=True)
                            dfobj=pd.concat([dfobj,pd.DataFrame({'param':[param],'mape':[res]})]).reset_index(drop=True)
                        except:
                            continue
                    #SARIMA
            
                    for param in pdq:
                        for param_seasonal in model_pdq:
                            try:
                                mod = sm.tsa.statespace.SARIMAX(df['Quantity'],
                                                                    order=param,
                                                                    seasonal_order=param_seasonal,
                                                                    enforce_stationarity=False,
                                                                    enforce_invertibility=False)
                                best_model =mod.fit()   
                               
                                pred = best_model.fittedvalues
                                #pred1 = pred.shift(periods=-1)
                                res=mape(df['Quantity'][1:],pred[1:])
                            #dfobj1 = dfobj1.append({'param':param,'seasonal':param_seasonal ,'mape': res}, ignore_index=True)
                                dfobj1=pd.concat([dfobj1,pd.DataFrame({'param':[param],'seasonal':[param_seasonal] ,'mape': [res]})]).reset_index(drop=True)
                            except:
                                continue
                    #TES
            
                    for sp in range(2,11,2):
                        tes = ExponentialSmoothing(df['Quantity'],trend='additive',seasonal='additive',seasonal_periods=sp,initialization_method='estimated')
                        tes_model = tes.fit()
                        pred = tes_model.fittedvalues
                        res=mape(df['Quantity'],pred)
                        #dfobj2 = dfobj2.append({'sp':sp,'mape': res}, ignore_index=True)
                        dfobj2=pd.concat([dfobj2,pd.DataFrame({'sp':[sp],'mape':[res]})]).reset_index(drop=True)
            
            
            
                    arima_mape = dfobj.sort_values(by=['mape'])[:1]['mape'].to_list()[0]
                    arima_order = dfobj.sort_values(by=['mape'])[:1]['param'].to_list()[0]
            
                    sarima_mape = dfobj1.sort_values(by=['mape'])[:1]['mape'].to_list()[0]
                    sarima_order = dfobj1.sort_values(by=['mape'])[:1]['param'].to_list()[0]
                    sarima_seasonal_order = dfobj1.sort_values(by=['mape'])[:1]['seasonal'].to_list()[0]
            
                    tes_mape = dfobj2.sort_values(by=['mape'])[:1]['mape'].to_list()[0]
                    tes_sp = dfobj2.sort_values(by=['mape'])[:1]['sp'].to_list()[0]
            
                    variables = {'x': arima_mape, 'y': sarima_mape, 'z': tes_mape}
                    largest_variable = min(variables, key=variables.get)
                    
                    
            
            
                    if(largest_variable=='x'):
                        arima = ARIMA(df['Quantity'],order=arima_order)
                        best_model=arima.fit()
                        st.write('Arima_MAPE: ',round(arima_mape,2))
                        if(d==range(0,1)):
                           pred = best_model.fittedvalues
                           res=mape(df['Quantity'],pred)
                           
            
                           dfa=df.reset_index()
                        #dfa.rename(columns={'dt':'Date','Quantity':'Actual Production'},inplace=True)
            
                           dfp=pd.DataFrame(pred)
                           dfp=dfp.reset_index()
                           dfp.rename(columns={0:'Quantity'},inplace=True)
            
                           fore=best_model.predict(start=(pd.Series(df.index[-1])[0])+ pd.DateOffset(weeks=1),end=(pd.Series(df.index[-1])[0])+ pd.DateOffset(weeks=fore_period))
                           fore_df=pd.DataFrame(data=fore.reset_index())
                           fore_df=fore_df.rename(columns={'index':'dt','predicted_mean':'Quantity'})
                           a=pd.DataFrame(pred.reset_index().iloc[-1]).T.rename(columns={0:'Quantity'}).reset_index(drop=True)
                           dff=pd.concat([a,fore_df],ignore_index=True)
                           dff['dt']=pd.to_datetime(dff['dt'])
                        #dff=dff.set_index(dff['Date'])
                        #dff=dff.drop(['Date'],axis=1)
            
                           fig = go.Figure()
                           fig.add_trace(go.Scatter(x=dfa['dt'], y=dfa['Quantity'], name='Actual Quantity', line=dict(color='blue')))
                           fig.add_trace(go.Scatter(x=dfp['dt'], y=dfp['Quantity'], name='Prediction', line=dict(color='orange')))
                           fig.add_trace(go.Scatter(x=dff['dt'], y=dff['Quantity'], name='Forecast', line=dict(color='green')))
                           st.plotly_chart(fig)
                           
                        elif(d==range(1,2)):
                           pred = best_model.fittedvalues
                           #pred = pred.shift(periods=-1)
                           res=mape(df['Quantity'][1:],pred[1:])
                           
            
                           dfa=df.reset_index()
                        #dfa.rename(columns={'dt':'Date','Quantity':'Actual Production'},inplace=True)
            
                           dfp=pd.DataFrame(pred)
                           dfp=dfp.reset_index()
                           dfp.rename(columns={0:'Quantity'},inplace=True)
            
                           fore=best_model.predict(start=(pd.Series(df.index[-1])[0])+ pd.DateOffset(weeks=1),end=(pd.Series(df.index[-1])[0])+ pd.DateOffset(weeks=fore_period))
                           fore_df=pd.DataFrame(data=fore.reset_index())
                           fore_df=fore_df.rename(columns={'index':'dt','predicted_mean':'Quantity'})
                           a=pd.DataFrame(pred.reset_index().iloc[-1]).T.rename(columns={0:'Quantity'}).reset_index(drop=True)
                           dff=pd.concat([a,fore_df],ignore_index=True)
                           dff['dt']=pd.to_datetime(dff['dt'])
                        #dff=dff.set_index(dff['Date'])
                        #dff=dff.drop(['Date'],axis=1)
            
                           fig = go.Figure()
                           fig.add_trace(go.Scatter(x=dfa['dt'][1:], y=dfa['Quantity'][1:], name='Actual Quantity', line=dict(color='blue')))
                           fig.add_trace(go.Scatter(x=dfp['dt'][1:], y=dfp['Quantity'][1:], name='Prediction', line=dict(color='orange')))
                           fig.add_trace(go.Scatter(x=dff['dt'], y=dff['Quantity'], name='Forecast', line=dict(color='green')))
                           st.plotly_chart(fig) 
            
                    if(largest_variable=='y'):
                        mod = sm.tsa.statespace.SARIMAX(df['Quantity'],order=sarima_order,seasonal_order=sarima_seasonal_order,
                                                        enforce_stationarity=False,enforce_invertibility=False)
                        best_model =mod.fit()
                        st.write('Sarima_MAPE: ',round(sarima_mape,2))
            
                        
                        pred = best_model.fittedvalues
                        #pred = pred.shift(periods=-1)
                        res=mape(df['Quantity'][1:],pred[1:])
                       
        
                        dfa=df.reset_index()
                    #dfa.rename(columns={'dt':'Date','Quantity':'Actual Production'},inplace=True)
        
                        dfp=pd.DataFrame(pred)
                        dfp=dfp.reset_index()
                        dfp.rename(columns={0:'Quantity'},inplace=True)
        
                        fore=best_model.predict(start=(pd.Series(df.index[-1])[0])+ pd.DateOffset(weeks=1),end=(pd.Series(df.index[-1])[0])+ pd.DateOffset(weeks=fore_period))
                        fore_df=pd.DataFrame(data=fore.reset_index())
                        fore_df=fore_df.rename(columns={'index':'dt','predicted_mean':'Quantity'})
                        a=pd.DataFrame(pred.reset_index().iloc[-1]).T.rename(columns={0:'Quantity'}).reset_index(drop=True)
                        dff=pd.concat([a,fore_df],ignore_index=True)
                        dff['dt']=pd.to_datetime(dff['dt'])
                    #dff=dff.set_index(dff['Date'])
                    #dff=dff.drop(['Date'],axis=1)
        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=dfa['dt'][1:], y=dfa['Quantity'][1:], name='Actual Quantity', line=dict(color='blue')))
                        fig.add_trace(go.Scatter(x=dfp['dt'][1:], y=dfp['Quantity'][1:], name='Prediction', line=dict(color='orange')))
                        fig.add_trace(go.Scatter(x=dff['dt'], y=dff['Quantity'], name='Forecast', line=dict(color='green')))
                        st.plotly_chart(fig) 
            
            
                    if(largest_variable=='z'):
                        tes = ExponentialSmoothing(df['Quantity'],trend='additive',seasonal='additive',seasonal_periods=tes_sp,
                                              initialization_method='estimated')
                        best_model =tes.fit()  
                        st.write('TES_MAPE: ',round(tes_mape,2))
                        pred = best_model.fittedvalues
                        #pred = pred.shift(periods=-1)
            
                        dfa=df.reset_index()
                        dfa.rename(columns={'dt':'Date','Quantity':'Actual Production'},inplace=True)
            
                        dfp=pd.DataFrame(pred)
                        dfp=dfp.reset_index()
                        dfp.rename(columns={'dt':'Date',0:'Predicted Production'},inplace=True)
            
                        fore=best_model.predict(start=(pd.Series(df.index[-1])[0])+ pd.DateOffset(weeks=1),end=(pd.Series(df.index[-1])[0])+ pd.DateOffset(weeks=fore_period))
                        fore_df=pd.DataFrame(data=fore.reset_index())
                        fore_df=fore_df.rename(columns={'index':'Date',0:'Forecasted Production'})
                        a=pd.DataFrame(pred.reset_index().iloc[-1]).T.rename(columns={'dt':'Date',0:'Forecasted Production'}).reset_index(drop=True)
                        dff=pd.concat([a,fore_df],ignore_index=True)
                        dff['Date']=pd.to_datetime(dff['Date'])
                        #dff=dff.set_index(dff['Date'])
                        #dff=dff.drop(['Date'],axis=1)
            
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=dfa['Date'], y=dfa['Actual Production'], name='Actual Production', line=dict(color='blue')))
                        fig.add_trace(go.Scatter(x=dfp['Date'], y=dfp['Predicted Production'], name='Predicted Production', line=dict(color='orange')))
                        fig.add_trace(go.Scatter(x=dff['Date'], y=dff['Forecasted Production'], name='Forecasted Production', line=dict(color='green')))
                        st.plotly_chart(fig)
                        
                    dfobj_t=dfobj.copy()
                    dfobj_t['Model']='ARIMA'
                    dfobj1_t=dfobj1.copy()
                    dfobj1_t['Model']='SARIMA'
                    dfobj2_t=dfobj2.copy()
                    dfobj2_t['Model']='TES'
                    
                    ff=pd.concat([dfobj_t,dfobj1_t])
                    ff=pd.concat([ff,dfobj2_t])
                    g = ff.sort_values(by='mape').reset_index(drop=True)[:3]
                    st.subheader('Top Models')
                    st.table(g)
                    
                    writer = pd.ExcelWriter('data.xlsx', engine='openpyxl')
                    fore_df.to_excel(writer, sheet_name='Sheet1', index=False)
                    writer.close()
                    with open('data.xlsx', 'rb') as f:
                        excel_data = f.read()
                        st.download_button(label='Click here to download', data=excel_data, file_name='data.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
                        st.stop()
    except:
        st.write(" Select the above options to continue")
if rad=='Monthly':
    
    try:

        uploaded_files = st.file_uploader("Upload Files", accept_multiple_files=True)
        
        filelist = []
        
        for file in uploaded_files:
        
            filelist.append(file.name)
        
        selected_file = st.selectbox("Select a file:", filelist)
        
        
        for i in uploaded_files:
            if(i.name==selected_file):
                df = pd.read_excel(i)
                c_box_m=st.checkbox('Exog_Variable')
                
                
                
                if(c_box_m==True):
                    
                    date=st.selectbox('**Select the date column**',['Select an option']+list(df.columns))
                    target=st.selectbox('**Select the target column**',['Select an option']+list(df.columns))
                    exog=st.multiselect('**Select the exog column**',['Select an option']+list(df.columns))
                    
                    fore_period=st.selectbox("Select the number of forecasting months:", ['',3,6])
                    fore_period=int(fore_period)
        
                    exog_var=[]
        
                    for i in range(len(exog)):
                        exog_var.append(exog[i])
                        
                    file1 = st.file_uploader('Upload a file for Exog variables',key='f1')
    
                    if file1 is not None:
                        df_exog=pd.read_excel(file1,engine="openpyxl")
                
                    try:
                        df.rename(columns={date:'dt',target:'Quantity'},inplace=True)
                        df['dt']=pd.to_datetime(df['dt'])
                        df = df.set_index('dt')
                    except:
                        pass
                   # df = pd.DataFrame()
                    #df['dt'] = dt
                    #df['Quantity'] = df1['Quantity']
            
                 
            
                    p = range(1,5)
                    q = range(1,5)
                    result = adfuller(df['Quantity'])
                    if(result[1]<0.05):
                        d = range(0,1)
                    elif(result[1]>0.05):
                        d = range(1,2)
            
                    pdq = list(itertools.product(p, d, q))
                    
                    z = (12,1)
            
                    a = [[(x[0],x[1],x[2],m)             for m in z         if(m>x[0]  and m>x[2]) ]       for x in list(pdq)]
            
                    model_pdq = [item for sublist in a for item in sublist]
            
                    dfobj = pd.DataFrame(columns=['param', 'mape'])
                    dfobj1 = pd.DataFrame(columns=['param','seasonal', 'mape'])
                    dfobj2 = pd.DataFrame(columns=['sp','mape'])
                    dfobj3=pd.DataFrame(columns=['param','seasonal', 'mape'])
                    dfobj4=pd.DataFrame(columns=['param', 'mape'])
            
            
                    #ARIMA
            
                    for param in pdq:
                       try:
                           arima_mod = ARIMA(df['Quantity'],order=param)
                           best_model =arima_mod.fit()  
                           if(d==range(0,1)):
                               pred = best_model.fittedvalues
                               res=mape(df['Quantity'],pred)
                           elif(d==range(1,2)):
                               pred = best_model.fittedvalues
                               #pred1 = pred.shift(periods=-1)
                               res=mape(df['Quantity'][1:],pred[1:])
                           #dfobj=dfobj.append({'param':param,'mape':res},ignore_index=True)
                           dfobj=pd.concat([dfobj,pd.DataFrame({'param':[param],'mape':[res]})]).reset_index(drop=True)
                       except:
                           continue
                    #SARIMA
            
                    for param in pdq:
                        for param_seasonal in model_pdq:
                            try:
                                mod = sm.tsa.statespace.SARIMAX(df['Quantity'],
                                                                    order=param,
                                                                    seasonal_order=param_seasonal,
                                                                    enforce_stationarity=False,
                                                                    enforce_invertibility=False)
                                best_model =mod.fit()   
                                
                                pred = best_model.fittedvalues
                                #pred1 = pred.shift(periods=-1)
                                res=mape(df['Quantity'][1:],pred[1:])
                                #dfobj1 = dfobj1.append({'param':param,'seasonal':param_seasonal ,'mape': res}, ignore_index=True)
                                dfobj1=pd.concat([dfobj1,pd.DataFrame({'param':[param],'seasonal':[param_seasonal] ,'mape': [res]})]).reset_index(drop=True)
                            except:
                                continue
                            
                            
            
                    #TES
            
                    for sp in range(2,11,2):
                        tes = ExponentialSmoothing(df['Quantity'],trend='additive',seasonal='additive',seasonal_periods=sp,initialization_method='estimated')
                        tes_model = tes.fit()
                        pred = tes_model.fittedvalues
                        res=mape(df['Quantity'],pred)
                        #dfobj2 = dfobj2.append({'sp':sp,'mape': res}, ignore_index=True)
                        dfobj2=pd.concat([dfobj2,pd.DataFrame({'sp':[sp],'mape':[res]})]).reset_index(drop=True)
                        
                        
                    #SARIMAX
                    
                    for param in pdq:
                        for param_seasonal in model_pdq:
                            try:
                                mod = sm.tsa.statespace.SARIMAX(df['Quantity'],
                                                                    order=param,
                                                                    seasonal_order=param_seasonal,
                                                                    exog=df[exog_var],
                                                                    enforce_stationarity=False,
                                                                    enforce_invertibility=False)
                                best_model =mod.fit()   
                                
                                pred = best_model.fittedvalues
                                #pred1 = pred.shift(periods=-1)
                                res=mape(df['Quantity'][1:],pred[1:])
                                #dfobj1 = dfobj1.append({'param':param,'seasonal':param_seasonal ,'mape': res}, ignore_index=True)
                                dfobj3=pd.concat([dfobj3,pd.DataFrame({'param':[param],'seasonal':[param_seasonal] ,'mape': [res]})]).reset_index(drop=True)
                            except:
                                continue
                    #ARIMAX
      
            
                    for param in pdq:
                       try:
                           arima_mod = ARIMA(df['Quantity'],order=param,exog=df[exog_var])
                           best_model =arima_mod.fit()  
                           if(d==range(0,1)):
                               pred = best_model.fittedvalues
                               res=mape(df['Quantity'],pred)
                           elif(d==range(1,2)):
                               pred = best_model.fittedvalues
                               #pred1 = pred.shift(periods=-1)
                               res=mape(df['Quantity'][1:],pred[1:])
                           #dfobj=dfobj.append({'param':param,'mape':res},ignore_index=True)
                           dfobj4=pd.concat([dfobj4,pd.DataFrame({'param':[param],'mape':[res]})]).reset_index(drop=True)
                       except:
                           continue
                    
      
                    
                    arima_results=dfobj.sort_values(by=['mape'])[:3].reset_index(drop=True)
                    arima_mape=arima_results[:1]['mape'].to_list()[0]
                    arima_order=arima_results[:1]['param'].to_list()[0]
                    
                    
                    sarima_results=dfobj1.sort_values(by=['mape'])[:3].reset_index(drop=True)
                    sarima_mape=sarima_results[:1]['mape'].to_list()[0]
                    sarima_order=sarima_results[:1]['param'].to_list()[0]
                    sarima_seasonal_order=sarima_results[:1]['seasonal'].to_list()[0]
                    
                    tes_results=dfobj2.sort_values(by=['mape'])[:3].reset_index(drop=True)
                    tes_mape=tes_results[:1]['mape'].to_list()[0]
                    tes_sp=tes_results[:1]['sp'].to_list()[0]
                
                    sarimax_results=dfobj3.sort_values(by=['mape'])[:3].reset_index(drop=True)
                    sarimax_mape=sarimax_results[:1]['mape'].to_list()[0]
                    sarimax_order=sarimax_results[:1]['param'].to_list()[0]
                    sarimax_seasonal_order=sarimax_results[:1]['seasonal'].to_list()[0]
                    
                    arimax_results=dfobj4.sort_values(by=['mape'])[:3].reset_index(drop=True)
                    arimax_mape=arimax_results[:1]['mape'].to_list()[0]
                    arimax_order=arimax_results[:1]['param'].to_list()[0]
                 
                    variables = {'x': arima_mape, 'y': sarima_mape, 'z': tes_mape,'a':sarimax_mape, 'u':arimax_mape}
                    largest_variable = min(variables, key=variables.get)
                    
   
            
                    if(largest_variable=='x'):
                        arima = ARIMA(df['Quantity'],order=arima_order)
                        best_model=arima.fit()
                        st.write('Arima_MAPE: ',round(arima_mape,2))
            
                        if(d==range(0,1)):
                           pred = best_model.fittedvalues
                           res=mape(df['Quantity'],pred)
                           
            
                           dfa=df.reset_index()
                        #dfa.rename(columns={'dt':'Date','Quantity':'Actual Production'},inplace=True)
            
                           dfp=pd.DataFrame(pred)
                           dfp=dfp.reset_index()
                           dfp.rename(columns={0:'Quantity'},inplace=True)
            
                           fore=best_model.predict(start=(pd.Series(df.index[-1])[0])+pd.DateOffset(months=1),end=(pd.Series(df.index[-1])[0])+ pd.DateOffset(months=fore_period))
                           fore_df=pd.DataFrame(data=fore.reset_index())
                           fore_df=fore_df.rename(columns={'index':'dt','predicted_mean':'Quantity'})
                           a=pd.DataFrame(pred.reset_index().iloc[-1]).T.rename(columns={0:'Quantity'}).reset_index(drop=True)
                           dff=pd.concat([a,fore_df],ignore_index=True)
                           dff['dt']=pd.to_datetime(dff['dt'])
                        #dff=dff.set_index(dff['Date'])
                        #dff=dff.drop(['Date'],axis=1)
            
                           fig = go.Figure()
                           fig.add_trace(go.Scatter(x=dfa['dt'], y=dfa['Quantity'], name='Actual Quantity', line=dict(color='blue')))
                           fig.add_trace(go.Scatter(x=dfp['dt'], y=dfp['Quantity'], name='Prediction', line=dict(color='orange')))
                           fig.add_trace(go.Scatter(x=dff['dt'], y=dff['Quantity'], name='Forecast', line=dict(color='green')))
                           st.plotly_chart(fig)
                           
                         
                           
                        elif(d==range(1,2)):
                           pred = best_model.fittedvalues
                           #pred = pred.shift(periods=-1)
                           res=mape(df['Quantity'][1:],pred[1:])
                           
            
                           dfa=df.reset_index()
                        #dfa.rename(columns={'dt':'Date','Quantity':'Actual Production'},inplace=True)
            
                           dfp=pd.DataFrame(pred)
                           dfp=dfp.reset_index()
                           dfp.rename(columns={0:'Quantity'},inplace=True)
            
                           fore=best_model.predict(start=(pd.Series(df.index[-1])[0])+ pd.DateOffset(months=1),end=(pd.Series(df.index[-1])[0])+ pd.DateOffset(months=fore_period))
                           fore_df=pd.DataFrame(data=fore.reset_index())
                           fore_df=fore_df.rename(columns={'index':'dt','predicted_mean':'Quantity'})
                           a=pd.DataFrame(pred.reset_index().iloc[-1]).T.rename(columns={0:'Quantity'}).reset_index(drop=True)
                           dff=pd.concat([a,fore_df],ignore_index=True)
                           dff['dt']=pd.to_datetime(dff['dt'])
                        #dff=dff.set_index(dff['Date'])
                        #dff=dff.drop(['Date'],axis=1)
            
                           fig = go.Figure()
                           fig.add_trace(go.Scatter(x=dfa['dt'][1:], y=dfa['Quantity'][1:], name='Actual Quantity', line=dict(color='blue')))
                           fig.add_trace(go.Scatter(x=dfp['dt'][1:], y=dfp['Quantity'][1:], name='Prediction', line=dict(color='orange')))
                           fig.add_trace(go.Scatter(x=dff['dt'], y=dff['Quantity'], name='Forecast', line=dict(color='green')))
                           st.plotly_chart(fig) 
            
                    if(largest_variable=='y'):
                       mod = sm.tsa.statespace.SARIMAX(df['Quantity'],order=sarima_order,seasonal_order=sarima_seasonal_order,
                                                       enforce_stationarity=False,enforce_invertibility=False)
                       best_model =mod.fit()
                       st.write('Sarima_MAPE: ',round(sarima_mape,2))
           
                       
                       pred = best_model.fittedvalues
                       #pred = pred.shift(periods=-1)
                       res=mape(df['Quantity'][1:],pred[1:])
                      
    
                       dfa=df.reset_index()
                   #dfa.rename(columns={'dt':'Date','Quantity':'Actual Production'},inplace=True)
    
                       dfp=pd.DataFrame(pred)
                       dfp=dfp.reset_index()
                       dfp.rename(columns={0:'Quantity'},inplace=True)
    
                       fore=best_model.predict(start=(pd.Series(df.index[-1])[0])+ pd.DateOffset(months=1),end=(pd.Series(df.index[-1])[0])+ pd.DateOffset(months=fore_period))
                       fore_df=pd.DataFrame(data=fore.reset_index())
                       fore_df=fore_df.rename(columns={'index':'dt','predicted_mean':'Quantity'})
                       a=pd.DataFrame(pred.reset_index().iloc[-1]).T.rename(columns={0:'Quantity'}).reset_index(drop=True)
                       dff=pd.concat([a,fore_df],ignore_index=True)
                       dff['dt']=pd.to_datetime(dff['dt'])
                   #dff=dff.set_index(dff['Date'])
                   #dff=dff.drop(['Date'],axis=1)
    
                       fig = go.Figure()
                       fig.add_trace(go.Scatter(x=dfa['dt'][1:], y=dfa['Quantity'][1:], name='Actual Quantity', line=dict(color='blue')))
                       fig.add_trace(go.Scatter(x=dfp['dt'][1:], y=dfp['Quantity'][1:], name='Prediction', line=dict(color='orange')))
                       fig.add_trace(go.Scatter(x=dff['dt'], y=dff['Quantity'], name='Forecast', line=dict(color='green')))
                       st.plotly_chart(fig) 
            
            
                    if(largest_variable=='z'):
                        tes = ExponentialSmoothing(df['Quantity'],trend='additive',seasonal='additive',seasonal_periods=tes_sp,
                                              initialization_method='estimated')
                        best_model =tes.fit()  
                        st.write('TES_MAPE: ',round(tes_mape,2))
                        pred = best_model.fittedvalues
                        #pred = pred.shift(periods=-1)
            
                        dfa=df.reset_index()
                        dfa.rename(columns={'dt':'Date','Quantity':'Actual Production'},inplace=True)
            
                        dfp=pd.DataFrame(pred)
                        dfp=dfp.reset_index()
                        dfp.rename(columns={'dt':'Date',0:'Predicted Production'},inplace=True)
            
                        fore=best_model.predict(start=(pd.Series(df.index[-1])[0])+ pd.DateOffset(months=1),end=(pd.Series(df.index[-1])[0])+ pd.DateOffset(months=fore_period))
                        fore_df=pd.DataFrame(data=fore.reset_index())
                        fore_df=fore_df.rename(columns={'index':'Date',0:'Forecasted Production'})
                        a=pd.DataFrame(pred.reset_index().iloc[-1]).T.rename(columns={'dt':'Date',0:'Forecasted Production'}).reset_index(drop=True)
                        dff=pd.concat([a,fore_df],ignore_index=True)
                        dff['Date']=pd.to_datetime(dff['Date'])
                        #dff=dff.set_index(dff['Date'])
                        #dff=dff.drop(['Date'],axis=1)
            
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=dfa['Date'], y=dfa['Actual Production'], name='Actual Production', line=dict(color='blue')))
                        fig.add_trace(go.Scatter(x=dfp['Date'], y=dfp['Predicted Production'], name='Predicted Production', line=dict(color='orange')))
                        fig.add_trace(go.Scatter(x=dff['Date'], y=dff['Forecasted Production'], name='Forecasted Production', line=dict(color='green')))
                        st.plotly_chart(fig)
                        
                   
                    
                    if(largest_variable=='a'):
                       mod = sm.tsa.statespace.SARIMAX(df['Quantity'],order=sarimax_order,seasonal_order=sarimax_seasonal_order,exog=df[exog_var],
                                                       enforce_stationarity=False,enforce_invertibility=False)
                       best_model =mod.fit()
                       st.write('Sarimax_MAPE: ',round(sarimax_mape,2))
           
                       
                       pred = best_model.fittedvalues
                       #pred = pred.shift(periods=-1)
                       res=mape(df['Quantity'][1:],pred[1:])
                      
    
                       dfa=df.reset_index()
                   #dfa.rename(columns={'dt':'Date','Quantity':'Actual Production'},inplace=True)
    
                       dfp=pd.DataFrame(pred)
                       dfp=dfp.reset_index()
                       dfp.rename(columns={0:'Quantity'},inplace=True)
    
                       fore=best_model.predict(start=(pd.Series(df.index[-1])[0])+ pd.DateOffset(months=1),end=(pd.Series(df.index[-1])[0])+ pd.DateOffset(months=fore_period),exog=df_exog[exog_var])
                       fore_df=pd.DataFrame(data=fore.reset_index())
                       fore_df=fore_df.rename(columns={'index':'dt','predicted_mean':'Quantity'})
                       a=pd.DataFrame(pred.reset_index().iloc[-1]).T.rename(columns={0:'Quantity'}).reset_index(drop=True)
                       dff=pd.concat([a,fore_df],ignore_index=True)
                       dff['dt']=pd.to_datetime(dff['dt'])
                   #dff=dff.set_index(dff['Date'])
                   #dff=dff.drop(['Date'],axis=1)
    
                       fig = go.Figure()
                       fig.add_trace(go.Scatter(x=dfa['dt'][1:], y=dfa['Quantity'][1:], name='Actual Quantity', line=dict(color='blue')))
                       fig.add_trace(go.Scatter(x=dfp['dt'][1:], y=dfp['Quantity'][1:], name='Prediction', line=dict(color='orange')))
                       fig.add_trace(go.Scatter(x=dff['dt'], y=dff['Quantity'], name='Forecast', line=dict(color='green')))
                       st.plotly_chart(fig)
                       
                    #
                    if(largest_variable=='u'):
                        
                        
                        arimax = sm.tsa.statespace.SARIMAX(df['Quantity'],order=arimax_order,seasonal_order=(0,0,0,0),exog = df[exog_var])
                        best_model=arimax.fit()
                        st.write('Arimax_MAPE: ',round(arimax_mape,2))
            
                        pred = best_model.fittedvalues
                        #pred = pred.shift(periods=-1)
                        res=mape(df['Quantity'][1:],pred[1:])
                       
    
                        dfa=df.reset_index()
                    #dfa.rename(columns={'dt':'Date','Quantity':'Actual Production'},inplace=True)
    
                        dfp=pd.DataFrame(pred)
                        dfp=dfp.reset_index()
                        dfp.rename(columns={0:'Quantity'},inplace=True)
    
                        fore=best_model.predict(start=(pd.Series(df.index[-1])[0])+ pd.DateOffset(months=1),end=(pd.Series(df.index[-1])[0])+ pd.DateOffset(months=fore_period),exog = df_exog[exog_var])
                        fore_df=pd.DataFrame(data=fore.reset_index())
                        fore_df=fore_df.rename(columns={'index':'dt','predicted_mean':'Quantity'})
                        a=pd.DataFrame(pred.reset_index().iloc[-1]).T.rename(columns={0:'Quantity'}).reset_index(drop=True)
                        dff=pd.concat([a,fore_df],ignore_index=True)
                        dff['dt']=pd.to_datetime(dff['dt'])
                    #dff=dff.set_index(dff['Date'])
                    #dff=dff.drop(['Date'],axis=1)
    
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=dfa['dt'][1:], y=dfa['Quantity'][1:], name='Actual Quantity', line=dict(color='blue')))
                        fig.add_trace(go.Scatter(x=dfp['dt'][1:], y=dfp['Quantity'][1:], name='Prediction', line=dict(color='orange')))
                        fig.add_trace(go.Scatter(x=dff['dt'], y=dff['Quantity'], name='Forecast', line=dict(color='green')))
                        st.plotly_chart(fig)
    
                  
                    arima_results['Model']='ARIMA'
                    sarima_results['Model']='SARIMA'
                    tes_results['Model']='TES'
                    sarimax_results['Model']='SARIMAX'
                    arimax_results['Model']='ARIMAX'
          
                    
                    ff=pd.concat([arima_results,sarima_results])
                    ff=pd.concat([ff,tes_results])
                    ff=pd.concat([ff,sarimax_results])
                    ff=pd.concat([ff,arimax_results])
                   
                    g = ff.sort_values(by='mape').reset_index(drop=True)[:3]
                    st.subheader('Top models')
                    st.table(g)
                    
                    
                    #if st.checkbox('SARIMAX Results',value=False):
                    st.subheader('SARIMAX Results')
                    st.table(sarimax_results)
                    
                    writer = pd.ExcelWriter('data.xlsx', engine='openpyxl')
                    fore_df.to_excel(writer, sheet_name='Sheet1', index=False)
                    writer.close()
                    with open('data.xlsx', 'rb') as f:
                        excel_data = f.read()
                        st.download_button(label='Click here to download', data=excel_data, file_name='data.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
                        st.stop()
                    
                else:
                    
                    
                    date=st.selectbox('**Select the date column**',['Select an option']+list(df.columns))
                    target=st.selectbox('**Select the target column**',['Select an option']+list(df.columns))
                    fore_period=st.selectbox("Select the number of forecasting months:", ['',3,6])
                    fore_period=int(fore_period)
                    
                    try:
                        df.rename(columns={date:'dt',target:'Quantity'},inplace=True)
                        df['dt']=pd.to_datetime(df['dt'])
                        df = df.set_index('dt')
                    except:
                        pass
                   # df = pd.DataFrame()
                    #df['dt'] = dt
                    #df['Quantity'] = df1['Quantity']
            
                 
                    
                    p = range(1,5)
                    q = range(1,5)
                    result = adfuller(df['Quantity'])
                    if(result[1]<0.05):
                        d = range(0,1)
                    elif(result[1]>0.05):
                        d = range(1,2)
            
                    pdq = list(itertools.product(p, d, q))
                    
                    z = (12,1)
            
                    a = [[(x[0],x[1],x[2],m)             for m in z         if(m>x[0]  and m>x[2]) ]       for x in list(pdq)]
            
                    model_pdq = [item for sublist in a for item in sublist]
            
                    dfobj = pd.DataFrame(columns=['param', 'mape'])
                    dfobj1 = pd.DataFrame(columns=['param','seasonal', 'mape'])
                    dfobj2 = pd.DataFrame(columns=['sp','mape'])
            
            
                    #ARIMA
            
                    for param in pdq:
                       try:
                           arima_mod = ARIMA(df['Quantity'],order=param)
                           best_model =arima_mod.fit()  
                           if(d==range(0,1)):
                               pred = best_model.fittedvalues
                               res=mape(df['Quantity'],pred)
                           elif(d==range(1,2)):
                               pred = best_model.fittedvalues
                               #pred1 = pred.shift(periods=-1)
                               res=mape(df['Quantity'][1:],pred[1:])
                           #dfobj=dfobj.append({'param':param,'mape':res},ignore_index=True)
                           dfobj=pd.concat([dfobj,pd.DataFrame({'param':[param],'mape':[res]})]).reset_index(drop=True)
                       except:
                           continue
                    #SARIMA
            
                    for param in pdq:
                        for param_seasonal in model_pdq:
                            try:
                                mod = sm.tsa.statespace.SARIMAX(df['Quantity'],
                                                                    order=param,
                                                                    seasonal_order=param_seasonal,
                                                                    enforce_stationarity=False,
                                                                    enforce_invertibility=False)
                                best_model =mod.fit()   
                                
                                pred = best_model.fittedvalues
                                #pred1 = pred.shift(periods=-1)
                                res=mape(df['Quantity'][1:],pred[1:])
                                #dfobj1 = dfobj1.append({'param':param,'seasonal':param_seasonal ,'mape': res}, ignore_index=True)
                                dfobj1=pd.concat([dfobj1,pd.DataFrame({'param':[param],'seasonal':[param_seasonal] ,'mape': [res]})]).reset_index(drop=True)
                            except:
                                continue
            
                    #TES
            
                    for sp in range(2,11,2):
                        tes = ExponentialSmoothing(df['Quantity'],trend='additive',seasonal='additive',seasonal_periods=sp,initialization_method='estimated')
                        tes_model = tes.fit()
                        pred = tes_model.fittedvalues
                        res=mape(df['Quantity'],pred)
                        #dfobj2 = dfobj2.append({'sp':sp,'mape': res}, ignore_index=True)
                        dfobj2=pd.concat([dfobj2,pd.DataFrame({'sp':[sp],'mape':[res]})]).reset_index(drop=True)
            
            
                    
                    
                    arima_results=dfobj.sort_values(by=['mape'])[:3].reset_index(drop=True)
                    arima_mape=arima_results[:1]['mape'].to_list()[0]
                    arima_order=arima_results[:1]['param'].to_list()[0]
                    
                    sarima_results=dfobj1.sort_values(by=['mape'])[:3].reset_index(drop=True)
                    sarima_mape=sarima_results[:1]['mape'].to_list()[0]
                    sarima_order=sarima_results[:1]['param'].to_list()[0]
                    sarima_seasonal_order=sarima_results[:1]['seasonal'].to_list()[0]
                    
                    tes_results=dfobj2.sort_values(by=['mape'])[:3].reset_index(drop=True)
                    tes_mape=tes_results[:1]['mape'].to_list()[0]
                    tes_sp=tes_results[:1]['sp'].to_list()[0]
            
                    variables = {'x': arima_mape, 'y': sarima_mape, 'z': tes_mape}
                    largest_variable = min(variables, key=variables.get)

            
            
                    if(largest_variable=='x'):
                        arima = ARIMA(df['Quantity'],order=arima_order)
                        best_model=arima.fit()
                        st.write('Arima_MAPE: ',round(arima_mape,2))
            
                        if(d==range(0,1)):
                           pred = best_model.fittedvalues
                           res=mape(df['Quantity'],pred)
                           
            
                           dfa=df.reset_index()
                        #dfa.rename(columns={'dt':'Date','Quantity':'Actual Production'},inplace=True)
            
                           dfp=pd.DataFrame(pred)
                           dfp=dfp.reset_index()
                           dfp.rename(columns={0:'Quantity'},inplace=True)
            
                           fore=best_model.predict(start=(pd.Series(df.index[-1])[0])+pd.DateOffset(months=1),end=(pd.Series(df.index[-1])[0])+ pd.DateOffset(months=fore_period))
                           fore_df=pd.DataFrame(data=fore.reset_index())
                           fore_df=fore_df.rename(columns={'index':'dt','predicted_mean':'Quantity'})
                           a=pd.DataFrame(pred.reset_index().iloc[-1]).T.rename(columns={0:'Quantity'}).reset_index(drop=True)
                           dff=pd.concat([a,fore_df],ignore_index=True)
                           dff['dt']=pd.to_datetime(dff['dt'])
                        #dff=dff.set_index(dff['Date'])
                        #dff=dff.drop(['Date'],axis=1)
            
                           fig = go.Figure()
                           fig.add_trace(go.Scatter(x=dfa['dt'], y=dfa['Quantity'], name='Actual Quantity', line=dict(color='blue')))
                           fig.add_trace(go.Scatter(x=dfp['dt'], y=dfp['Quantity'], name='Prediction', line=dict(color='orange')))
                           fig.add_trace(go.Scatter(x=dff['dt'], y=dff['Quantity'], name='Forecast', line=dict(color='green')))
                           st.plotly_chart(fig)
                           
                        elif(d==range(1,2)):
                           pred = best_model.fittedvalues
                           #pred = pred.shift(periods=-1)
                           res=mape(df['Quantity'][1:],pred[1:])
                           
            
                           dfa=df.reset_index()
                        #dfa.rename(columns={'dt':'Date','Quantity':'Actual Production'},inplace=True)
            
                           dfp=pd.DataFrame(pred)
                           dfp=dfp.reset_index()
                           dfp.rename(columns={0:'Quantity'},inplace=True)
            
                           fore=best_model.predict(start=(pd.Series(df.index[-1])[0])+ pd.DateOffset(months=1),end=(pd.Series(df.index[-1])[0])+ pd.DateOffset(months=fore_period))
                           fore_df=pd.DataFrame(data=fore.reset_index())
                           fore_df=fore_df.rename(columns={'index':'dt','predicted_mean':'Quantity'})
                           a=pd.DataFrame(pred.reset_index().iloc[-1]).T.rename(columns={0:'Quantity'}).reset_index(drop=True)
                           dff=pd.concat([a,fore_df],ignore_index=True)
                           dff['dt']=pd.to_datetime(dff['dt'])
                        #dff=dff.set_index(dff['Date'])
                        #dff=dff.drop(['Date'],axis=1)
            
                           fig = go.Figure()
                           fig.add_trace(go.Scatter(x=dfa['dt'][1:], y=dfa['Quantity'][1:], name='Actual Quantity', line=dict(color='blue')))
                           fig.add_trace(go.Scatter(x=dfp['dt'][1:], y=dfp['Quantity'][1:], name='Prediction', line=dict(color='orange')))
                           fig.add_trace(go.Scatter(x=dff['dt'], y=dff['Quantity'], name='Forecast', line=dict(color='green')))
                           st.plotly_chart(fig) 
            
                    if(largest_variable=='y'):
                       mod = sm.tsa.statespace.SARIMAX(df['Quantity'],order=sarima_order,seasonal_order=sarima_seasonal_order,
                                                       enforce_stationarity=False,enforce_invertibility=False)
                       best_model =mod.fit()
                       st.write('Sarima_MAPE: ',round(sarima_mape,2))
           
                       
                       pred = best_model.fittedvalues
                       #pred = pred.shift(periods=-1)
                       res=mape(df['Quantity'][1:],pred[1:])
                      
    
                       dfa=df.reset_index()
                   #dfa.rename(columns={'dt':'Date','Quantity':'Actual Production'},inplace=True)
    
                       dfp=pd.DataFrame(pred)
                       dfp=dfp.reset_index()
                       dfp.rename(columns={0:'Quantity'},inplace=True)
    
                       fore=best_model.predict(start=(pd.Series(df.index[-1])[0])+ pd.DateOffset(months=1),end=(pd.Series(df.index[-1])[0])+ pd.DateOffset(months=fore_period))
                       fore_df=pd.DataFrame(data=fore.reset_index())
                       fore_df=fore_df.rename(columns={'index':'dt','predicted_mean':'Quantity'})
                       a=pd.DataFrame(pred.reset_index().iloc[-1]).T.rename(columns={0:'Quantity'}).reset_index(drop=True)
                       dff=pd.concat([a,fore_df],ignore_index=True)
                       dff['dt']=pd.to_datetime(dff['dt'])
                   #dff=dff.set_index(dff['Date'])
                   #dff=dff.drop(['Date'],axis=1)
    
                       fig = go.Figure()
                       fig.add_trace(go.Scatter(x=dfa['dt'][1:], y=dfa['Quantity'][1:], name='Actual Quantity', line=dict(color='blue')))
                       fig.add_trace(go.Scatter(x=dfp['dt'][1:], y=dfp['Quantity'][1:], name='Prediction', line=dict(color='orange')))
                       fig.add_trace(go.Scatter(x=dff['dt'], y=dff['Quantity'], name='Forecast', line=dict(color='green')))
                       st.plotly_chart(fig) 
            
            
                    if(largest_variable=='z'):
                        tes = ExponentialSmoothing(df['Quantity'],trend='additive',seasonal='additive',seasonal_periods=tes_sp,
                                              initialization_method='estimated')
                        best_model =tes.fit()  
                        st.write('TES_MAPE: ',round(tes_mape,2))
                        pred = best_model.fittedvalues
                        #pred = pred.shift(periods=-1)
            
                        dfa=df.reset_index()
                        dfa.rename(columns={'dt':'Date','Quantity':'Actual Production'},inplace=True)
            
                        dfp=pd.DataFrame(pred)
                        dfp=dfp.reset_index()
                        dfp.rename(columns={'dt':'Date',0:'Predicted Production'},inplace=True)
            
                        fore=best_model.predict(start=(pd.Series(df.index[-1])[0])+ pd.DateOffset(months=1),end=(pd.Series(df.index[-1])[0])+ pd.DateOffset(months=fore_period))
                        fore_df=pd.DataFrame(data=fore.reset_index())
                        fore_df=fore_df.rename(columns={'index':'Date',0:'Forecasted Production'})
                        a=pd.DataFrame(pred.reset_index().iloc[-1]).T.rename(columns={'dt':'Date',0:'Forecasted Production'}).reset_index(drop=True)
                        dff=pd.concat([a,fore_df],ignore_index=True)
                        dff['Date']=pd.to_datetime(dff['Date'])
                        #dff=dff.set_index(dff['Date'])
                        #dff=dff.drop(['Date'],axis=1)
            
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=dfa['Date'], y=dfa['Actual Production'], name='Actual Production', line=dict(color='blue')))
                        fig.add_trace(go.Scatter(x=dfp['Date'], y=dfp['Predicted Production'], name='Predicted Production', line=dict(color='orange')))
                        fig.add_trace(go.Scatter(x=dff['Date'], y=dff['Forecasted Production'], name='Forecasted Production', line=dict(color='green')))
                        st.plotly_chart(fig)
                        
                    arima_results['Model']='ARIMA'
                    sarima_results['Model']='SARIMA'
                    tes_results['Model']='TES'
                 
                    ff=pd.concat([arima_results,sarima_results])
                    ff=pd.concat([ff,tes_results])
               
                    g = ff.sort_values(by='mape').reset_index(drop=True)[:3]
                    st.table(g)
                    
                    writer = pd.ExcelWriter('data.xlsx', engine='openpyxl')
                    fore_df.to_excel(writer, sheet_name='Sheet1', index=False)
                    writer.close()
                    with open('data.xlsx', 'rb') as f:
                        excel_data = f.read()
                        st.download_button(label='Click here to download', data=excel_data, file_name='data.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
                        st.stop()
    except:
        st.write(" Select the above options to continue")               
if rad=='Daily':
    try:

        uploaded_files = st.file_uploader("Upload Files", accept_multiple_files=True)
        
        filelist = []
        
        for file in uploaded_files:
        
            filelist.append(file.name)
        
        selected_file = st.selectbox("Select a file:", filelist)
        
        
        for i in uploaded_files:
            if(i.name==selected_file):
                df = pd.read_excel(i)
                c_box_d=st.checkbox('Exog Variable')
                
                
                if(c_box_d==True):
                    
                    date=st.selectbox('**Select the date column**',['Select an option']+list(df.columns))
                    target=st.selectbox('**Select the target column**',['Select an option']+list(df.columns))
                    exog=st.multiselect('**Select the exog column**',['Select an option']+list(df.columns))
                    
                    fore_period=st.selectbox("Select the number of forecasting days:", ['',5,10])
                    fore_period=int(fore_period)
                      
                    
                    
                    exog_var=[]
        
                    for i in range(len(exog)):
                        exog_var.append(exog[i])
                        
                    file1 = st.file_uploader('Upload a file for Exog variables',key='f1')
    
                    if file1 is not None:
                        df_exog=pd.read_excel(file1,engine="openpyxl")
                
                    try:
                        df.rename(columns={date:'dt',target:'Quantity'},inplace=True)
                        df['dt']=pd.to_datetime(df['dt'])
                        df = df.set_index('dt')
                    except:
                        pass
                   # df = pd.DataFrame()
                    #df['dt'] = dt
                    #df['Quantity'] = df1['Quantity']
            
                 
            
                    p = range(1,5)
                    q = range(1,5)
                    result = adfuller(df['Quantity'])
                    if(result[1]<0.05):
                        d = range(0,1)
                    elif(result[1]>0.05):
                        d = range(1,2)
            
                    pdq = list(itertools.product(p, d, q))
                    
                    z = (7,1)
            
                    a = [[(x[0],x[1],x[2],m)             for m in z         if(m>x[0]  and m>x[2]) ]       for x in list(pdq)]
            
                    model_pdq = [item for sublist in a for item in sublist]
            
                    dfobj = pd.DataFrame(columns=['param', 'mape'])
                    dfobj1 = pd.DataFrame(columns=['param','seasonal', 'mape'])
                    dfobj2 = pd.DataFrame(columns=['sp','mape'])
                    dfobj3=pd.DataFrame(columns=['param','seasonal', 'mape'])
                    dfobj4=pd.DataFrame(columns=['param', 'mape'])
            
            
                    #ARIMA
            
                    for param in pdq:
                       try:
                           arima_mod = ARIMA(df['Quantity'],order=param)
                           best_model =arima_mod.fit()  
                           if(d==range(0,1)):
                               pred = best_model.fittedvalues
                               res=mape(df['Quantity'],pred)
                           elif(d==range(1,2)):
                               pred = best_model.fittedvalues
                               #pred1 = pred.shift(periods=-1)
                               res=mape(df['Quantity'][1:],pred[1:])
                           #dfobj=dfobj.append({'param':param,'mape':res},ignore_index=True)
                           dfobj=pd.concat([dfobj,pd.DataFrame({'param':[param],'mape':[res]})]).reset_index(drop=True)
                       except:
                           continue
                    #SARIMA
            
                    for param in pdq:
                        for param_seasonal in model_pdq:
                            try:
                                mod = sm.tsa.statespace.SARIMAX(df['Quantity'],
                                                                    order=param,
                                                                    seasonal_order=param_seasonal,
                                                                    enforce_stationarity=False,
                                                                    enforce_invertibility=False)
                                best_model =mod.fit()   
                                
                                pred = best_model.fittedvalues
                                #pred1 = pred.shift(periods=-1)
                                res=mape(df['Quantity'][1:],pred[1:])
                                #dfobj1 = dfobj1.append({'param':param,'seasonal':param_seasonal ,'mape': res}, ignore_index=True)
                                dfobj1=pd.concat([dfobj1,pd.DataFrame({'param':[param],'seasonal':[param_seasonal] ,'mape': [res]})]).reset_index(drop=True)
                            except:
                                continue
                            
                            
            
                    #TES
            
                    for sp in range(2,11,2):
                        tes = ExponentialSmoothing(df['Quantity'],trend='additive',seasonal='additive',seasonal_periods=sp,initialization_method='estimated')
                        tes_model = tes.fit()
                        pred = tes_model.fittedvalues
                        res=mape(df['Quantity'],pred)
                        #dfobj2 = dfobj2.append({'sp':sp,'mape': res}, ignore_index=True)
                        dfobj2=pd.concat([dfobj2,pd.DataFrame({'sp':[sp],'mape':[res]})]).reset_index(drop=True)
                        
                        
                    #SARIMAX
                    
                    for param in pdq:
                        for param_seasonal in model_pdq:
                            try:
                                mod = sm.tsa.statespace.SARIMAX(df['Quantity'],
                                                                    order=param,
                                                                    seasonal_order=param_seasonal,
                                                                    exog=df[exog_var],
                                                                    enforce_stationarity=False,
                                                                    enforce_invertibility=False)
                                best_model =mod.fit()   
                                
                                pred = best_model.fittedvalues
                                #pred1 = pred.shift(periods=-1)
                                res=mape(df['Quantity'][1:],pred[1:])
                                #dfobj1 = dfobj1.append({'param':param,'seasonal':param_seasonal ,'mape': res}, ignore_index=True)
                                dfobj3=pd.concat([dfobj3,pd.DataFrame({'param':[param],'seasonal':[param_seasonal] ,'mape': [res]})]).reset_index(drop=True)
                            except:
                                continue
                    #ARIMAX
      
            
                    for param in pdq:
                       try:
                           arima_mod = ARIMA(df['Quantity'],order=param,exog=df[exog_var])
                           best_model =arima_mod.fit()  
                           if(d==range(0,1)):
                               pred = best_model.fittedvalues
                               res=mape(df['Quantity'],pred)
                           elif(d==range(1,2)):
                               pred = best_model.fittedvalues
                               #pred1 = pred.shift(periods=-1)
                               res=mape(df['Quantity'][1:],pred[1:])
                           #dfobj=dfobj.append({'param':param,'mape':res},ignore_index=True)
                           dfobj4=pd.concat([dfobj4,pd.DataFrame({'param':[param],'mape':[res]})]).reset_index(drop=True)
                       except:
                           continue
                    
                    #ARIMAX
      
            
                    for param in pdq:
                       try:
                           arima_mod = ARIMA(df['Quantity'],order=param,exog=df[exog_var])
                           best_model =arima_mod.fit()  
                           if(d==range(0,1)):
                               pred = best_model.fittedvalues
                               res=mape(df['Quantity'],pred)
                           elif(d==range(1,2)):
                               pred = best_model.fittedvalues
                               #pred1 = pred.shift(periods=-1)
                               res=mape(df['Quantity'][1:],pred[1:])
                           #dfobj=dfobj.append({'param':param,'mape':res},ignore_index=True)
                           dfobj4=pd.concat([dfobj4,pd.DataFrame({'param':[param],'mape':[res]})]).reset_index(drop=True)
                       except:
                           continue
      
                    
                    
                    arima_results=dfobj.sort_values(by=['mape'])[:3].reset_index(drop=True)
                    arima_mape=arima_results[:1]['mape'].to_list()[0]
                    arima_order=arima_results[:1]['param'].to_list()[0]
                    
                    
                    sarima_results=dfobj1.sort_values(by=['mape'])[:3].reset_index(drop=True)
                    sarima_mape=sarima_results[:1]['mape'].to_list()[0]
                    sarima_order=sarima_results[:1]['param'].to_list()[0]
                    sarima_seasonal_order=sarima_results[:1]['seasonal'].to_list()[0]
                    
                    tes_results=dfobj2.sort_values(by=['mape'])[:3].reset_index(drop=True)
                    tes_mape=tes_results[:1]['mape'].to_list()[0]
                    tes_sp=tes_results[:1]['sp'].to_list()[0]
                
                    sarimax_results=dfobj3.sort_values(by=['mape'])[:3].reset_index(drop=True)
                    sarimax_mape=sarimax_results[:1]['mape'].to_list()[0]
                    sarimax_order=sarimax_results[:1]['param'].to_list()[0]
                    sarimax_seasonal_order=sarimax_results[:1]['seasonal'].to_list()[0]
                    
                    arimax_results=dfobj4.sort_values(by=['mape'])[:3].reset_index(drop=True)
                    arimax_mape=arimax_results[:1]['mape'].to_list()[0]
                    arimax_order=arimax_results[:1]['param'].to_list()[0]
                 
                    variables = {'x': arima_mape, 'y': sarima_mape, 'z': tes_mape,'a':sarimax_mape, 'u':arimax_mape}
                    largest_variable = min(variables, key=variables.get)
                    

            
                    if(largest_variable=='x'):
                        arima = ARIMA(df['Quantity'],order=arima_order)
                        best_model=arima.fit()
                        st.write('Arima_MAPE: ',round(arima_mape,2))
            
                        if(d==range(0,1)):
                           pred = best_model.fittedvalues
                           res=mape(df['Quantity'],pred)
                           
            
                           dfa=df.reset_index()
                        #dfa.rename(columns={'dt':'Date','Quantity':'Actual Production'},inplace=True)
            
                           dfp=pd.DataFrame(pred)
                           dfp=dfp.reset_index()
                           dfp.rename(columns={0:'Quantity'},inplace=True)
            
                           fore=best_model.predict(start=(pd.Series(df.index[-1])[0])+pd.DateOffset(days=1),end=(pd.Series(df.index[-1])[0])+ pd.DateOffset(days=fore_period))
                           fore_df=pd.DataFrame(data=fore.reset_index())
                           fore_df=fore_df.rename(columns={'index':'dt','predicted_mean':'Quantity'})
                           a=pd.DataFrame(pred.reset_index().iloc[-1]).T.rename(columns={0:'Quantity'}).reset_index(drop=True)
                           dff=pd.concat([a,fore_df],ignore_index=True)
                           dff['dt']=pd.to_datetime(dff['dt'])
                        #dff=dff.set_index(dff['Date'])
                        #dff=dff.drop(['Date'],axis=1)
            
                           fig = go.Figure()
                           fig.add_trace(go.Scatter(x=dfa['dt'], y=dfa['Quantity'], name='Actual Quantity', line=dict(color='blue')))
                           fig.add_trace(go.Scatter(x=dfp['dt'], y=dfp['Quantity'], name='Prediction', line=dict(color='orange')))
                           fig.add_trace(go.Scatter(x=dff['dt'], y=dff['Quantity'], name='Forecast', line=dict(color='green')))
                           st.plotly_chart(fig)
                           
                          
                           
                        elif(d==range(1,2)):
                           pred = best_model.fittedvalues
                           #pred = pred.shift(periods=-1)
                           res=mape(df['Quantity'][1:],pred[1:])
                           
            
                           dfa=df.reset_index()
                        #dfa.rename(columns={'dt':'Date','Quantity':'Actual Production'},inplace=True)
            
                           dfp=pd.DataFrame(pred)
                           dfp=dfp.reset_index()
                           dfp.rename(columns={0:'Quantity'},inplace=True)
            
                           fore=best_model.predict(start=(pd.Series(df.index[-1])[0])+ pd.DateOffset(days=1),end=(pd.Series(df.index[-1])[0])+ pd.DateOffset(days=fore_period))
                           fore_df=pd.DataFrame(data=fore.reset_index())
                           fore_df=fore_df.rename(columns={'index':'dt','predicted_mean':'Quantity'})
                           a=pd.DataFrame(pred.reset_index().iloc[-1]).T.rename(columns={0:'Quantity'}).reset_index(drop=True)
                           dff=pd.concat([a,fore_df],ignore_index=True)
                           dff['dt']=pd.to_datetime(dff['dt'])
                        #dff=dff.set_index(dff['Date'])
                        #dff=dff.drop(['Date'],axis=1)
            
                           fig = go.Figure()
                           fig.add_trace(go.Scatter(x=dfa['dt'][1:], y=dfa['Quantity'][1:], name='Actual Quantity', line=dict(color='blue')))
                           fig.add_trace(go.Scatter(x=dfp['dt'][1:], y=dfp['Quantity'][1:], name='Prediction', line=dict(color='orange')))
                           fig.add_trace(go.Scatter(x=dff['dt'], y=dff['Quantity'], name='Forecast', line=dict(color='green')))
                           st.plotly_chart(fig) 
            
                    if(largest_variable=='y'):
                       mod = sm.tsa.statespace.SARIMAX(df['Quantity'],order=sarima_order,seasonal_order=sarima_seasonal_order,
                                                       enforce_stationarity=False,enforce_invertibility=False)
                       best_model =mod.fit()
                       st.write('Sarima_MAPE: ',round(sarima_mape,2))
           
                       
                       pred = best_model.fittedvalues
                       #pred = pred.shift(periods=-1)
                       res=mape(df['Quantity'][1:],pred[1:])
                      
    
                       dfa=df.reset_index()
                   #dfa.rename(columns={'dt':'Date','Quantity':'Actual Production'},inplace=True)
    
                       dfp=pd.DataFrame(pred)
                       dfp=dfp.reset_index()
                       dfp.rename(columns={0:'Quantity'},inplace=True)
    
                       fore=best_model.predict(start=(pd.Series(df.index[-1])[0])+ pd.DateOffset(days=1),end=(pd.Series(df.index[-1])[0])+ pd.DateOffset(days=fore_period))
                       fore_df=pd.DataFrame(data=fore.reset_index())
                       fore_df=fore_df.rename(columns={'index':'dt','predicted_mean':'Quantity'})
                       a=pd.DataFrame(pred.reset_index().iloc[-1]).T.rename(columns={0:'Quantity'}).reset_index(drop=True)
                       dff=pd.concat([a,fore_df],ignore_index=True)
                       dff['dt']=pd.to_datetime(dff['dt'])
                   #dff=dff.set_index(dff['Date'])
                   #dff=dff.drop(['Date'],axis=1)
    
                       fig = go.Figure()
                       fig.add_trace(go.Scatter(x=dfa['dt'][1:], y=dfa['Quantity'][1:], name='Actual Quantity', line=dict(color='blue')))
                       fig.add_trace(go.Scatter(x=dfp['dt'][1:], y=dfp['Quantity'][1:], name='Prediction', line=dict(color='orange')))
                       fig.add_trace(go.Scatter(x=dff['dt'], y=dff['Quantity'], name='Forecast', line=dict(color='green')))
                       st.plotly_chart(fig) 
            
            
                    if(largest_variable=='z'):
                        tes = ExponentialSmoothing(df['Quantity'],trend='additive',seasonal='additive',seasonal_periods=tes_sp,
                                              initialization_method='estimated')
                        best_model =tes.fit()  
                        st.write('TES_MAPE: ',round(tes_mape,2))
                        pred = best_model.fittedvalues
                        #pred = pred.shift(periods=-1)
            
                        dfa=df.reset_index()
                        dfa.rename(columns={'dt':'Date','Quantity':'Actual Production'},inplace=True)
            
                        dfp=pd.DataFrame(pred)
                        dfp=dfp.reset_index()
                        dfp.rename(columns={'dt':'Date',0:'Predicted Production'},inplace=True)
            
                        fore=best_model.predict(start=(pd.Series(df.index[-1])[0])+ pd.DateOffset(days=1),end=(pd.Series(df.index[-1])[0])+ pd.DateOffset(days=fore_period))
                        fore_df=pd.DataFrame(data=fore.reset_index())
                        fore_df=fore_df.rename(columns={'index':'Date',0:'Forecasted Production'})
                        a=pd.DataFrame(pred.reset_index().iloc[-1]).T.rename(columns={'dt':'Date',0:'Forecasted Production'}).reset_index(drop=True)
                        dff=pd.concat([a,fore_df],ignore_index=True)
                        dff['Date']=pd.to_datetime(dff['Date'])
                        #dff=dff.set_index(dff['Date'])
                        #dff=dff.drop(['Date'],axis=1)
            
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=dfa['Date'], y=dfa['Actual Production'], name='Actual Production', line=dict(color='blue')))
                        fig.add_trace(go.Scatter(x=dfp['Date'], y=dfp['Predicted Production'], name='Predicted Production', line=dict(color='orange')))
                        fig.add_trace(go.Scatter(x=dff['Date'], y=dff['Forecasted Production'], name='Forecasted Production', line=dict(color='green')))
                        st.plotly_chart(fig)
                        
                   
                    
                    if(largest_variable=='a'):
                       mod = sm.tsa.statespace.SARIMAX(df['Quantity'],order=sarimax_order,seasonal_order=sarimax_seasonal_order,exog=df[exog_var],
                                                       enforce_stationarity=False,enforce_invertibility=False)
                       best_model =mod.fit()
                       st.write('Sarimax_MAPE: ',round(sarimax_mape,2))
           
                       
                       pred = best_model.fittedvalues
                       #pred = pred.shift(periods=-1)
                       res=mape(df['Quantity'][1:],pred[1:])
                      
    
                       dfa=df.reset_index()
                   #dfa.rename(columns={'dt':'Date','Quantity':'Actual Production'},inplace=True)
    
                       dfp=pd.DataFrame(pred)
                       dfp=dfp.reset_index()
                       dfp.rename(columns={0:'Quantity'},inplace=True)
    
                       fore=best_model.predict(start=(pd.Series(df.index[-1])[0])+ pd.DateOffset(days=1),end=(pd.Series(df.index[-1])[0])+ pd.DateOffset(days=fore_period),exog=df_exog[exog_var])
                       fore_df=pd.DataFrame(data=fore.reset_index())
                       fore_df=fore_df.rename(columns={'index':'dt','predicted_mean':'Quantity'})
                       a=pd.DataFrame(pred.reset_index().iloc[-1]).T.rename(columns={0:'Quantity'}).reset_index(drop=True)
                       dff=pd.concat([a,fore_df],ignore_index=True)
                       dff['dt']=pd.to_datetime(dff['dt'])
                   #dff=dff.set_index(dff['Date'])
                   #dff=dff.drop(['Date'],axis=1)
    
                       fig = go.Figure()
                       fig.add_trace(go.Scatter(x=dfa['dt'][1:], y=dfa['Quantity'][1:], name='Actual Quantity', line=dict(color='blue')))
                       fig.add_trace(go.Scatter(x=dfp['dt'][1:], y=dfp['Quantity'][1:], name='Prediction', line=dict(color='orange')))
                       fig.add_trace(go.Scatter(x=dff['dt'], y=dff['Quantity'], name='Forecast', line=dict(color='green')))
                       st.plotly_chart(fig)
                       
                    #
                    if(largest_variable=='u'):
                        
                        
                        arimax = sm.tsa.statespace.SARIMAX(df['Quantity'],order=arimax_order,seasonal_order=(0,0,0,0),exog = df[exog_var])
                        best_model=arimax.fit()
                        st.write('Arimax_MAPE: ',round(arimax_mape,2))
            
                        pred = best_model.fittedvalues
                        #pred = pred.shift(periods=-1)
                        res=mape(df['Quantity'][1:],pred[1:])
                       
    
                        dfa=df.reset_index()
                    #dfa.rename(columns={'dt':'Date','Quantity':'Actual Production'},inplace=True)
    
                        dfp=pd.DataFrame(pred)
                        dfp=dfp.reset_index()
                        dfp.rename(columns={0:'Quantity'},inplace=True)
    
                        fore=best_model.predict(start=(pd.Series(df.index[-1])[0])+ pd.DateOffset(days=1),end=(pd.Series(df.index[-1])[0])+ pd.DateOffset(days=fore_period),exog = df_exog[exog_var])
                        fore_df=pd.DataFrame(data=fore.reset_index())
                        fore_df=fore_df.rename(columns={'index':'dt','predicted_mean':'Quantity'})
                        a=pd.DataFrame(pred.reset_index().iloc[-1]).T.rename(columns={0:'Quantity'}).reset_index(drop=True)
                        dff=pd.concat([a,fore_df],ignore_index=True)
                        dff['dt']=pd.to_datetime(dff['dt'])
                    #dff=dff.set_index(dff['Date'])
                    #dff=dff.drop(['Date'],axis=1)
    
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=dfa['dt'][1:], y=dfa['Quantity'][1:], name='Actual Quantity', line=dict(color='blue')))
                        fig.add_trace(go.Scatter(x=dfp['dt'][1:], y=dfp['Quantity'][1:], name='Prediction', line=dict(color='orange')))
                        fig.add_trace(go.Scatter(x=dff['dt'], y=dff['Quantity'], name='Forecast', line=dict(color='green')))
                        st.plotly_chart(fig)
                    
                   
                    
                    
                    
                    dfobj_t=dfobj.copy()
                    dfobj_t['Model']='ARIMA'
                    dfobj1_t=dfobj1.copy()
                    dfobj1_t['Model']='SARIMA'
                    dfobj2_t=dfobj2.copy()
                    dfobj2_t['Model']='TES'
                    dfobj3_t=dfobj3.copy()
                    dfobj3_t['Model']='SARIMAX'
                    dfobj4_t=dfobj4.copy()
                    dfobj4_t['Model']='ARIMAX'
    
                    
                    ff=pd.concat([dfobj_t,dfobj1_t])
                    ff=pd.concat([ff,dfobj2_t])
                    ff=pd.concat([ff,dfobj3_t])
                    ff=pd.concat([ff,dfobj4_t])
                   
                    g = ff.sort_values(by='mape').reset_index(drop=True)[:3]
                    st.subheader('Top Models')
                    st.dataframe(g)
                    
                    st.subheader('SARIMAX Results:')
                    st.table(sarimax_results)
                    
                    writer = pd.ExcelWriter('data.xlsx', engine='openpyxl')
                    fore_df.to_excel(writer, sheet_name='Sheet1', index=False)
                    writer.close()
                    with open('data.xlsx', 'rb') as f:
                        excel_data = f.read()
                        st.download_button(label='Click here to download', data=excel_data, file_name='data.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
                        st.stop()
                    
                else:
                    
                    date=st.selectbox('**Select the date column**',['Select an option']+list(df.columns))
                    target=st.selectbox('**Select the target column**',['Select an option']+list(df.columns))
                    
                    fore_period=st.selectbox("Select the number of forecasting days:", ['',5,10])
                    fore_period=int(fore_period)
                   
                    try:
                        df.rename(columns={date:'dt',target:'Quantity'},inplace=True)
                        df['dt']=pd.to_datetime(df['dt'])
                        df = df.set_index('dt')
                    except:
                        pass
                   # df = pd.DataFrame()
                    #df['dt'] = dt
                    #df['Quantity'] = df1['Quantity']
            
                 
            
                    p = range(1,5)
                    q = range(1,5)
                    result = adfuller(df['Quantity'])
                    if(result[1]<0.05):
                        d = range(0,1)
                    elif(result[1]>0.05):
                        d = range(1,2)
            
                    pdq = list(itertools.product(p, d, q))
                    
                    z = (7,1)
            
                    a = [[(x[0],x[1],x[2],m)             for m in z         if(m>x[0]  and m>x[2]) ]       for x in list(pdq)]
            
                    model_pdq = [item for sublist in a for item in sublist]
            
                    dfobj = pd.DataFrame(columns=['param', 'mape'])
                    dfobj1 = pd.DataFrame(columns=['param','seasonal', 'mape'])
                    dfobj2 = pd.DataFrame(columns=['sp','mape'])
            
            
                    #ARIMA
            
                    for param in pdq:
                       try:
                           arima_mod = ARIMA(df['Quantity'],order=param)
                           best_model =arima_mod.fit()  
                           if(d==range(0,1)):
                               pred = best_model.fittedvalues
                               res=mape(df['Quantity'],pred)
                           elif(d==range(1,2)):
                               pred = best_model.fittedvalues
                               #pred1 = pred.shift(periods=-1)
                               res=mape(df['Quantity'][1:],pred[1:])
                           #dfobj=dfobj.append({'param':param,'mape':res},ignore_index=True)
                           dfobj=pd.concat([dfobj,pd.DataFrame({'param':[param],'mape':[res]})]).reset_index(drop=True)
                       except:
                           continue
                    #SARIMA
            
                    for param in pdq:
                        for param_seasonal in model_pdq:
                            try:
                                mod = sm.tsa.statespace.SARIMAX(df['Quantity'],
                                                                    order=param,
                                                                    seasonal_order=param_seasonal,
                                                                    enforce_stationarity=False,
                                                                    enforce_invertibility=False)
                                best_model =mod.fit()   
                                
                                pred = best_model.fittedvalues
                                #pred1 = pred.shift(periods=-1)
                                res=mape(df['Quantity'][1:],pred[1:])
                                #dfobj1 = dfobj1.append({'param':param,'seasonal':param_seasonal ,'mape': res}, ignore_index=True)
                                dfobj1=pd.concat([dfobj1,pd.DataFrame({'param':[param],'seasonal':[param_seasonal] ,'mape': [res]})]).reset_index(drop=True)
                            except:
                                continue
            
                    #TES
            
                    for sp in range(2,11,2):
                        tes = ExponentialSmoothing(df['Quantity'],trend='additive',seasonal='additive',seasonal_periods=sp,initialization_method='estimated')
                        tes_model = tes.fit()
                        pred = tes_model.fittedvalues
                        res=mape(df['Quantity'],pred)
                        #dfobj2 = dfobj2.append({'sp':sp,'mape': res}, ignore_index=True)
                        dfobj2=pd.concat([dfobj2,pd.DataFrame({'sp':[sp],'mape':[res]})]).reset_index(drop=True)
      
        
                
                
                arima_mape = dfobj.sort_values(by=['mape'])[:1]['mape'].to_list()[0]
                arima_order = dfobj.sort_values(by=['mape'])[:1]['param'].to_list()[0]
        
                sarima_mape = dfobj1.sort_values(by=['mape'])[:1]['mape'].to_list()[0]
                sarima_order = dfobj1.sort_values(by=['mape'])[:1]['param'].to_list()[0]
                sarima_seasonal_order = dfobj1.sort_values(by=['mape'])[:1]['seasonal'].to_list()[0]
        
                tes_mape = dfobj2.sort_values(by=['mape'])[:1]['mape'].to_list()[0]
                tes_sp = dfobj2.sort_values(by=['mape'])[:1]['sp'].to_list()[0]
        
                variables = {'x': arima_mape, 'y': sarima_mape, 'z': tes_mape}
                largest_variable = min(variables, key=variables.get)
                
               
                
                
                
        
        
                if(largest_variable=='x'):
                    arima = ARIMA(df['Quantity'],order=arima_order)
                    best_model=arima.fit()
                    st.write('Arima_MAPE: ',round(arima_mape,2))
        
                    if(d==range(0,1)):
                       pred = best_model.fittedvalues
                       res=mape(df['Quantity'],pred)
                       
        
                       dfa=df.reset_index()
                    #dfa.rename(columns={'dt':'Date','Quantity':'Actual Production'},inplace=True)
        
                       dfp=pd.DataFrame(pred)
                       dfp=dfp.reset_index()
                       dfp.rename(columns={0:'Quantity'},inplace=True)
        
                       fore=best_model.predict(start=(pd.Series(df.index[-1])[0])+pd.DateOffset(days=1),end=(pd.Series(df.index[-1])[0])+ pd.DateOffset(days=fore_period))
                       fore_df=pd.DataFrame(data=fore.reset_index())
                       fore_df=fore_df.rename(columns={'index':'dt','predicted_mean':'Quantity'})
                       a=pd.DataFrame(pred.reset_index().iloc[-1]).T.rename(columns={0:'Quantity'}).reset_index(drop=True)
                       dff=pd.concat([a,fore_df],ignore_index=True)
                       dff['dt']=pd.to_datetime(dff['dt'])
                    #dff=dff.set_index(dff['Date'])
                    #dff=dff.drop(['Date'],axis=1)
        
                       fig = go.Figure()
                       fig.add_trace(go.Scatter(x=dfa['dt'], y=dfa['Quantity'], name='Actual Quantity', line=dict(color='blue')))
                       fig.add_trace(go.Scatter(x=dfp['dt'], y=dfp['Quantity'], name='Prediction', line=dict(color='orange')))
                       fig.add_trace(go.Scatter(x=dff['dt'], y=dff['Quantity'], name='Forecast', line=dict(color='green')))
                       st.plotly_chart(fig)
                       
                    elif(d==range(1,2)):
                       pred = best_model.fittedvalues
                       #pred = pred.shift(periods=-1)
                       res=mape(df['Quantity'][1:],pred[1:])
                       
        
                       dfa=df.reset_index()
                    #dfa.rename(columns={'dt':'Date','Quantity':'Actual Production'},inplace=True)
        
                       dfp=pd.DataFrame(pred)
                       dfp=dfp.reset_index()
                       dfp.rename(columns={0:'Quantity'},inplace=True)
        
                       fore=best_model.predict(start=(pd.Series(df.index[-1])[0])+ pd.DateOffset(days=1),end=(pd.Series(df.index[-1])[0])+ pd.DateOffset(days=fore_period))
                       fore_df=pd.DataFrame(data=fore.reset_index())
                       fore_df=fore_df.rename(columns={'index':'dt','predicted_mean':'Quantity'})
                       a=pd.DataFrame(pred.reset_index().iloc[-1]).T.rename(columns={0:'Quantity'}).reset_index(drop=True)
                       dff=pd.concat([a,fore_df],ignore_index=True)
                       dff['dt']=pd.to_datetime(dff['dt'])
                    #dff=dff.set_index(dff['Date'])
                    #dff=dff.drop(['Date'],axis=1)
        
                       fig = go.Figure()
                       fig.add_trace(go.Scatter(x=dfa['dt'][1:], y=dfa['Quantity'][1:], name='Actual Quantity', line=dict(color='blue')))
                       fig.add_trace(go.Scatter(x=dfp['dt'][1:], y=dfp['Quantity'][1:], name='Prediction', line=dict(color='orange')))
                       fig.add_trace(go.Scatter(x=dff['dt'], y=dff['Quantity'], name='Forecast', line=dict(color='green')))
                       st.plotly_chart(fig) 
        
                if(largest_variable=='y'):
                   mod = sm.tsa.statespace.SARIMAX(df['Quantity'],order=sarima_order,seasonal_order=sarima_seasonal_order,
                                                   enforce_stationarity=False,enforce_invertibility=False)
                   best_model =mod.fit()
                   st.write('Sarima_MAPE: ',round(sarima_mape,2))
       
                   
                   pred = best_model.fittedvalues
                   #pred = pred.shift(periods=-1)
                   res=mape(df['Quantity'][1:],pred[1:])
                  

                   dfa=df.reset_index()
               #dfa.rename(columns={'dt':'Date','Quantity':'Actual Production'},inplace=True)

                   dfp=pd.DataFrame(pred)
                   dfp=dfp.reset_index()
                   dfp.rename(columns={0:'Quantity'},inplace=True)

                   fore=best_model.predict(start=(pd.Series(df.index[-1])[0])+ pd.DateOffset(days=1),end=(pd.Series(df.index[-1])[0])+ pd.DateOffset(days=fore_period))
                   fore_df=pd.DataFrame(data=fore.reset_index())
                   fore_df=fore_df.rename(columns={'index':'dt','predicted_mean':'Quantity'})
                   a=pd.DataFrame(pred.reset_index().iloc[-1]).T.rename(columns={0:'Quantity'}).reset_index(drop=True)
                   dff=pd.concat([a,fore_df],ignore_index=True)
                   dff['dt']=pd.to_datetime(dff['dt'])
               #dff=dff.set_index(dff['Date'])
               #dff=dff.drop(['Date'],axis=1)

                   fig = go.Figure()
                   fig.add_trace(go.Scatter(x=dfa['dt'][1:], y=dfa['Quantity'][1:], name='Actual Quantity', line=dict(color='blue')))
                   fig.add_trace(go.Scatter(x=dfp['dt'][1:], y=dfp['Quantity'][1:], name='Prediction', line=dict(color='orange')))
                   fig.add_trace(go.Scatter(x=dff['dt'], y=dff['Quantity'], name='Forecast', line=dict(color='green')))
                   st.plotly_chart(fig) 
        
        
                if(largest_variable=='z'):
                    tes = ExponentialSmoothing(df['Quantity'],trend='additive',seasonal='additive',seasonal_periods=tes_sp,
                                          initialization_method='estimated')
                    best_model =tes.fit()  
                    st.write('TES_MAPE: ',round(tes_mape,2))
                    pred = best_model.fittedvalues
                    #pred = pred.shift(periods=-1)
        
                    dfa=df.reset_index()
                    dfa.rename(columns={'dt':'Date','Quantity':'Actual Production'},inplace=True)
        
                    dfp=pd.DataFrame(pred)
                    dfp=dfp.reset_index()
                    dfp.rename(columns={'dt':'Date',0:'Predicted Production'},inplace=True)
        
                    fore=best_model.predict(start=(pd.Series(df.index[-1])[0])+ pd.DateOffset(days=1),end=(pd.Series(df.index[-1])[0])+ pd.DateOffset(days=fore_period))
                    fore_df=pd.DataFrame(data=fore.reset_index())
                    fore_df=fore_df.rename(columns={'index':'Date',0:'Forecasted Production'})
                    a=pd.DataFrame(pred.reset_index().iloc[-1]).T.rename(columns={'dt':'Date',0:'Forecasted Production'}).reset_index(drop=True)
                    dff=pd.concat([a,fore_df],ignore_index=True)
                    dff['Date']=pd.to_datetime(dff['Date'])
                    #dff=dff.set_index(dff['Date'])
                    #dff=dff.drop(['Date'],axis=1)
        
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=dfa['Date'], y=dfa['Actual Production'], name='Actual Production', line=dict(color='blue')))
                    fig.add_trace(go.Scatter(x=dfp['Date'], y=dfp['Predicted Production'], name='Predicted Production', line=dict(color='orange')))
                    fig.add_trace(go.Scatter(x=dff['Date'], y=dff['Forecasted Production'], name='Forecasted Production', line=dict(color='green')))
                    st.plotly_chart(fig)
                    
                dfobj_t=dfobj.copy()
                dfobj_t['Model']='ARIMA'
                dfobj1_t=dfobj1.copy()
                dfobj1_t['Model']='SARIMA'
                dfobj2_t=dfobj2.copy()
                dfobj2_t['Model']='TES'
                
                ff=pd.concat([dfobj_t,dfobj1_t])
                ff=pd.concat([ff,dfobj2_t])
                g = ff.sort_values(by='mape').reset_index(drop=True)[:3]
                st.subheader('Top Models')
                st.table(g) 
                
                writer = pd.ExcelWriter('data.xlsx', engine='openpyxl')
                fore_df.to_excel(writer, sheet_name='Sheet1', index=False)
                writer.close()
                with open('data.xlsx', 'rb') as f:
                    excel_data = f.read()
                    st.download_button(label='Click here to download', data=excel_data, file_name='data.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
                    st.stop()
                
                
    except:
        st.write(" Select the above options to continue")             
