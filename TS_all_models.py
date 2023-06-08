# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 12:00:47 2023

@author: Akshays
"""

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
        
        
rad = st.sidebar.radio('**Navigation**',['Select an option','Weekly','Monthly'])

if rad=='Select an option':
    pass

if rad=='Weekly':

    uploaded_files = st.file_uploader("Upload Files", accept_multiple_files=True)
    
    filelist = []
    
    for file in uploaded_files:
    
        filelist.append(file.name)
    
    selected_file = st.selectbox("Select a file:", filelist)
    
    
    for i in uploaded_files:
        if(i.name==selected_file):
            df1 = pd.read_excel(i)
            fw = df1['FY_Week'] + '-1'
            dt = pd.to_datetime(fw, format='%Y-%W-%w', errors='coerce')
    
            df = pd.DataFrame()
            df['dt'] = dt
            df['Quantity'] = df1['Quantity']
    
            df = df.set_index('dt')
            
            #####AR Model######

            p = range(0,5)
            d= range(0,1)
            q = range(0,1)
            
            dfobj_ar = pd.DataFrame(columns=['param', 'mape'])
            
            pdq_ar = list(itertools.product(p,d,q))
            
            for param in pdq_ar:
                try:
                    arima_mod = ARIMA(df['Quantity'],order=param)
                    best_model =arima_mod.fit()  
                    pred = best_model.fittedvalues
                    res=mape(df['Quantity'],pred)
            #         print('ARIMA{} - mape:{}'.format(param,res))
                    dfobj_ar=pd.concat([dfobj_ar,pd.DataFrame({'param':[param],'mape':[res]})]).reset_index(drop=True)
                except:
                    continue
            
            ######MA Model######

            p = range(0,1)
            d= range(0,1)
            q = range(0,5)
            
            dfobj_ma = pd.DataFrame(columns=['param', 'mape'])
            
            pdq_ma = list(itertools.product(p,d,q))
            
            for param in pdq_ma:
                try:
                    arima_mod = ARIMA(df['Quantity'],order=param)
                    best_model =arima_mod.fit()  
                    pred = best_model.fittedvalues
                    res=mape(df['Quantity'],pred)
            #         print('ARIMA{} - mape:{}'.format(param,res))
                    dfobj_ma=pd.concat([dfobj_ma,pd.DataFrame({'param':[param],'mape':[res]})]).reset_index(drop=True)
                except:
                    continue
            
            ###########ARMA Model###########

            p = range(0,5)
            d= range(0,1)
            q = range(0,5)
            
            dfobj_arma = pd.DataFrame(columns=['param', 'mape'])
            
            pdq_arma = list(itertools.product(p,d,q))
            
            for param in pdq_arma:
                try:
                    arima_mod = ARIMA(df['Quantity'],order=param)
                    best_model =arima_mod.fit()  
                    pred = best_model.fittedvalues
                    res=mape(df['Quantity'],pred)
            #         print('ARIMA{} - mape:{}'.format(param,res))
                    dfobj_arma=pd.concat([dfobj_arma,pd.DataFrame({'param':[param],'mape':[res]})]).reset_index(drop=True)
                except:
                    continue
            
            
                
            ####ARIMA & SARIMA Order#########
    
            p = range(0,5)
            q = range(0,5)
            result = adfuller(df)
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
    
    
            ##########ARIMA Model############
    
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
                
                
            #############SARIMA Model##################
    
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
    
            ###########TES Model##############
    
            for sp in range(2,11,2):
                tes = ExponentialSmoothing(df['Quantity'],trend='additive',seasonal='additive',seasonal_periods=sp,initialization_method='estimated')
                tes_model = tes.fit()
                pred = tes_model.fittedvalues
                res=mape(df['Quantity'],pred)
                #dfobj2 = dfobj2.append({'sp':sp,'mape': res}, ignore_index=True)
                dfobj2=pd.concat([dfobj2,pd.DataFrame({'sp':[sp],'mape':[res]})]).reset_index(drop=True)
    
    
            ar_mape = dfobj_ar.sort_values(by=['mape'])[:1]['mape'].to_list()[0]
            ar_order = dfobj_ar.sort_values(by=['mape'])[:1]['param'].to_list()[0]
            
            ma_mape = dfobj_ma.sort_values(by=['mape'])[:1]['mape'].to_list()[0]
            ma_order = dfobj_ma.sort_values(by=['mape'])[:1]['param'].to_list()[0]
            
            arma_mape = dfobj_arma.sort_values(by=['mape'])[:1]['mape'].to_list()[0]
            arma_order = dfobj_arma.sort_values(by=['mape'])[:1]['param'].to_list()[0]
    
            arima_mape = dfobj.sort_values(by=['mape'])[:1]['mape'].to_list()[0]
            arima_order = dfobj.sort_values(by=['mape'])[:1]['param'].to_list()[0]
    
            sarima_mape = dfobj1.sort_values(by=['mape'])[:1]['mape'].to_list()[0]
            sarima_order = dfobj1.sort_values(by=['mape'])[:1]['param'].to_list()[0]
            sarima_seasonal_order = dfobj1.sort_values(by=['mape'])[:1]['seasonal'].to_list()[0]
    
            tes_mape = dfobj2.sort_values(by=['mape'])[:1]['mape'].to_list()[0]
            tes_sp = dfobj2.sort_values(by=['mape'])[:1]['sp'].to_list()[0]
    
            variables = {'x': arima_mape, 'y': sarima_mape, 'z': tes_mape,'t': ar_mape , 'u': ma_mape   , 'v': arma_mape}
            largest_variable = min(variables, key=variables.get)
            
            
            if(largest_variable=='t'):
                arima = ARIMA(df['Quantity'],order=ar_order)
                best_model=arima.fit()
                st.write('Ar_MAPE: ',ar_mape)
                
                pred = best_model.fittedvalues
                res=mape(df['Quantity'],pred)
               

                dfa=df.reset_index()
            #dfa.rename(columns={'dt':'Date','Quantity':'Actual Production'},inplace=True)

                dfp=pd.DataFrame(pred)
                dfp=dfp.reset_index()
                dfp.rename(columns={0:'Quantity'},inplace=True)

                fore=best_model.predict(start=(pd.Series(df.index[-1])[0])+ pd.DateOffset(weeks=1),end=(pd.Series(df.index[-1])[0])+ pd.DateOffset(weeks=4))
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
            
            if(largest_variable=='u'):
                arima = ARIMA(df['Quantity'],order=ma_order)
                best_model=arima.fit()
                st.write('MA_MAPE: ',ma_mape)
                
                pred = best_model.fittedvalues
                res=mape(df['Quantity'],pred)
               

                dfa=df.reset_index()
            #dfa.rename(columns={'dt':'Date','Quantity':'Actual Production'},inplace=True)

                dfp=pd.DataFrame(pred)
                dfp=dfp.reset_index()
                dfp.rename(columns={0:'Quantity'},inplace=True)

                fore=best_model.predict(start=(pd.Series(df.index[-1])[0])+ pd.DateOffset(weeks=1),end=(pd.Series(df.index[-1])[0])+ pd.DateOffset(weeks=4))
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
                
            if(largest_variable=='v'):
                arima = ARIMA(df['Quantity'],order=arma_order)
                best_model=arima.fit()
                st.write('ARMA_MAPE: ',arma_mape)
                
                pred = best_model.fittedvalues
                res=mape(df['Quantity'],pred)
               

                dfa=df.reset_index()
            #dfa.rename(columns={'dt':'Date','Quantity':'Actual Production'},inplace=True)

                dfp=pd.DataFrame(pred)
                dfp=dfp.reset_index()
                dfp.rename(columns={0:'Quantity'},inplace=True)

                fore=best_model.predict(start=(pd.Series(df.index[-1])[0])+ pd.DateOffset(weeks=1),end=(pd.Series(df.index[-1])[0])+ pd.DateOffset(weeks=4))
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
            
    
    
            if(largest_variable=='x'):
                arima = ARIMA(df['Quantity'],order=arima_order)
                best_model=arima.fit()
                st.write('Arima_MAPE: ',arima_mape)
                if(d==range(0,1)):
                   pred = best_model.fittedvalues
                   res=mape(df['Quantity'],pred)
                   
    
                   dfa=df.reset_index()
                #dfa.rename(columns={'dt':'Date','Quantity':'Actual Production'},inplace=True)
    
                   dfp=pd.DataFrame(pred)
                   dfp=dfp.reset_index()
                   dfp.rename(columns={0:'Quantity'},inplace=True)
    
                   fore=best_model.predict(start=(pd.Series(df.index[-1])[0])+ pd.DateOffset(weeks=1),end=(pd.Series(df.index[-1])[0])+ pd.DateOffset(weeks=4))
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
    
                   fore=best_model.predict(start=(pd.Series(df.index[-1])[0])+ pd.DateOffset(weeks=1),end=(pd.Series(df.index[-1])[0])+ pd.DateOffset(weeks=4))
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
                st.write('Sarima_MAPE: ',sarima_mape)
    
                
                pred = best_model.fittedvalues
                #pred = pred.shift(periods=-1)
                res=mape(df['Quantity'][1:],pred[1:])
               

                dfa=df.reset_index()
            #dfa.rename(columns={'dt':'Date','Quantity':'Actual Production'},inplace=True)

                dfp=pd.DataFrame(pred)
                dfp=dfp.reset_index()
                dfp.rename(columns={0:'Quantity'},inplace=True)

                fore=best_model.predict(start=(pd.Series(df.index[-1])[0])+ pd.DateOffset(weeks=1),end=(pd.Series(df.index[-1])[0])+ pd.DateOffset(weeks=4))
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
                st.write('TES_MAPE: ',tes_mape)
                pred = best_model.fittedvalues
                #pred = pred.shift(periods=-1)
    
                dfa=df.reset_index()
                dfa.rename(columns={'dt':'Date','Quantity':'Actual Production'},inplace=True)
    
                dfp=pd.DataFrame(pred)
                dfp=dfp.reset_index()
                dfp.rename(columns={'dt':'Date',0:'Predicted Production'},inplace=True)
    
                fore=best_model.predict(start=(pd.Series(df.index[-1])[0])+ pd.DateOffset(weeks=1),end=(pd.Series(df.index[-1])[0])+ pd.DateOffset(weeks=4))
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
            dfobj3_t=dfobj_ar.copy()
            dfobj3_t['Model']='AR'
            dfobj4_t=dfobj_ma.copy()
            dfobj4_t['Model']='MA'
            dfobj5_t=dfobj_arma.copy()
            dfobj5_t['Model']='ARMA'
            
            ff=pd.concat([dfobj_t,dfobj1_t])
            ff=pd.concat([ff,dfobj2_t])
            ff=pd.concat([ff,dfobj3_t])
            ff=pd.concat([ff,dfobj4_t])
            ff=pd.concat([ff,dfobj5_t])
            g = ff.sort_values(by='mape').reset_index(drop=True)[:10]
            st.dataframe(g)

if rad=='Monthly':

    uploaded_files = st.file_uploader("Upload Files", accept_multiple_files=True)
    
    filelist = []
    
    for file in uploaded_files:
    
        filelist.append(file.name)
    
    selected_file = st.selectbox("Select a file:", filelist)
    
    
    for i in uploaded_files:
        if(i.name==selected_file):
            df = pd.read_excel(i)
            
            df.rename(columns={'FY_Week':'dt'},inplace=True)
            df['dt']=pd.to_datetime(df['dt'])
            df = df.set_index('dt')
           # df = pd.DataFrame()
            #df['dt'] = dt
            #df['Quantity'] = df1['Quantity']
            
            #####AR Model######

            p = range(0,5)
            d= range(0,1)
            q = range(0,1)
            
            dfobj_ar = pd.DataFrame(columns=['param', 'mape'])
            
            pdq_ar = list(itertools.product(p,d,q))
            
            for param in pdq_ar:
                try:
                    arima_mod = ARIMA(df['Quantity'],order=param)
                    best_model =arima_mod.fit()  
                    pred = best_model.fittedvalues
                    res=mape(df['Quantity'],pred)
            #         print('ARIMA{} - mape:{}'.format(param,res))
                    dfobj_ar=pd.concat([dfobj_ar,pd.DataFrame({'param':[param],'mape':[res]})]).reset_index(drop=True)
                except:
                    continue
            
            ######MA Model######

            p = range(0,1)
            d= range(0,1)
            q = range(0,5)
            
            dfobj_ma = pd.DataFrame(columns=['param', 'mape'])
            
            pdq_ma = list(itertools.product(p,d,q))
            
            for param in pdq_ma:
                try:
                    arima_mod = ARIMA(df['Quantity'],order=param)
                    best_model =arima_mod.fit()  
                    pred = best_model.fittedvalues
                    res=mape(df['Quantity'],pred)
            #         print('ARIMA{} - mape:{}'.format(param,res))
                    dfobj_ma=pd.concat([dfobj_ma,pd.DataFrame({'param':[param],'mape':[res]})]).reset_index(drop=True)
                except:
                    continue
            
            ###########ARMA Model###########

            p = range(0,5)
            d= range(0,1)
            q = range(0,5)
            
            dfobj_arma = pd.DataFrame(columns=['param', 'mape'])
            
            pdq_arma = list(itertools.product(p,d,q))
            
            for param in pdq_arma:
                try:
                    arima_mod = ARIMA(df['Quantity'],order=param)
                    best_model =arima_mod.fit()  
                    pred = best_model.fittedvalues
                    res=mape(df['Quantity'],pred)
            #         print('ARIMA{} - mape:{}'.format(param,res))
                    dfobj_arma=pd.concat([dfobj_arma,pd.DataFrame({'param':[param],'mape':[res]})]).reset_index(drop=True)
                except:
                    continue
            
            
            
            ####ARIMA & SARIMA Order#########
         
    
            p = range(0,5)
            q = range(0,5)
            result = adfuller(df)
            if(result[1]<0.05):
                d = range(0,1)
            elif(result[1]>0.05):
                d = range(1,2)
    
            pdq = list(itertools.product(p, d, q))
            
            z = (4,6)
    
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
    
    
            ar_mape = dfobj_ar.sort_values(by=['mape'])[:1]['mape'].to_list()[0]
            ar_order = dfobj_ar.sort_values(by=['mape'])[:1]['param'].to_list()[0]
            
            ma_mape = dfobj_ma.sort_values(by=['mape'])[:1]['mape'].to_list()[0]
            ma_order = dfobj_ma.sort_values(by=['mape'])[:1]['param'].to_list()[0]
            
            arma_mape = dfobj_arma.sort_values(by=['mape'])[:1]['mape'].to_list()[0]
            arma_order = dfobj_arma.sort_values(by=['mape'])[:1]['param'].to_list()[0]
            
            arima_mape = dfobj.sort_values(by=['mape'])[:1]['mape'].to_list()[0]
            arima_order = dfobj.sort_values(by=['mape'])[:1]['param'].to_list()[0]
    
            sarima_mape = dfobj1.sort_values(by=['mape'])[:1]['mape'].to_list()[0]
            sarima_order = dfobj1.sort_values(by=['mape'])[:1]['param'].to_list()[0]
            sarima_seasonal_order = dfobj1.sort_values(by=['mape'])[:1]['seasonal'].to_list()[0]
    
            tes_mape = dfobj2.sort_values(by=['mape'])[:1]['mape'].to_list()[0]
            tes_sp = dfobj2.sort_values(by=['mape'])[:1]['sp'].to_list()[0]
    
            variables = {'x': arima_mape, 'y': sarima_mape, 'z': tes_mape,'t': ar_mape , 'u': ma_mape   , 'v': arma_mape}
            largest_variable = min(variables, key=variables.get)
            
            
            if(largest_variable=='t'):
                arima = ARIMA(df['Quantity'],order=ar_order)
                best_model=arima.fit()
                st.write('Ar_MAPE: ',ar_mape)
                
                pred = best_model.fittedvalues
                res=mape(df['Quantity'],pred)
               

                dfa=df.reset_index()
            #dfa.rename(columns={'dt':'Date','Quantity':'Actual Production'},inplace=True)

                dfp=pd.DataFrame(pred)
                dfp=dfp.reset_index()
                dfp.rename(columns={0:'Quantity'},inplace=True)

                fore=best_model.predict(start=(pd.Series(df.index[-1])[0])+ pd.DateOffset(months=1),end=(pd.Series(df.index[-1])[0])+ pd.DateOffset(months=4))
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
            
            if(largest_variable=='u'):
                arima = ARIMA(df['Quantity'],order=ma_order)
                best_model=arima.fit()
                st.write('MA_MAPE: ',ma_mape)
                
                pred = best_model.fittedvalues
                res=mape(df['Quantity'],pred)
               

                dfa=df.reset_index()
            #dfa.rename(columns={'dt':'Date','Quantity':'Actual Production'},inplace=True)

                dfp=pd.DataFrame(pred)
                dfp=dfp.reset_index()
                dfp.rename(columns={0:'Quantity'},inplace=True)

                fore=best_model.predict(start=(pd.Series(df.index[-1])[0])+ pd.DateOffset(months=1),end=(pd.Series(df.index[-1])[0])+ pd.DateOffset(months=4))
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
                
            if(largest_variable=='v'):
                arima = ARIMA(df['Quantity'],order=arma_order)
                best_model=arima.fit()
                st.write('ARMA_MAPE: ',arma_mape)
                
                pred = best_model.fittedvalues
                res=mape(df['Quantity'],pred)
               

                dfa=df.reset_index()
            #dfa.rename(columns={'dt':'Date','Quantity':'Actual Production'},inplace=True)

                dfp=pd.DataFrame(pred)
                dfp=dfp.reset_index()
                dfp.rename(columns={0:'Quantity'},inplace=True)

                fore=best_model.predict(start=(pd.Series(df.index[-1])[0])+ pd.DateOffset(months=1),end=(pd.Series(df.index[-1])[0])+ pd.DateOffset(months=4))
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
            
            
            
    
    
            if(largest_variable=='x'):
                arima = ARIMA(df['Quantity'],order=arima_order)
                best_model=arima.fit()
                st.write('Arima_MAPE: ',arima_mape)
    
                if(d==range(0,1)):
                   pred = best_model.fittedvalues
                   res=mape(df['Quantity'],pred)
                   
    
                   dfa=df.reset_index()
                #dfa.rename(columns={'dt':'Date','Quantity':'Actual Production'},inplace=True)
    
                   dfp=pd.DataFrame(pred)
                   dfp=dfp.reset_index()
                   dfp.rename(columns={0:'Quantity'},inplace=True)
    
                   fore=best_model.predict(start=(pd.Series(df.index[-1])[0])+pd.DateOffset(months=1),end=(pd.Series(df.index[-1])[0])+ pd.DateOffset(months=4))
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
    
                   fore=best_model.predict(start=(pd.Series(df.index[-1])[0])+ pd.DateOffset(months=1),end=(pd.Series(df.index[-1])[0])+ pd.DateOffset(months=4))
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
                st.write('Sarima_MAPE: ',sarima_mape)
    
                
                pred = best_model.fittedvalues
                #pred = pred.shift(periods=-1)
                res=mape(df['Quantity'][1:],pred[1:])
               

                dfa=df.reset_index()
            #dfa.rename(columns={'dt':'Date','Quantity':'Actual Production'},inplace=True)

                dfp=pd.DataFrame(pred)
                dfp=dfp.reset_index()
                dfp.rename(columns={0:'Quantity'},inplace=True)

                fore=best_model.predict(start=(pd.Series(df.index[-1])[0])+ pd.DateOffset(months=1),end=(pd.Series(df.index[-1])[0])+ pd.DateOffset(months=4))
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
                st.write('TES_MAPE: ',tes_mape)
                pred = best_model.fittedvalues
                #pred = pred.shift(periods=-1)
    
                dfa=df.reset_index()
                dfa.rename(columns={'dt':'Date','Quantity':'Actual Production'},inplace=True)
    
                dfp=pd.DataFrame(pred)
                dfp=dfp.reset_index()
                dfp.rename(columns={'dt':'Date',0:'Predicted Production'},inplace=True)
    
                fore=best_model.predict(start=(pd.Series(df.index[-1])[0])+ pd.DateOffset(months=1),end=(pd.Series(df.index[-1])[0])+ pd.DateOffset(months=4))
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
            dfobj3_t=dfobj_ar.copy()
            dfobj3_t['Model']='AR'
            dfobj4_t=dfobj_ma.copy()
            dfobj4_t['Model']='MA'
            dfobj5_t=dfobj_arma.copy()
            dfobj5_t['Model']='ARMA'
            
            
            
            ff=pd.concat([dfobj_t,dfobj1_t])
            ff=pd.concat([ff,dfobj2_t])
            ff=pd.concat([ff,dfobj3_t])
            ff=pd.concat([ff,dfobj4_t])
            ff=pd.concat([ff,dfobj5_t])
            g = ff.sort_values(by='mape').reset_index(drop=True)[:10]
            st.dataframe(g)

 

        
        
        
