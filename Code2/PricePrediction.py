# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 18:10:39 2021

@author: Mohamad Ahmadabadi 
"""
# -------------------------------------------------------------------- import library
import time
import datetime
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import cufflinks as cf
import streamlit as st
from PIL import Image
from plotly.offline import plot
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import tsemodule as tm  #---- Tehran Data Library 

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split


global Flag
# -------------------------------------------------------------------- Set Constan
MarketList=['Crypto','Nazdaq','Tehran']
Currency='USD'

#Begin ----------------------------------------------------------------Change Page Title 
def set_page_title(title):
    st.sidebar.markdown(unsafe_allow_html=True, body=f"""
        <iframe height=0 srcdoc="<script>
            const title = window.parent.document.querySelector('title') \
                
            const oldObserver = window.parent.titleObserver
            if (oldObserver) {{
                oldObserver.disconnect()
            }} \

            const newObserver = new MutationObserver(function(mutations) {{
                const target = mutations[0].target
                if (target.text !== '{title}') {{
                    target.text = '{title}'
                }}
            }}) \

            newObserver.observe(title, {{ childList: true }})
            window.parent.titleObserver = newObserver \

            title.text = '{title}'
        </script>" />
    """)
set_page_title("Price Prediction ")
#End ----------------------------------------------------------------Change Page Title 


# Begin -----------------------------------------------------------Site Header ----------
st.write('''
# Visualizing Stock Data
**Ahmadabadi.com**
''')

url='https://raw.githubusercontent.com/mohamad6300/StockData/main/Pic/Trading.jpg'
im = Image.open(requests.get(url, stream=True).raw)
st.image(im,width=600,caption='Price Prediction')
# End -----------------------------------------------------------Site Header ----------


st.sidebar.header('INSERT DATA')

def SelectMarket ():        
    Market=st.sidebar.selectbox('Select The Market : ',MarketList)
    return Market

def GetMarketSymbol(Market):
    if Market==MarketList[0]:        
        # ticker_list = pd.read_json('https://raw.githubusercontent.com/mohamad6300/StockData/main/Crypto.json',typ='series')
        ticker_list2 = pd.read_csv('https://raw.githubusercontent.com/mohamad6300/StockData/main/Crypto.csv')
        ticker_list =ticker_list2['Symbol']
        Symbol = st.sidebar.selectbox('Stock ticker', ticker_list) # Select ticker symbol           
        cond = (ticker_list2 ['Symbol'] == Symbol) 
        CompleteSymbol= ticker_list2[cond].Name.values[0]        
        return (Symbol ,CompleteSymbol )
    
    elif Market== MarketList[1]:
        ticker_list = pd.read_csv('https://raw.githubusercontent.com/mohamad6300/StockData/main/Nsymbols.txt')
        Symbol = st.sidebar.selectbox('Stock ticker', ticker_list) # Select ticker symbol           
        CompleteSymbol = ""
        return (Symbol ,CompleteSymbol )
    
    elif Market== MarketList[2]:
        ticker_list2 = pd.read_csv('https://raw.githubusercontent.com/mohamad6300/StockData/main/namad.csv',encoding='utf-8')
        ticker_list =ticker_list2['Fnamad']
        CompleteSymbol = st.sidebar.selectbox('Stock ticker', ticker_list) # Select ticker symbol           
        cond = (ticker_list2 ['Fnamad'] == CompleteSymbol) 
        Symbol = ticker_list2[cond].Enamad.values[0]
        return (Symbol ,CompleteSymbol )
    
    else:
        return  'NONE'


#---------------------------------------------------Date Correction 

def DateCorrection(Df,start_date,end_date):
    # Sample WrongDate='2019-03-22' ## Check if exist in range then delete 
    startYear=int(start_date.year) -1
    endYear= int(end_date.year) + 1
    
    for i in range(startYear,endYear ):
       WrongDate1=str(i) + '-03-22'
       WrongDate2=str(i) + '-03-21'
       if WrongDate1  in  Df.index:
          x=(Df.loc[[WrongDate1]].index)
          Df.drop(x,inplace=True)  
       if WrongDate2  in  Df.index:
          x=(Df.loc[[WrongDate2]].index)
          Df.drop(x,inplace=True)      
    return Df

def ShowInfo(tickerData,Market):
    if Market== MarketList[1]:
        string_logo = '<img src=%s>' % tickerData.info['logo_url']
        st.markdown(string_logo, unsafe_allow_html=True)
        string_name = tickerData.info['longName']
        st.header('**%s**' % string_name)
        string_summary = tickerData.info['longBusinessSummary']
        st.info(string_summary)    

def showTickerData(tickerDf,Market):
    st.header('**Stock Data**')
    st.write(tickerDf)    
    st.header('**Summary**')
    if Market== MarketList[2]:
        del tickerDf['<DTYYYYMMDD>']
    st.write(tickerDf.describe())
    
def ShowVol(tickerDf,Market,TickerSymbol,CompleteSymbol):
    if Market== MarketList[2]:
        st.header('**' +  ' حجم معاملات ' + CompleteSymbol  + '**')
    else:
        st.header( '**' + TickerSymbol + ' Volume\n'+ '**')
    st.bar_chart(tickerDf['Volume'])
    
    
def ShowMFT(tickerDf,TickerSymbol):   
    # MFT Indicator
    st.header('**MFT Indicator**')    
    plt.style.use('fivethirtyeight')    
    
    typical_price=(tickerDf['Close']+tickerDf['High']+tickerDf['Low'])/3
    period=14
    money_flow=typical_price*tickerDf['Volume']    
    
    positive_flow=[]
    negative_flow=[]
    
    for i in range(1,len(typical_price)):
        if typical_price[i]>typical_price[i-1]:
            positive_flow.append(money_flow[i])
            negative_flow.append(0)
        elif typical_price[i]<typical_price[i-1]:
            positive_flow.append(0)
            negative_flow.append(money_flow[i])            
        else :
            positive_flow.append(0)
            negative_flow.append(0)
    
    positive_mf=[]
    negative_mf=[]
    
    for i in range(period-1,len(positive_flow)):
        positive_mf.append(sum(positive_flow[i+1-period:i+1]))
                           
    for i in range(period-1,len(negative_flow)):
        negative_mf.append(sum(negative_flow[i+1-period:i+1]))     
    
    mfi=100 * (np.array(positive_mf) / (np.array(positive_mf) + np.array(negative_mf) ) )
    
    df2=pd.DataFrame()
    df2['MFI']=mfi
    fig , (ax1 , ax2)=plt.subplots(nrows=2,ncols=1,figsize=(12.5,6))
    ax1.plot(tickerDf['Close'],label='Close Price')
    ax2.plot(df2['MFI'],label='MFI')
    ax2.axhline(20,linestyle='--',color="r",alpha=0.5)
    ax2.axhline(30,linestyle='--',color="b",alpha=0.5)
    ax2.axhline(70,linestyle='--',color="b",alpha=0.5)
    ax2.axhline(80,linestyle='--',color="r",alpha=0.5)
    ax1.set_title('MFI Visualizer')
    st.pyplot(plt)


def showBollingerBands(tickerDf):    
    # Bollinger bands
    st.header('**Bollinger Bands**')
    qf=cf.QuantFig(tickerDf,title='First Quant Figure',legend='top',name='GS')
    qf.add_bollinger_bands()
    fig = qf.iplot(asFigure=True)
    st.plotly_chart(fig)


def showCandleStickChart(tickerDf,TickerSymbol):
    #Candle Stick 
    st.header('**Candle Stick**')
    chkShowMAOnCandle= st.checkbox("Show Moving Average ")
    fig=go.Figure(
    data=[
        go.Candlestick(
        x=tickerDf.index,
            low=tickerDf['Low'],
            high=tickerDf['High'],
            close=tickerDf['Close'],
            open=tickerDf['Open'],
            increasing_line_color='green',
            decreasing_line_color='red'
        )
        
    ]
    )
    fig.update_layout(
    title=TickerSymbol,
    yaxis_title=Currency,
    xaxis_title='Date' ,
    autosize=False,
    width=900,
    height=800
    
    )
    
    if chkShowMAOnCandle:
        MAA= ShowMAOnCandle(tickerDf,TickerSymbol)
        fig.add_trace(MAA[0])
        fig.add_trace(MAA[1])
        if MAA[4]:
            fig.add_trace(MAA[2])
            fig.add_trace(MAA[3])    
    st.plotly_chart(fig)
    
    
    
    




def ShowMASignal(tickerDf,SlowColName,FastColName) : 
    signalBuy=[]
    signalSell=[]
    f=-1
    for i in range(len(tickerDf)):
        if tickerDf[FastColName][i]>tickerDf[SlowColName][i]:
            if f!=1:
                signalBuy.append(tickerDf['Price'][i])
                signalSell.append(np.nan)
                f=1
            else:
                signalBuy.append(np.nan)
                signalSell.append(np.nan)
        elif tickerDf[FastColName][i]<tickerDf[SlowColName][i]: 
            if f!=0:
                signalBuy.append(np.nan)
                signalSell.append(tickerDf['Price'][i])
                f=0
            else:
                signalBuy.append(np.nan)
                signalSell.append(np.nan)
        else:
            signalBuy.append(np.nan)
            signalSell.append(np.nan) 

    return (signalBuy , signalSell )          



def ShowMA(tickerDf,TickerSymbol):
    # Moving Average Indicator
    st.header('**Moving Average Indicator**')    
    myrange=range(1,100)
    slen=st.select_slider ("Choose your slow Moving Average Length",options=myrange,value=1)    
    flen=st.select_slider ("Choose your fast Moving Average Length",options=myrange,value=1)    
    
    chkShowMASignal= st.checkbox("Show Buy & Sell Signal ")
    
    SlowColName= 'MA'  + str(slen)
    FastColName= 'MA'  + str(flen)
    data=pd.DataFrame()
    data['Price']=tickerDf['Adj Close']
    data[FastColName]=tickerDf['Adj Close'].rolling(window=flen).mean()
    data[SlowColName]=tickerDf['Adj Close'].rolling(window=slen).mean()
    
    f, ax = plt.subplots(1,1,figsize=(16,8))
    ax.plot(data['Price'], alpha=0.7, linewidth=2, label='Price')
    ax.plot(data[FastColName], alpha=0.7, linewidth=2, label=FastColName,color='k')
    ax.plot(data[SlowColName], alpha=0.7, linewidth=2, label=SlowColName,color='m')
    if chkShowMASignal:
        buy_sell= ShowMASignal(data,SlowColName,FastColName)
        data['buy signal']=buy_sell[0]
        data['sell signal']=buy_sell[1]
        ax.scatter(data.index,data['buy signal'],label='BUY',marker='^',color='g')
        ax.scatter(data.index,data['sell signal'],label='SELL',marker='v',color='r')

    ax.set_xlabel('Date')
    ax.set_ylabel('Price(USD)')
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
  # ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
  # for spine in ('top', 'right', 'bottom', 'left'):
  #     ax.spines[spine].set_visible(False)
    plt.show()
    st.pyplot(plt)

def ShowMAOnCandle(tickerDf,TickerSymbol):   
    
    myrange=range(1,101)
    flen=st.select_slider ("Choose your slow Moving Average Length",options=myrange,value=1)    
    slen=st.select_slider ("Choose your fast Moving Average Length",options=myrange,value=1)    
    
    chkShowMASignalOnCandle= st.checkbox("Show Buy & Sell Signal ")
    
    SlowColName= 'MA'  + str(slen)
    FastColName= 'MA'  + str(flen)
    data=pd.DataFrame()
    data['Price']=tickerDf['Adj Close']
    data[FastColName]=tickerDf['Adj Close'].rolling(window=flen).mean()
    data[SlowColName]=tickerDf['Adj Close'].rolling(window=slen).mean()
    
    ma_fast = go.Scatter(x=data.index, y=data[FastColName], mode='lines', name=FastColName, marker_color='rgba(0, 0, 255, .8)')
    ma_slow = go.Scatter(x=data.index, y=data[SlowColName], mode='lines', name=SlowColName, marker_color='rgba(255, 0, 0, .8)')
        
    bs=""
    ss=""
    Signal=False
    if chkShowMASignalOnCandle:
        buy_sell= ShowMASignal(data,SlowColName,FastColName)
        data['buy signal']=buy_sell[0]
        data['sell signal']=buy_sell[1]
    
        bs = go.Scatter(x=data.index, y=data['buy signal'], mode='markers', name='Buy', marker_color='rgba(0, 255, 0, .8)', marker_symbol='star-triangle-up',marker_size=20)
        ss = go.Scatter(x=data.index, y=data['sell signal'], mode='markers', name='Sell', marker_color='rgba(255, 0, 0, .8)', marker_symbol='star-triangle-down',marker_size=20)
        Signal=True  

    return (ma_fast,ma_slow,bs,ss, Signal)
  

    
    

def SVRPrediction(tickerDf,days):
    tickerDf=tickerDf[['Close']]
    forecast=int(days)
    tickerDf['Prediction']=tickerDf[['Close']].shift(-forecast)
    x= np.array(tickerDf.drop(['Prediction'],1))
    x= x[:-forecast]
    y= np.array(tickerDf['Prediction'])
    y=y[:-forecast]
    
    xtrain , xtest , ytrain , ytest=train_test_split(x,y,test_size=0.8)
    mysvr=SVR(kernel='rbf',C=1000,gamma=0.1)
    mysvr.fit(xtrain,ytrain)
    svmconf=mysvr.score(xtest,ytest)


    x_forecast=np.array(tickerDf.drop(['Prediction'],1))[-forecast:]
    svmpred=mysvr.predict(x_forecast)
    st.header('SVM Prediction')
    
    for i in range(len(x_forecast)):
        st.success("Predicted=%s" % (svmpred[i]))
    
    
    
    # st.success(svmpred)

    st.header('SVM Accuracy')
    st.success(svmconf)
    
    
def LRegression(tickerDf,days):
    
    tickerDf=tickerDf[['Close']]
    forecast=int(days)
    tickerDf['Prediction']=tickerDf[['Close']].shift(-forecast)
    x= np.array(tickerDf.drop(['Prediction'],1))
    x= x[:-forecast]
    y= np.array(tickerDf['Prediction'])
    y=y[:-forecast]
    xtrain , xtest , ytrain , ytest=train_test_split(x,y,test_size=0.8)
    
    
    
    lr=LinearRegression()
    lr.fit(xtrain,ytrain)
    
    
    x_forecast=np.array(tickerDf.drop(['Prediction'],1))[-forecast:]        
    lrpred=lr.predict(x_forecast)
    st.header('LinearRegression Prediction')
    
    
    
    for i in range(len(x_forecast)):
        # print("X=%s, Predicted=%s" % (x_forecast[i], lrpred[i]))
        # st.success("X=%s, Predicted=%s" % (x_forecast[i], lrpred[i]))
        st.success("Predicted=%s" % (lrpred[i]))
    # st.success(lrpred)
    
    lrconf=lr.score(xtest,ytest)
    st.header('LinearRegression Accuracy')
    st.success(lrconf)
    
    
        
    
    
        
        






    




  
Market = SelectMarket()
ss=GetMarketSymbol(Market)
TickerSymbol=ss[0]
CompleteSymbol=ss[1]

start_date = st.sidebar.date_input("Start date", datetime.date(2021, 1, 1))
end_date = st.sidebar.date_input("End date", datetime.date(2021, 5, 31))
ChkInfo=  st.sidebar.checkbox('Show Info')
ChkData= st.sidebar.checkbox('Show Data')
ChkVol= st.sidebar.checkbox('Show Volume')
chkMFT= st.sidebar.checkbox("Show MFT Indicator")
ChkCandleStock= st.sidebar.checkbox('Show Candle Stick Chart')    
chkMA= st.sidebar.checkbox("Show Moving Average Indicator")
chkBB= st.sidebar.checkbox("Show Bollinger Bands Indicator")
chkPredict = st.sidebar.checkbox("Price Pridiction")


if ChkCandleStock  or ChkInfo or ChkData or chkBB or chkMFT or chkMA or chkPredict or ChkVol:    
     if Market=='Nazdaq':
        try:            
           with st.spinner('Extracting Features... '):
               time.sleep(1)        
           tickerData = yf.Ticker(TickerSymbol) # Get ticker data
        except:
             st.markdown('If you wants to make money, your **_Ticker_ symbol** should be correct!!! :p ')
             raise             
        try:
            st.sidebar.success("Now you can see " + TickerSymbol + " info")        
            tickerDf = yf.download(TickerSymbol,start=start_date,end=end_date,interval='1d' )
            tickerDf= DateCorrection(tickerDf,start_date,end_date)
        except:
             st.markdown('unexpected error')
             raise
             
     elif Market=='Tehran':      
        tickerDf= tm.stock(TickerSymbol)
        mask = (tickerDf.index>= pd.to_datetime(start_date)) & (tickerDf.index<= pd.to_datetime(end_date))
        tickerDf= (tickerDf.loc[mask])
        tickerData=""
     elif Market=='Crypto':
         
        try:            
          # with st.spinner('Extracting Features... '):
          #     time.sleep(1)                   
           tickerData = yf.Ticker(TickerSymbol) # Get ticker data
        except:
             st.markdown('If you wants to make money, your **_Ticker_ symbol** should be correct!!! :p ')
             st.markdown('unexpected error')
             raise             
        try:            
            st.sidebar.success("Now you can see " + TickerSymbol + " info")        
            
            
            #tickerDf  = tickerData.history(period="max")
            #tickerDf = yf.download(TickerSymbol,start=start_date,end=end_date,interval='1d',period="max" )
          #  tickerDf = yf.download(TickerSymbol,start='2022-01-01',end='2022-01-11',interval='1d',period="max" )
            tickerDf =yf.download('APPL')
            set_page_title("asghar")
            
            #tickerDf= DateCorrection(tickerDf,start_date,end_date)
        except:
             st.markdown('unexpected error')
             raise
         
        
         

            
     if ChkInfo: 
        ShowInfo(tickerData,Market)
     if ChkData: 
        showTickerData(tickerDf ,Market)        
     if ChkVol:
         ShowVol(tickerDf,Market,TickerSymbol,CompleteSymbol)
     if ChkCandleStock:
        showCandleStickChart(tickerDf,TickerSymbol)
     if chkBB:
        showBollingerBands(tickerDf)
     if chkMA:
         ShowMA(tickerDf,TickerSymbol)
     if chkMFT:
         ShowMFT(tickerDf,TickerSymbol)
     if chkPredict:
         days=st.sidebar.text_input('How many days you wanna predict? ',5)
         btnLR = st.sidebar.button("Linear Regrisson Prediction")        
         btnSVR = st.sidebar.button("SVR Prediction")                 
         if btnSVR:
            SVRPrediction(tickerDf,days)
         if btnLR :
            LRegression(tickerDf,days)
           
         


# streamlit run "D:\University Shamsipour 1398\992-AI\Final Project\PricePrediction\PricePrediction.py"