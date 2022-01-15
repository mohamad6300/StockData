# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 10:13:45 2021

@author: Mohamad
"""
import pandas as pd
import yfinance as yf
import tsemodule as tm  #---- Tehran Data Library 


class Stocks:
    def __init__(self, period, start_date, end_date,MarketIndex,Symbol ,CompleteSymbol):
        self.Period = period
        self.Start_Date = start_date
        self.End_Date = end_date
        self.MarketIndex = MarketIndex
        self.Symbol = Symbol
        self.CompleteSymbol = CompleteSymbol
    
     #---------------------------------------------------Date Correction    
    def __DateCorrection(self, Df):
    # Sample WrongDate='2019-03-22' ## Check if exist in range then delete 
        startYear=int(self.Start_Date.year) -1
        endYear= int(self.End_Date.year) + 1
        
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
    
    def get_stock_data(self):
        if self.MarketIndex == 0:
           tickerData = yf.Ticker(self.Symbol) # Get ticker data        
           tickerDf = yf.download(self.Symbol,start=self.Start_Date,end=self.End_Date,interval=self.Period )
           tickerDf= self.__DateCorrection(tickerDf) 
           
        elif self.MarketIndex==1:
             tickerData = yf.Ticker(self.Symbol) # Get ticker data
             tickerDf = yf.download(self.Symbol,start=self.Start_Date,end=self.End_Date,interval=self.Period )
             tickerDf= self.__DateCorrection(tickerDf)
        elif self.MarketIndex==2:
             tickerDf= tm.stock(self.Symbol)
             mask = (tickerDf.index>= pd.to_datetime(self.Start_Date)) & (tickerDf.index<= pd.to_datetime(self.End_Date))
             tickerDf= (tickerDf.loc[mask])
             tickerData=""
        return (tickerData, tickerDf)          
            


  

        

        