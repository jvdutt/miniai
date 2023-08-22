import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import scipy
import scipy.cluster.hierarchy as hc


def set_df_display(r=1000,c=1000):
    pd.get_option("display.max_rows",r)
    pd.get_option("display.max_columns",c)

def plot_dendogram(df,figsize=[10,10],method='average',distance=lambda x:1-x): 
    
    #dropping columns with nunique value =1
    columns = df.columns[df.nunique()==1]
    df = df.drop(columns,axis=1)
    
    #calculating spearmanr make sure that corr==corr.T
    corr = np.round(scipy.stats.spearmanr(df).correlation,4)
    plt.imshow(corr,'gray')
    plt.show()
    
    #converting into distance(1-corr) and condensed form cause linkage accepts raw vectors or condensed form
    corr_dist_condensed = hc.distance.squareform(distance(corr))
    
    #plotting the dendogram method(use ?hc.linkage) 
    z = hc.linkage(corr_dist_condensed,method=method)
    
    plt.figure(figsize=figsize)
    #labels and orientation on left because of clarity for text
    dend = hc.dendrogram(z,labels=df.columns,orientation='left')
    plt.show()
    
    return corr,df 

    

    
class ProcDf():
    def __init__(self,df,train=True,cache=None):
        self.df = df
        self.train = train
        self.cache = cache
        if self.cache is None:
            self.cache={'cat':{},'nas':{}}
        
    def add_date_attributes(self,column_name):
        df = self.df
        if df[column_name].isnull().any():assert(False)
        
        ts = df[column_name].astype("datetime64[ns]")
        
        df[column_name+"_week"] = ts.dt.isocalendar().week
        for attr in ["year","month","day","day_of_year","day_of_week","is_month_end","is_month_start","is_quarter_end","is_quarter_start","is_year_end","is_year_start"]:
            df[column_name+"_"+attr]=getattr(ts.dt,attr)
        
        if self.train:
            self.cache['min_date'] = ts.min()
            
        df[column_name+"_"+"elapsed_days"] = (ts-self.cache['min_date']).dt.days
        df.drop(column_name,axis=1,inplace=True)
        
    def convert2cat(self,column_name,ordered=True,order_list=None):
        df = self.df
        if pd.api.types.is_string_dtype(df[column_name]) or pd.api.types.is_categorical_dtype(df[column_name]):

            if not self.train:
                ordered,order_list = self.cache['cat'][column_name]

            df[column_name] = df[column_name].astype("category").cat.as_ordered()
            if not ordered:
                df[column_name] = df[column_name].cat.as_unordered()
            if order_list is not None:
                df[column_name] = df[column_name].cat.set_categories(order_list,ordered=ordered)
            
            if self.train:
                self.cache['cat'][column_name] = [df[column_name].cat.ordered,df[column_name].cat.categories]

    def convert_all_cat(self):
        for c in self.df.columns:
            self.convert2cat(c)
                
    def nunique_cat(self):
        return self.df.nunique()[self.df.dtypes=='category']
    
    def max_n_cat(self,n=7):
        n_cat = self.nunique_cat()
        n_cat = n_cat[n_cat<=n]
        for column_name in n_cat.index:
            self.convert2cat(column_name,ordered=False)
            
    def random_sample(self,n,replace=False):
        return ProcDf(self.df.sample(n,replace=False,ignore_index=True),self.train,self.cache)
    
    def shuffle(self):
        self.df = self.df.sample(frac=1.0,ignore_index=True)
        
    def train_valid_split(self,n=0.8):
        if type(n)==float:
            n = int(len(self.df)*n)
        assert type(n)==int
        return ProcDf(self.df[:n].copy(),train=True,cache=self.cache),ProcDf(self.df[n:].copy(),train=False,cache=self.cache)
        
    def fix_missing_values(self):
        for n,s in self.df.items():
            if pd.api.types.is_numeric_dtype(s) and s.isnull().any():
                self.df[n+"_na"]=s.isnull()
                if self.train:
                    self.cache['nas'][n] = s.median()
                self.df[n] = s.fillna(self.cache['nas'][n])
                
    def numericalize(self):
        for n,s in self.df.items():
            if pd.api.types.is_categorical_dtype(s) and s.cat.ordered:
                self.df[n]=s.cat.codes+1
        self.df = pd.get_dummies(self.df,dummy_na=True)
