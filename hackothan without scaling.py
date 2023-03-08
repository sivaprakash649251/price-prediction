#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df1=pd.read_csv('Training_Data_Set.csv')


# In[3]:


df1.head(1)


# In[4]:


num=['Id','Distance ', 'manufacture_year', 'Age of car', 'engine_displacement', 'engine_power', 'Vroom Audit Rating', 'door_count', 'seat_count','Price']
cat=['Maker','model','Location','Owner Type','body_type','transmission','fuel_type']


# In[5]:


df1['Distance '].fillna(df1['Distance '].mode()[0],inplace=True)


# In[6]:


df1['engine_power'].fillna(df1['engine_power'].mode()[0],inplace=True)
df1['body_type'].fillna(df1['body_type'].mode()[0],inplace=True)
df1['Owner Type'] = df1['Owner Type'].str.replace('&', '')


# In[7]:


from sklearn.preprocessing import MinMaxScaler,OneHotEncoder


# In[8]:


df1_num=df1[num]


# In[9]:


ohe=OneHotEncoder(sparse=False)


# In[10]:


df1_ohe=pd.DataFrame(ohe.fit_transform(df1[['Maker', 'model', 'Location', 'Owner Type', 'body_type', 'transmission', 'fuel_type']]))


# In[11]:


df1_ohe.columns=ohe.get_feature_names_out()


# In[ ]:





# In[12]:


df1_ohe.index=df1.index


# In[ ]:





# In[13]:


df1_final=pd.concat([df1_num,df1_ohe],axis=1)


# In[ ]:





# In[14]:


df1_final


# In[15]:


df1_final.replace('None',1,inplace=True)


# In[16]:


df1_final.columns


# In[17]:


df1_final.rename(columns={'Owner Type_Fourth  Above': 'Owner Type_Fourth Above'}, inplace=True)


# In[53]:


x=df1_final[['Id', 'Distance ', 'manufacture_year', 'Age of car',
       'engine_displacement', 'engine_power', 'Vroom Audit Rating',
       'door_count', 'seat_count', 'Maker_audi', 'Maker_bmw',
       'Maker_fiat', 'Maker_hyundai', 'Maker_maserati', 'Maker_nissan',
       'Maker_skoda', 'Maker_toyota', 'model_auris', 'model_avensis',
       'model_aygo', 'model_citigo', 'model_coupe', 'model_i30', 'model_juke',
       'model_micra', 'model_octavia', 'model_panda', 'model_q3', 'model_q5',
       'model_q7', 'model_qashqai', 'model_rapid', 'model_roomster',
       'model_superb', 'model_tt', 'model_x1', 'model_x3', 'model_x5',
       'model_yaris', 'model_yeti', 'Location_Ahmedabad', 'Location_Bangalore',
       'Location_Chennai', 'Location_Coimbatore', 'Location_Delhi',
       'Location_Hyderabad', 'Location_Jaipur', 'Location_Kochi',
       'Location_Kolkata', 'Location_Mumbai', 'Location_Pune',
       'Owner Type_First', 'Owner Type_Fourth Above', 'Owner Type_Second',
       'Owner Type_Third', 'body_type_compact', 'body_type_van',
       'transmission_auto', 'transmission_man', 'fuel_type_diesel',
       'fuel_type_petrol']]


# In[54]:


y=df1_final['Price']


# In[55]:


from sklearn.linear_model import LinearRegression


# In[56]:


model=LinearRegression()


# In[ ]:





# In[57]:


model.fit(x,y)


# In[23]:


y_price=pd.DataFrame(model.predict(x))


# In[24]:


y_price


# In[ ]:





# In[ ]:





# In[ ]:





# In[25]:


df=pd.read_csv('Test_Data_Set.csv')


# In[26]:


df['body_type'].fillna(df['body_type'].mode()[0],inplace=True)
df['Distance '].fillna(df['Distance '].mode()[0],inplace=True)
df['engine_power'].fillna(df['engine_power'].mode()[0],inplace=True)
df['Owner Type']=df['Owner Type'].str.replace('&',' ')


# In[27]:


num=['Id','Distance ', 'manufacture_year', 'Age of car', 'engine_displacement', 'engine_power', 'Vroom Audit Rating', 'door_count', 'seat_count']


# In[28]:


df_num=df[num]


# In[29]:


ohe=OneHotEncoder(sparse=False)


# In[30]:


df_ohe=pd.DataFrame(ohe.fit_transform(df[['Maker','model','Location','Owner Type','body_type','transmission','fuel_type']]))


# In[ ]:





# In[31]:


df_ohe.columns=ohe.get_feature_names_out()


# In[32]:


df_ohe.index=df.index


# In[33]:


df_ohe


# In[34]:


df_final=pd.concat([df_num,df_ohe],axis=1)


# In[ ]:





# In[35]:


df_final[['Owner Type_Fourth  Above']]=df_final[['Owner Type_Fourth   Above']]


# In[36]:


df_final[['Owner Type_Fourth  Above']]


# In[37]:


df_final


# In[ ]:





# In[38]:


model=LinearRegression()


# In[39]:


df_final.columns


# In[40]:


df_final.replace('None',1,inplace=True)


# In[41]:


df_final.rename(columns={'Owner Type_Fourth   Above': 'Owner Type_Fourth Above'}, inplace=True)


# In[58]:


x=df_final[['Id', 'Distance ', 'manufacture_year', 'Age of car',
       'engine_displacement', 'engine_power', 'Vroom Audit Rating',
       'door_count', 'seat_count', 'Maker_audi', 'Maker_bmw', 'Maker_fiat',
       'Maker_hyundai', 'Maker_maserati', 'Maker_nissan', 'Maker_skoda',
       'Maker_toyota', 'model_auris', 'model_avensis', 'model_aygo',
       'model_citigo', 'model_coupe', 'model_i30', 'model_juke', 'model_micra',
       'model_octavia', 'model_panda', 'model_q3', 'model_q5', 'model_q7',
       'model_qashqai', 'model_rapid', 'model_roomster', 'model_superb',
       'model_tt', 'model_x1', 'model_x3', 'model_x5', 'model_yaris',
       'model_yeti', 'Location_Ahmedabad', 'Location_Bangalore',
       'Location_Chennai', 'Location_Coimbatore', 'Location_Delhi',
       'Location_Hyderabad', 'Location_Jaipur', 'Location_Kochi',
       'Location_Kolkata', 'Location_Mumbai', 'Location_Pune',
       'Owner Type_First', 'Owner Type_Fourth Above', 'Owner Type_Second',
       'Owner Type_Third', 'body_type_compact', 'body_type_van',
       'transmission_auto', 'transmission_man', 'fuel_type_diesel',
       'fuel_type_petrol']]


# In[43]:


df_final['Owner Type_Fourth  Above']


# In[ ]:





# In[59]:


y_price=pd.DataFrame(model.predict(x))


# In[ ]:





# In[60]:


y_price.columns=['Price']


# In[61]:


y_price


# In[ ]:





# In[ ]:




