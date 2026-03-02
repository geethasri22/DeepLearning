#Aim : to implement one hot encoding of words or characters
#method1 :
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
df=pd.DataFrame({'color':['red','blue','green','yellow']})
print('Original Data Frame:\n',df)
print("-"*20)
encoder=OneHotEncoder()
encoded_data = encoder.fit_transform(df[['color']])
encoded_array=encoded_data.toarray()
feature_names=encoder.get_feature_names_out(['color'])
one_hot_df=pd.DataFrame(encoded_array,columns=feature_names)
print("One Hot Encoded Data Frame:")
print(one_hot_df)


#Method2:
'''
import pandas as pd 
df = pd.DataFrame({'Course': ['AIML','CE', 'EEE', 'ME', 'ECE']})
print('Original Data Frame:', df) 
print('-'*20) 
one_hot_df_pd = pd.get_dummies(df, columns =['Course'], dtype=int) 
print("One-Hot Encoded DataFrame (using pandas):") 
print(one_hot_df_pd) 
'''
