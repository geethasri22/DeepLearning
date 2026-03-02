import pandas as pd
from sklearn.preprocessing import OneHotEncoder
sentence = "I am staying in Vijayawada." 
words = sentence.lower().replace('.', '').split()
print(f"Tokenized words: {words}") 
print("-" * 30) 
df = pd.DataFrame(words, columns=['word'])
print("DataFrame from words:")
print(df)
print("-" * 30) 
encoder = OneHotEncoder(sparse_output=False) 
encoded_words = encoder.fit_transform(df[['word']]) 
feature_names = encoder.get_feature_names_out(['word']) 
one_hot_df = pd.DataFrame(encoded_words, columns=feature_names) 
print("One-Hot Encoded DataFrame:")
print(one_hot_df) 
