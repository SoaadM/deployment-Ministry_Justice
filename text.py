import streamlit as st
import pickle
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import re
import  string
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd 
from sklearn.model_selection import train_test_split



data = pd.read_csv('data_clean_text1.csv', index_col=0)

X_train=data['judgment_text_1']
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2)


def text_similirty(text):

    cv_tfidf = TfidfVectorizer(min_df=3,max_df=0.9)
    X_tf = cv_tfidf.fit_transform(X_train)

    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components = 100)
    svdMatrix1 = svd.fit_transform(X_tf)



    similarities =cosine_similarity(svdMatrix1[1:2,:],svdMatrix1) 

    index = np.argsort(similarities[0])[-20:]
    index1= list(reversed(index))

    value = ((similarities[0][index])[-20:]) 
    result=list(reversed(value))

    z = 0
    for l, i in zip(result, index1):
            d = l * 100
            if d >= 95: 
                   continue
            
            else:
                st.write('%','نسبة تشابه القضيتين', round(d))
                st.write('القضية:', data.judgment_text_1.iloc[i])
                st.write('---------------------')
                z = z +1 
                if z == 5: 
                    break
         



def main():
    st.title("نسبة تشابه القضايا  ")
    text = st.text_area("ادخل القضية :")


   
  
    if st.button("Predict"):      
        output=text_similirty(text)
        

if __name__=='__main__':
    main()
