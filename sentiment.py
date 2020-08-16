# please install following libraries before running nltk,pandas,matplotlib,seaborn,chardet,numpy,seaborn,demote,sklearn,joblib,pickle
# download nltk.stopwords
# please keep the data and this file in same location



import nltk
#please decomment comment below and download stopwords module
#nltk.download_shell()




#modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')




# checking file type
import chardet
with open('./training.1600000.processed.noemoticon.csv', 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result




#reading file and adding it to dataframe
df=pd.read_csv('./training.1600000.processed.noemoticon.csv',engine = 'python')





#df.head()




#renaming columns below
df.rename(columns = {'0':'sentiment'}, inplace = True)




df.rename(columns = {'1467810369':'id'}, inplace = True)





df.rename(columns = {'Mon Apr 06 22:19:45 PDT 2009':'date'}, inplace = True)





df.rename(columns = {'NO_QUERY':'flag'}, inplace = True)





df.rename(columns = {'_TheSpecialOne_':'user'}, inplace = True)





df.rename(columns = {"@switchfoot http://twitpic.com/2y1zl - Awww, that's a bummer.  You shoulda got David Carr of Third Day to do it. ;D":'text'}, inplace = True)





#df.head()





#df.groupby('sentiment').count()




#converting dataframe df to a list
products_list = df.values.tolist()





#print(products_list[:2])




# shuffling rows of array to increase accuracy
products_list
import random
random.shuffle(products_list)




# converting array back to dataframe
df=pd.DataFrame(products_list,columns=['target','id','date','flag','user','text'])




#using emoji library
import demoji
demoji.download_codes()





# function to convert emoji into text line by line
def rep_emojis(message):
    butter=[]
    bandit=demoji.findall(message)
    no=[char for char in message]
    for word in no:
        if word in bandit.keys():
            butter.append(' ')
            butter.append(word.upper().replace(word, bandit[word]))
        else:
            butter.append(word)
    return ''.join(butter)        



# appliying above function to the dataframe df
df['new']=df['text'].apply(rep_emojis)





#df['text']





#df['new']





import string
from nltk.corpus import stopwords




# function to remove punctuation and stopwords like he,she etc.
def text_process(mess):
    nopunc=[char for char in mess if char not in string.punctuation]
    nopunc=''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    # remove punc ,remove sto words ,return list of clean words




# selecting 2 lakh rows in dataframe of 16 lake rows to process
df=df.iloc[:200000,:]





#text_process(df['new'][1])





from sklearn.feature_extraction.text import CountVectorizer 




# creating an object of countvectoriser to use the function text_process mentioned above to process data and then converting textual data to numbers(bag of words)
bow_transformer=CountVectorizer(analyzer=text_process).fit(df['new'])





#write data
#import pickle
#import joblib
#joblib.dump(bow_transformer, 'bow.pkl') 





#bow_transformer= joblib.load('bow.pkl')




#print(len(bow_transformer.vocabulary_))




# transform textual data in numbers
messages_bow=bow_transformer.transform(df['new'])





#messages_bow.shape





#messages_bow.nnz





#df['target']




# to normalize data
from sklearn.feature_extraction.text import TfidfTransformer



#to nomalize converted numerical data
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer=TfidfTransformer().fit(messages_bow)
messages_tfidf=tfidf_transformer.transform(messages_bow)





#write data
#import pickle
#import joblib
#joblib.dump(tfidf_transformer, 'tfidf.pkl') 





#tfidf_transformer= joblib.load('tfidf.pkl')



# importing training model
from sklearn.naive_bayes import MultinomialNB




# training our data
twitter_sentiment_model=MultinomialNB().fit(messages_tfidf,df['target'])





import pickle





#saved_model = pickle.dumps(twitter_sentiment_model)





#model = pickle.loads(saved_model)





from sklearn import externals
import joblib






  
# Save the model as a pickle in a file 
#please make below line a comment after running it one time
joblib.dump(twitter_sentiment_model, 'twitter_sentiment_model.pkl') 
  
 
  





# Load the model from the file 
model= joblib.load('twitter_sentiment_model.pkl') 




# using pipeline approach to do same we did above for prediction of our accuracy it can be said as second approach all we did above in shorter way
from sklearn.model_selection import train_test_split



# dividie data into train and test data
msr_train,msg_test,label_train,label_test=train_test_split(df['new'],df['target'])





from sklearn.pipeline import Pipeline




# using pipeline to do all the above
pipeline=Pipeline([('bow',CountVectorizer(analyzer=text_process)),
                  ('tfidf',TfidfTransformer()),
                    ('classifier',MultinomialNB())
                  ])




# converting data into bag of words normalising it and training it
pipeline.fit(msr_train,label_train)







#Save the model as a pickle in a file 
#please make below line a comment after running it one time
joblib.dump(pipeline, 'pipeli.pkl') 
  





# Load the model from the file 
model_pipe= joblib.load('pipeli.pkl') 





#type(msg_test)




# using our divided test data 30% of 2 lakh to be inserted into model to for testing
predictions=model_pipe.predict(msg_test)





#predictions[1000]





from sklearn import metrics




#checking accuracy of model
print(metrics.classification_report(label_test,predictions))


# just change whats written inside x to make predictions of model (predictions will be either positive or negative) in your terminal




#predicting value using normal way
x='i am free'
x=rep_emojis(x)
messn=x
bown=bow_transformer.transform([messn])
print(bown)
print(bown.shape)
tfidfn=tfidf_transformer.transform(bown)
if model.predict(tfidfn)[0] ==0:
    print('negative')
else:
    print('positive')
rate=model.predict(tfidfn)[0]




# just change whats written inside x to make predictions of model (predictions will be either 0 means (negative), 4 means (positive) )





#predicting value using pipeline
x='i am free'
x=rep_emojis(x)
messn=x
messn=[messn]
new=model_pipe.predict(messn)
print(new[0])





#df_nyan=pd.DataFrame(columns=['word','terget'])





#df_nyan=df_nyan.append({'word':x,'terget':twitter_sentiment_model.predict(tfidfn)[0]},ignore_index=True)





#bomp=bow_transformer.transform(df_nyan['word'])
#tbomp=tfidf_transformer.transform(bomp)





#twitter_sentiment_model.partial_fit(tbomp,df_nyan['terget'],classes=np.unique(df_nyan['terget']))






























    







