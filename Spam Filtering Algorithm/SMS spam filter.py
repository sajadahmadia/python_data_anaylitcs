#!/usr/bin/env python
# coding: utf-8

# # Building An SMS Spam Filter Using Multinomial Naive Bayes.

# ## Introduction:
# 
# In this project, we are going to be using the `Multinomial Naive Bayes` algorithm to classify SMS as either spam messages or non-spam messages. The `Multinomial Naive Bayes` algorithm is based on `Bayes' Theorem`; this theorem describes the probability of an event based on prior knowledge of conditions that might be related to the event.
# 
# Our goal in this project is to train the computer how to classify messages using the `Multinomial Naive Bayes` algorithm as well as a dataset of 5,572 SMS messages that have already been classified by humans. 
# The dataset was put together by Tiago A. Almeida and José María Gómez Hidalgo and it can be downloaded from the <a href='https://archive.ics.uci.edu/ml/datasets/sms+spam+collection'>The UCI Machine Learning Repository<a/>. 
# 
# Our algorithm should at least have an accuracy score 80% for us to be confident in it.
#     

# ## Initial Data Exploration

# In[1]:


import pandas as pd


# In[8]:


sms = pd.read_csv('SMSSpamCollection',
                 sep ='\t',
                 header = None,
                 names = ['Label', 'SMS']
                 )
sms.head()


# In[9]:


sms.shape


# In[10]:


sms['Label'].value_counts(normalize=True) * 100 


# ## Splitting Our Dataset Into Training and Test Datasets:
# 
# Nearly 87% of the messages in our dataset are ham(non-spam) messages and about 13% of them are spam messages. Now that we are farmiliar with our dataset we are going to split it into a training set and a test set. 
# We are going to keep 80% of our dataset for training and 20% for testing; this is because we want to train the algorithm on as much data as possible and still have enough data for testing the algorithm. The dataset has 5,572 messages, which means that:
# 
# * The training set will have 4,458 messages (about 80% of the dataset).
# * The test set will have 1,114 messages (about 20% of the dataset).

# In[11]:


sample = sms.sample(frac=1, random_state=1) 
training_set = sample.iloc[0:4458].reset_index(drop=True) 
test_set = sample.iloc[4458:].reset_index(drop=True)


# In[12]:


print(training_set.shape)

print(test_set.shape)


# In[13]:


#getting the percentages of spam and non-spam messages in our training and tests set
print('training set:', '\n',
      training_set['Label'].value_counts(normalize=True) * 100
     )
print('\n')
print('test set:', '\n',
      test_set['Label'].value_counts(normalize=True) * 100
     )


# The perecentages for spam and non-spam messages in both our training and test dataset are roughly equal to the percentages of spam and non-spam messages in our original sms dataset.

# ## Data Cleaning.
# To train the algorithm we are going to clean our dataset t make it easier for to calculate the probability of each individual word in the dataset. We are going to transform the dataset. The result we will get is going to look like this.
# 
# <img src = 'https://dq-content.s3.amazonaws.com/433/cpgp_dataset_3.png'/>

# In[14]:


#before transformation
training_set.head()


# In[18]:


#removing punctuations
training_set['SMS'] = training_set['SMS'].str.replace('\W', ' ', regex=True)

#turning every texts lower case
training_set['SMS'] = training_set['SMS'].str.lower()


# In[19]:


training_set.head()


# In[20]:


#extracting every individual word in the sms column
training_set['SMS'] = training_set['SMS'].str.split()
vocabulary = []
for value in training_set['SMS']:
    for i in value:
        vocabulary.append(i)
vocabulary = set(vocabulary) # using the set function gets rid of duplicate values
vocabulary = list(vocabulary) # transforming the vocabulary variable from a set back to a list

len(vocabulary)


# ## Final Transformation Of The Training Dataset.
# 
# There are 7783 unique words in our vocabulary. We are going to create a dictionary containing each unique word as a key and the frequency of the word as its key. Then we are going to transform the dictionary to a pandas dataframe and then combine it with our `training_set` dataframe.

# In[21]:


#creates a dictionary with every unique word in the vocabulary list
word_counts_per_sms = {unique_word: [0] * len(training_set['SMS']) for unique_word in vocabulary}

for index, message in enumerate(training_set['SMS']):
    for word in message:
        word_counts_per_sms[word][index] += 1


# In[22]:


word_counts = pd.DataFrame(word_counts_per_sms)
word_counts.head()


# In[23]:


clean_training_set = pd.concat([training_set, word_counts], axis=1) # joins the dataset together
clean_training_set.head()


# ## Probability Of Spam & Non-Spam Messages.
# Here, we are going to do the following:
# 
# * split our training dataset into datasets for ham and spam messages.
# * calculate the probability of the constants. To do this, we are going:
#   1. Calculate the probability of both ham and spam messages.
#   2. Create a variable alpha with a value of 1.
#   3. Calculate the number of words in both ham and spam messages.
#   4. Calculate the number of words in the vocabulary dictionary.
#   
# * Calculate the parameters. This includes the probability of every word given that a message is either spam or ham. 

# ### Calculating the constants:

# In[24]:


spam_messages = clean_training_set[clean_training_set['Label']=='spam'].shape[0] # returns the number of rows with spam messages

ham_messages = clean_training_set[clean_training_set['Label']=='ham'].shape[0] # returns the number of rows with ham messages

total_messages = clean_training_set.shape[0]

probability_spam = spam_messages / total_messages
probability_ham = ham_messages / total_messages

print('probability of spam messages:', '\n', probability_spam)
print('\n')
print('probability of non-spam messages:', '\n', probability_ham)


# In[25]:


spam_words = clean_training_set[clean_training_set['Label']=='spam']['SMS'].apply(len) # len function counts the individual words in the sms column
n_spam = spam_words.sum() 

ham_words = clean_training_set[clean_training_set['Label']=='ham']['SMS'].apply(len)
n_ham = ham_words.sum()

n_vocabulary = len(vocabulary)
alpha = 1 #laplace smoothing


# ### Calculating the parameters:

# In[26]:


spam_dict = {unique_word:0 for unique_word in vocabulary} # initializes  a dictionary with every word in the vocabulary list as a key.
ham_dict = {unique_word:0 for unique_word in vocabulary}

#creating new DataFrames for both Spam and Ham messages
spam_df = clean_training_set[clean_training_set['Label']=='spam'].copy() #using copy() method avoids SettingWithCopy Warning.
ham_df = clean_training_set[clean_training_set['Label']=='ham'].copy()

#calculating the probability for each word in spam messages
for word in vocabulary:
    n_word_spam_messages = spam_df[word].sum() # the number of times the word occurs in the spam DataFrame
    p_word_spam_messages = (n_word_spam_messages + alpha) / (n_spam + alpha * n_vocabulary)
    spam_dict[word] =  p_word_spam_messages # updates the dictionary values with the probability of each unique word
    

#calculating the probability for each word in ham messages
for word in vocabulary:
    n_word_ham_messages = ham_df[word].sum() # the number of times the word occurs in the spam DataFrame
    p_word_ham_messages = (n_word_ham_messages + alpha) / (n_ham + alpha * n_vocabulary)
    ham_dict[word] =  p_word_ham_messages # updates the dictionary values with the probability of each unique word
        


# ## Classifying Messages:
# 
# * First we are going to write a mock `classify()`function that does the classification and confirm that it can classify messages as spam or non-spam.
# * We are going to update the `classify()` function and then use it on our test_set DataFrame.

# In[27]:


import re

def classify(message):
    ''' Takes in a string value
    and cleans it'''

    message = re.sub('\W', ' ', message)
    message = message.lower()
    message = message.split()


    p_spam_given_message = probability_spam
    p_ham_given_message = probability_ham
    
    for word in message:
# multiplies the already known probability for spam messages by the probability of a word being in a spam message
        if word in spam_dict:
            p_spam_given_message *= spam_dict[word] 
            
# multiplies the already known probability for sham messages by the probability of a word being in a ham message
            
        if word in ham_dict:
            p_ham_given_message *= ham_dict[word]
            
   

    print('P(Spam|message):', p_spam_given_message)
    print('P(Non-Spam|message):', p_ham_given_message)

    if p_ham_given_message > p_spam_given_message:
        print('Label: Non-Spam')
    elif p_ham_given_message < p_spam_given_message:
        print('Label: Spam')
    else:
        print('Equal proabilities, have a human classify this!')


# In[28]:


# Testing the fucntion
test_message1 = 'WINNER!! This is the secret code to unlock the money: C3421.'
test_message2 = "Sounds good, Tom, then see u there"

classify(test_message1)
print('\n')
classify(test_message2)


# In[29]:


def classify_test_set(message):

    message = re.sub('\W', ' ', message)
    message = message.lower()
    message = message.split()


    p_spam_given_message = probability_spam
    p_ham_given_message = probability_ham
    
    for word in message:
# multiplies the already known probability for spam messages by the probability of a word being in a spam message
        if word in spam_dict:
            p_spam_given_message *= spam_dict[word] 
            
# multiplies the already known probability for sham messages by the probability of a word being in a ham message
            
        if word in ham_dict:
            p_ham_given_message *= ham_dict[word]
            

    if p_ham_given_message > p_spam_given_message:
        return 'ham'
    elif p_ham_given_message < p_spam_given_message:
        return 'spam'
    else:
        return 'needs human classification'


# In[30]:


#applying the classify function to the training_set DataFrame
test_set['predicted'] = test_set['SMS'].apply(classify_test_set)
test_set.head()


# ## Calculating The Accuracy Of The Algorithm.
# To calculate the accuracy of the algorithm, we are going to do the following:
# 
# * initialize a variable `correct` with a value of 0 and also a variable `total` with the total number of messages in the test set.
# * Loop through our `test_set` DataFrame and increment the value of correct by 1 if the `Label` column is the same as the `predicted` column.
# * Finally we divide `correct` by `total` to get our accuracy score.

# In[31]:


# function labels each row as correct or incorrect if we predicted correctly
def classification(row):
    if row['Label'] != row['predicted']:
        return 'incorrect'
    else:
        return 'correct'


# In[32]:


# creates a new colun that shows the classification of each column
test_set['classification'] = test_set.apply(classification, axis=1)

test_set.head()


# In[33]:


# calculating the number of incorrect columns
incorrect = test_set[test_set['classification'] == 'incorrect']
incorrect


# In[34]:


# getting the number of incorrectly classified messages
incorrect.shape[0] - 1 # subtract one to account for the message that needs human classification


# In[35]:


# calculating the accuracy of the algorithm
correct = 0
total = test_set.shape[0]

for row in test_set.iterrows():
    row = row[1]
    if row['Label'] == row['predicted']:
        correct += 1

accuracy = correct / total * 100
accuracy


# From the accuracy score, our algorithm is doing pretty good. We managed to classify 98.7% of messages correctly.

# ## Conclusion:
# We wanted to create an algorithm that classifies messages as either spam or non-spam messages using the multinomial Naive Bayes algorithm. So far we have been able to do that and we had an accuracy score of 98.7%. This means that we are confident that in the real world our algorithm will do so well.
# Although our algorithm is doing quite good, it failed to classify 13 messsages correctly. There are a few things we can do to improve on it. We could make the algorithm case sensitive to see if it will increase the accuracy.
# 
# -- written by Sajad Ahamdi
