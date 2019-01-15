
import os
import glob


# Converting each of the NewsGroups into a txt file :

# In[4]:


#COMBINING EACH DATAFILE IN EACH TRAIN_NEWSGROUP TO FORM SINGLE NEWSGROUP TEXT FILES

p_train = r'C:\Users\asus\Documents\Classroom\My projects\TextClassification\TRAIN_DATA_NEWSGROUPS'   #LOCATION OF TRAIN DATA
p2_train = r'C:\Users\asus\Documents\Classroom\My projects\TextClassification\TRAIN_DATA'             #LOCATION OF TEXT FILES
listing_w = os.listdir(p_train)                 #List of all the folders in DATA
for infile in listing_w:                        #Iterating through all the folders
    folder = os.path.join(p_train, infile)      #File path of the folder currently iterating
    listing_f = os.listdir(folder)              #All the files in that folder
    pp = infile + '1.txt'                                                       #NEW_FILE NAME
    new_file_location = os.path.join(p2_train, pp)    #creating a textfile of the current iterating folder containing data of all the files in it
    new_file = open(new_file_location, 'w')
    
    for i in listing_f:
        filepath = os.path.join(folder, i)
        file = open(filepath, 'r')
        new_file.write(file.read())
    new_file.close()


# Making EveryCharacter lower case :

# In[5]:


#REMOVING PUNCTUATIONS ANDDELETING THE OLD FILE 
listing_txt = os.listdir(p2_train)

for infile in listing_w:
    pp = infile + '1.txt'                                               #OLDFILE NAME
    file_p = os.path.join(p2_train, pp)
    file = open(file_p, 'r')
    pp_new = infile + '2.txt'                                           #NEWFILE NAME
    file_n_p = os.path.join(p2_train, pp_new)
    new_file = open(file_n_p, 'w')
    for c in file.read():
        if c.isalpha() :
            new_file.write(c.lower())
        else:
            new_file.write(" ")
    file.close()
    new_file.close()


# In[6]:


#REMOVING PUNCTUATIONS FROM EACH OF THE 200 DATA FILES IN EACH NEWSGROUP
p_test = r'C:\Users\asus\Documents\Classroom\My projects\TextClassification\TEST_DATA_NEWSGROUPS'            #LOCATION OF TESTDATA
listing_w_t = os.listdir(p_test)                                                       #FOLDER NAMES IN ABOVE LOCATION

for infile in listing_w_t:
    folder = os.path.join(p_test, infile)
    listing_f = os.listdir(folder)
    for i in listing_f:
        filepath = os.path.join(folder, i)
        file = open(filepath, 'r')
        new_file_name = i + '1'
        new_filepath = os.path.join(folder, new_file_name)
        new_file = open(new_filepath, 'w')
        for c in file.read():
            if c.isalpha():
                new_file.write(c.lower())
            else:
                new_file.write(" ")
        file.close()
        os.remove(filepath)
        new_file.close()


# In[7]:


#LIST CONTAINING THE STOPWORDS
list1 = []
stop_words = open(r'C:\Users\asus\Documents\Classroom\My projects\TextClassification\stopwords_en.txt', 'r')
for line in stop_words:
    words = line.split()
    for word in words:
        list1.append(word)


# Removing Stop_Words from the txt files :

# In[8]:


#REMOVING THE STOPWORDS FROM TRAINING DATA FILES
for infile in listing_w:
    pp = infile + '2.txt'
    file_p = os.path.join(p2_train, pp)
    file = open(file_p, 'r')
    pp_new = infile + '1.txt'
    file_n_p = os.path.join(p2_train, pp_new)
    new_file = open(file_n_p, 'w')
    for line in file :
        words = line.split()
        for word in words:
            if word not in list1 and len(word) > 2:
                new_file.write(word)
                new_file.write(" ")
            else:
                new_file.write(" ")
    file.close()
    new_file.close()


# In[9]:


#REMOVING THE STOPWORDS FROM EACH TEST_FILE OF EACH NEWSGROUP

for infile in listing_w_t:
    folder = os.path.join(p_test, infile)
    listing_f = os.listdir(folder)
    for i in listing_f:
        filepath = os.path.join(folder, i)
        file = open(filepath, 'r')
        new_file_name = i + '.txt'
        new_filepath = os.path.join(folder, new_file_name)
        new_file = open(new_filepath, 'w')
        for line in file:
            words = line.split()
            for word in words:
                if word not in list1 and len(word) > 2:
                    new_file.write(word)
                    new_file.write(" ")
                else:
                    new_file.write(" ")
        file.close()
        os.remove(filepath)
        new_file.close()


# In[10]:


from collections import Counter


# Forming the Dictionary for each Newsgroup

# In[11]:


#FORMING THE DICTIONARY OF WORDS USING THE 20 DATAFILES FORMED ABOVE USING THE COUNTER FUNCTION 
dictionary = {}
output = 0
for infile in listing_w :
    dictionary[output] = {}
    List = []
    pp = infile + '1.txt'
    file_p = os.path.join(p2_train, pp)
    file = open(file_p, 'r')
    for line in file :
        words = line.split()
        for word in words :
            List.append(word)
    K = Counter(List).most_common(1000)
    dictionary[output] = dict(K)
    output += 1
    file.close()


# In[16]:


dictionary


# In[17]:


import pandas as pd
import numpy as np


# In[18]:


#FROM THE DICTIONARY OF THE WORDS ABOVE FINDING THE UNIQUE WORDS FROM EACH Y_CLASS
#FORMING THE DATA 2D ARRAY FROM THE ABOVE UNIQUE WORDS CONSISTING OF THE FREQUENCY FOR EACH WORD TAKEN FOR EACH DATA FILE TAKEN

unique = []
possible_outputs = dictionary.keys()
for output in possible_outputs:
    for j in dictionary[output].keys():
        unique.append(j)
k = set(unique)
#print(k, unique)
data = np.zeros((len(listing_w), len(k)))
for output in possible_outputs:
    lt = []
    for j in k:
        if j in list(dictionary[output].keys()):
            lt.append(dictionary[output][j])
        else:
            lt.append(0)
    data[output] = lt


# In[21]:


data


# In[19]:


# FORMING test NUMPY 2D ARRAY WHICH CONTAINS ALL THE WORD FREQUENCY FROM EACH DATA FILE IN EACH NEWSGROUP

test = np.zeros((4000, len(k)))
List_y = []                                                        #FORMING A LIST OF THE Y VALUES OF EACH DATA FILE
cc = 0
n = 0
for infile in listing_w_t:
    folder = os.path.join(p_test, infile)
    listing_f = os.listdir(folder)
    for i in listing_f:
        List_y.append(cc)
        filepath = os.path.join(folder, i)
        file = open(filepath, 'r')
        Lt = []
        for line in file:
            words = line.split()
            for word in words:
                Lt.append(word)
        K_test = dict(Counter(Lt).most_common(100))                #FORMING A LIST OF 100 most FREQUENCY WORDS FROM EACH FILE
        Ltt = []
        for word in k:                                             #WORDS IN THE DICTIONARY
            if word in list(K_test.keys()):                        #IF PRESENT IN ABOVE LIST APPEND THE COUNT ELSE ZERO
                Ltt.append(K_test[word])
            else :
                Ltt.append(0)
        test[n] = Ltt        
      #  print(test[n].sum())
        n += 1
    cc += 1


# In[13]:


#TESTING DATAFRAME USING THE TEST AND WITH COLUMNS, K
X_test = pd.DataFrame(test, columns = k)
#X_test.sum(axis = 1)


# In[14]:


import pandas as pd


# In[15]:


#TRAINING DATAFRAME FORMED USING THE DATA AND COLUMNS, K
X_train = pd.DataFrame(data, columns= k)


# In[16]:


# FUNCTION TO DETERMINE THE PROBABILITY OF EACH WORD

def prob_word(x_train, word, current_class):
    given_words = x_train.loc[current_class, word] + 1
    total_words = len(k) + 800
    p = np.log(given_words) - np.log(total_words)
    return p


# In[17]:


# FUNCTION TO DETERMINE THE PROBABILITY OF EACH X RECORD IN ACCORDANCE WITH THE NAIVE BAYES ALGORITHM

def probability(x_train, x,current_class):
    output = 0
    prog_log = 0
    #words = columns
    for word, value in x :
        if value != 0:
            p = prob_word(x_train, word, current_class) * value
            prog_log += p
    output = prog_log + np.log(0.05)
    return output


# In[18]:


# FUNCTION TO RETURN THE BEST_CLASS USING THE BEST_PROBABILITY_PRODUCT

def predictSinglePoint(x_train, x):
    classes = dictionary.keys()
    best_p = -1
    best_class = -1
    first_run = True
    for current_class in classes:
        p_current_class = probability(x_train, x, current_class)
        if (first_run or p_current_class > best_p):
            best_p = p_current_class
            best_class = current_class
        first_run = False
    return best_class


# In[19]:


def predict(x_train, x_test):
    y_pred = []
    for i in range(x_test.shape[0]):
        x = np.c_[x_train.columns, x_test.iloc[i]]            #FORMING A 2D NUMPY ARRAY COMPRISING COLUMNS AND VALUES OF THAT RECORD
        x_class = predictSinglePoint(x_train, x)
        #print(i)
        y_pred.append(x_class)
    return y_pred


# In[20]:


#INVOKING THE PREDICT FUNCTION
y_pred = predict(X_train, X_test)


# In[21]:


#DETERMINING THE SCORE USING SELF IMPLEMENTED NAIVE BAYES CLASSIFIER

# a dataframe comparing original y_test values with the predicted_Y
final = pd.DataFrame(np.c_[y_pred, List_y])

n = 0
for i in range(final.shape[0]):
    if y_pred[i] == List_y[i]:
        n += 1
score = n / final.shape[0]
score


# Using the MultinomialNB Classifier

# In[22]:


from sklearn.naive_bayes import MultinomialNB


# In[23]:


#TRAINING_Y VALUES
Y_train = np.arange(20)                      


# In[24]:


#FIT THE DATA WITH TESTING VALUES
MNB = MultinomialNB()
MNB.fit(X_train, Y_train)


# In[25]:


# y variable contains the predicted values for X_test
y = MNB.predict(X_test)


# In[26]:


# forming the original test Y dataframe
y_test = pd.DataFrame(np.array(List_y))


# In[27]:


# determining the score from MULTINOMIALNB CLASSIFIER
s2 = MNB.score(X_test, y_test)
s2


# In[28]:


print("COMPARISON : \nMULTINOMIALNB_SCORE - ", s2, " \nSELF_IMPLEMENTED_SCORE - ", score)

