print("welcome to genre_detection")
import os
import numpy as np
import sys
print(sys.path)
import scipy
import scipy.io.wavfile
from python_speech_features import mfcc
import matplotlib
import matplotlib.pyplot
from matplotlib.pyplot import specgram
from utils import GENRE_DIR,  GENRE_LIST

from sklearn.externals import joblib
from collections import Counter

from utils import GENRE_DIR, fetch_genre, TEST_DIR


test_file = "output_ghazal.wav"
test_path="/Users/mohitbindal/Desktop/testset/"+test_file
genre_list = fetch_genre(test_path)
print(genre_list)

clf = None
    

import mysql.connector
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#          Please run the classifier script first
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def test_model_on_single_file(file_path):
    clf = joblib.load('saved_model/hindiknnmodel.pkl')
    print(clf)
    
    result=[]
    #for i in range(0,100):
    sample_rate,X = scipy.io.wavfile.read(test_path)
    
    #print(" X is " , X, "len of x is ",len(X),X.shape)
    Z=[]
    Y=X.ravel()
    ceps = mfcc(Y)
    num_ceps = len(ceps)
    Z.append(np.mean(ceps[int(num_ceps / 10):int(num_ceps * 9 / 10)], axis=0))
    
            #print(" Z is " , Z, "len of z is ",len(Z))
                            
               
                           
    for x in range(0,100):             
            
            probs = clf.predict_proba(Z)
            print(probs)
            print ("\t".join(str(x) for x in genre_list))
            print ("\t".join(str("%.3f" % x) for x in (probs[0])))
            probs=probs[0]
            #print(probs)
            max_prob = max(probs)
            #print(max_prob)
            for i,j in enumerate(probs):
                if probs[i] == max_prob:
                    max_prob_index=i
            #print (max_prob_index)
            predicted_genre = genre_list[max_prob_index]
            result.append(predicted_genre)
    
    pg=Counter(result).most_common(1)
    
    for gr in genre_list:
        if gr in pg:
            predicted_genre=gr
    
    print("PRediccted GEnre is =======>>>>>>>>>>>",predicted_genre)
    Y[Y==0]=1
    specgram(Y, Fs=sample_rate//100,xextent=(0,0.1))
    matplotlib.pyplot.show()
    hostname = 'localhost'
    username = 'root'
    password = ''
    database = 'minor'
    
    # Simple routine to run a query on a database and print the results:
    print ("Using mysql.connector…")
    
    myConnection = mysql.connector.connect( host=hostname, user=username, passwd=password, db=database )
    if(myConnection)  :  
        print("connection successful")
    cur = myConnection.cursor()
    if(cur):
        print("cursor successfully formed")
    cur.execute( "INSERT INTO genre_finder VALUES( %s,%s)",(test_file,predicted_genre))
    myConnection.commit()
    
    print("insert successfull")
    myConnection.close()
    return predicted_genre

if __name__ == "__main__":

    global traverse
    for subdir, dirs, files in os.walk(GENRE_DIR):
        traverse = list(set(dirs).intersection(set(GENRE_LIST)))
        break

    # should predict genre as "ROCK"
    predicted_genre = test_model_on_single_file(test_path)
    