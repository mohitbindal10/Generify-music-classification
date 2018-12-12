import os
import timeit
import numpy as np
from collections import defaultdict
import glob
import sys
import scipy
import scipy.io.wavfile
from python_speech_features import mfcc
import matplotlib
import matplotlib.pyplot
from matplotlib.pyplot import specgram
from utils import GENRE_DIR, CHART_DIR, GENRE_LIST
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc
from sklearn.cross_validation import ShuffleSplit
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from collections import Counter
from scipy.stats import mode

from utils import plot_roc_curves, plot_confusion_matrix, ENGLISH_GENRE_DIR, ENGLISH_GENRE_LIST, TEST_DIR
import mysql.connector
from ceps import read_ceps, create_ceps_test, read_ceps_test

from pydub import AudioSegment
test_file = "EnterSandman.wav"
test_path="/Users/mohitbindal/Desktop/testset/"+test_file
genre_list = ENGLISH_GENRE_LIST
GENRE_DIR=ENGLISH_GENRE_DIR

clf = None

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#          Please run the classifier script first
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def test_model_on_single_file(test_path):
    clf = joblib.load('saved_model/englishLDAmodel.pkl')
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
            print(probs)
            max_prob = max(probs)
            print(max_prob)
            for i,j in enumerate(probs):
                if probs[i] == max_prob:
                    max_prob_index=i
            print (max_prob_index)
            predicted_genre = genre_list[max_prob_index]
            result.append(predicted_genre)
    
    pg=Counter(result).most_common(1)
    
    for gr in genre_list:
        if gr in pg:
            predicted_genre=gr
    
    print("PRediccted GEnre is =======>>>>>>>>>>>",predicted_genre)
    Y[Y==0]=1
    specgram(Y, Fs=sample_rate,xextent=(0,30))
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
    cur.execute( "INSERT INTO eng_genre_finder VALUES( %s,%s)",(test_file,predicted_genre))
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
    
