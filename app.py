#=================flask code starts here
from flask import Flask, render_template, request, redirect, url_for, session,send_from_directory
import os
from werkzeug.utils import secure_filename
from distutils.log import debug
from fileinput import filename
import pandas as pd
#importing all required python libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from math import sqrt
import os
from sklearn.metrics import mean_absolute_error
from sklearn import svm
from minepy import MINE #loading class to select features using MIC (Maximal Information Coefficient)
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, RepeatVector, Bidirectional, LSTM, GRU, AveragePooling2D
from keras.layers import Convolution2D
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint
import pickle
import matplotlib.pyplot as plt #use to visualize dataset vallues
from keras.layers import *
from keras.models import *
from keras import backend as K
from sklearn.linear_model import LinearRegression

import numpy as np
import smtplib 
from email.message import EmailMessage
from datetime import datetime
from werkzeug.utils import secure_filename
import sqlite3
import pandas as pd
import numpy as np
import pickle
import sqlite3
import random

import smtplib 
from email.message import EmailMessage
from datetime import datetime



UPLOAD_FOLDER = os.path.join('static', 'uploads')
# Define allowed files
ALLOWED_EXTENSIONS = {'csv'}


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'welcome'

#defining self attention layer
class attention(Layer):
    def __init__(self, return_sequences=True, name=None, **kwargs):
        super(attention,self).__init__(name=name)
        self.return_sequences = return_sequences
        super(attention, self).__init__(**kwargs)

    def build(self, input_shape):

        self.W=self.add_weight(name="att_weight", shape=(input_shape[-1],1),
                               initializer="normal")
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1),
                               initializer="zeros")

        super(attention,self).build(input_shape)

    def call(self, x):

        e = K.tanh(K.dot(x,self.W)+self.b)
        a = K.softmax(e, axis=1)
        output = x*a

        if self.return_sequences:
            return output

        return K.sum(output, axis=1)


#class to normalize dataset values
scaler = MinMaxScaler(feature_range = (0, 1))
scaler1 = MinMaxScaler(feature_range = (0, 1))

#loading and displaying Aneshtesia clinical dataset
dataset = pd.read_csv("Dataset/household_power_consumption.csv", sep=";", nrows=10000)
#converting sub meter values as float data     
dataset['Sub_metering_1'] = dataset['Sub_metering_1'].astype(float)
dataset['Sub_metering_2'] = dataset['Sub_metering_2'].astype(float)
dataset['Sub_metering_3'] = dataset['Sub_metering_3'].astype(float)
dataset.fillna(0, inplace = True)

#applying dataset processing such as converting date and time into numeric values and then summing all 3
#submeters consumption as single target value to forecast future electricity
dataset['Date'] = pd.to_datetime(dataset['Date'])
dataset['year'] = dataset['Date'].dt.year
dataset['month'] = dataset['Date'].dt.month
dataset['day'] = dataset['Date'].dt.day
dataset['Time'] = pd.to_datetime(dataset['Time'])
dataset['hour'] = dataset['Time'].dt.hour
dataset['minute'] = dataset['Time'].dt.minute
dataset['second'] = dataset['Time'].dt.second
dataset['label'] = dataset['Sub_metering_1'] + dataset['Sub_metering_2'] + dataset['Sub_metering_3']
dataset.drop(['Date', 'Time', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'], axis = 1,inplace=True)
dataset.fillna(0, inplace = True)

Y = dataset['label'].ravel() #getting target column
dataset.drop(['label'], axis = 1,inplace=True)
columns = dataset.columns
X = dataset.values #get dataset features
print("Total features exists in Dataset before applying MIC features Selection algorithm : "+str(X.shape[1]))
mic_scores = []
mine = MINE()
for i in range(0, len(columns)-1):#loop and compute mic score for each features
    mine.compute_score(X[:,i], Y)
    mic_scores.append((columns[i], mine.mic()))
# Sort features by MIC score
mic_scores.sort(key=lambda x: x[1], reverse=True)
# Select top features
top_features = [feature for feature, _ in mic_scores[:8]]  # Select top 2 features
X = dataset[top_features]
print("Total features exists in Dataset before applying MIC features Selection algorithm : "+str(X.shape[1]))
X = dataset.values

Y = Y.reshape(-1, 1)
scaler = MinMaxScaler((0,1))
scaler1 = MinMaxScaler((0,1))
X = dataset.values
X = scaler.fit_transform(X)
Y = scaler1.fit_transform(Y)

@app.route('/home', methods=['GET', 'POST'])
def home():
    return render_template('home.html')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/notebook')
def notebook():
    return render_template('ElectricityConsumption.html')


def getModel():
    extension_model = Sequential()
    extension_model.add(Convolution2D(32, (1 , 1), input_shape = (10, 1, 1), activation = 'relu'))
    extension_model.add(MaxPooling2D(pool_size = (1, 1)))
    extension_model.add(Convolution2D(32, (1, 1), activation = 'relu'))
    extension_model.add(MaxPooling2D(pool_size = (1, 1)))
    extension_model.add(Flatten())
    extension_model.add(RepeatVector(3))
    extension_model.add(attention(return_sequences=True,name='attention')) # ========define Attention layer
    #adding bidirectional LSTM as CRNN layer
    extension_model.add(Bidirectional(GRU(64, activation = 'relu', reset_after=False)))#==================adding BIGRU
    extension_model.add(RepeatVector(3))
    extension_model.add(Bidirectional(GRU(64, activation = 'relu', reset_after=False)))#==================adding BIGRU
    #defining output classification layer with 256 neurons 
    extension_model.add(Dense(units = 256, activation = 'relu'))
    extension_model.add(Dropout(0.3))
    extension_model.add(Dense(units = 1))
    extension_model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    extension_model.load_weights("model/extension_weights.hdf5")
    return extension_model

@app.route('/PredictAction', methods=['GET', 'POST'])
def PredictAction():
    if request.method == 'POST':
        f = request.files.get('file')
        data_filename = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'],data_filename))
        session['uploaded_data_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'],data_filename)
        data_file_path = session.get('uploaded_data_file_path', None)
        testData = pd.read_csv(data_file_path, sep=";")
        #print(type(testData))
        #testData = pd.read_csv("Dataset/testData.csv")#read test data
        extension_model = getModel()
        data = testData.values
        #handling and removing missing values        
        testData.fillna(0, inplace = True)
        testData['Date'] = pd.to_datetime(testData['Date'])#convert date and time to year, month, day, hour, second and minutes
        testData['year'] = testData['Date'].dt.year
        testData['month'] = testData['Date'].dt.month
        testData['day'] = testData['Date'].dt.day
        testData['Time'] = pd.to_datetime(testData['Time'])
        testData['hour'] = testData['Time'].dt.hour
        testData['minute'] = testData['Time'].dt.minute
        testData['second'] = testData['Time'].dt.second
        testData.drop(['Date', 'Time'], axis = 1,inplace=True)
        testData.fillna(0, inplace = True)
        X = testData[top_features]#select MIC top features
        testData = testData.values
        testData = scaler.transform(testData)#normalize dataset values
        testData = np.reshape(testData, (testData.shape[0], testData.shape[1], 1, 1))
        predict = extension_model.predict(testData)#predict electricity consumption using extension model
        predict = predict.reshape(-1, 1)
        predict = scaler1.inverse_transform(predict)#reverse normalize predicted SOC to normal integer value
        output = ""
        for i in range(len(predict)):
            output += "Test Data = "+str(data[i])+" Predicted Electricity Consumption ===> "+str(predict[i,0])+"<br/><br/>"
        return render_template('result.html', msg=output)

@app.route('/logon')
def logon():
	return render_template('signup.html')

@app.route('/login')
def login():
	return render_template('signin.html')

@app.route("/signup")
def signup():
    global otp, username, name, email, number, password
    username = request.args.get('user','')
    name = request.args.get('name','')
    email = request.args.get('email','')
    number = request.args.get('mobile','')
    password = request.args.get('password','')
    otp = random.randint(1000,5000)
    print(otp)
    msg = EmailMessage()
    msg.set_content("Your OTP is : "+str(otp))
    msg['Subject'] = 'OTP'
    msg['From'] = "myprojectstp@gmail.com"
    msg['To'] = email
    
    
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login("myprojectstp@gmail.com", "paxgxdrhifmqcrzn")
    s.send_message(msg)
    s.quit()
    return render_template("val.html")

@app.route('/predict_lo', methods=['POST'])
def predict_lo():
    global otp, username, name, email, number, password
    if request.method == 'POST':
        message = request.form['message']
        print(message)
        if int(message) == otp:
            print("TRUE")
            con = sqlite3.connect('signup.db')
            cur = con.cursor()
            cur.execute("insert into `info` (`user`,`email`, `password`,`mobile`,`name`) VALUES (?, ?, ?, ?, ?)",(username,email,password,number,name))
            con.commit()
            con.close()
            return render_template("signin.html")
    return render_template("signup.html")

@app.route("/signin")
def signin():

    mail1 = request.args.get('user','')
    password1 = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
    data = cur.fetchone()

    if data == None:
        return render_template("signin.html")    

    elif mail1 == str(data[0]) and password1 == str(data[1]):
        return render_template("home.html")
    else:
        return render_template("signin.html")


    
if __name__ == '__main__':
    app.run()