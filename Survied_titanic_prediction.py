# -*- coding: utf-8 -*-
"""
Created on Mon May 31 21:58:12 2021

@author: Ravik
"""
import os
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import Binarizer

os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'

path_train=r'train.csv'
path_test=r'test.csv'
test_data=pd.read_csv(path_test)
imputer=SimpleImputer()
test_data['Age']=imputer.fit_transform(test_data['Age'].values.reshape(-1,1))
test_data['SibSp']=imputer.fit_transform(test_data['SibSp'].values.reshape(-1,1))
test_data['Parch']=imputer.fit_transform(test_data['Parch'].values.reshape(-1,1))
test_data_df=test_data.drop(['PassengerId','Name', 'Ticket','Cabin', 'Fare'],axis=1).copy()
test_data_df=pd.concat([test_data_df,pd.get_dummies(test_data_df['Pclass'],prefix='Pclass')],axis=1)
le=LabelEncoder()
le.fit(['male','female'])
test_data_df['Sex']=le.transform(test_data_df['Sex'].astype(str))
test_data_df.isnull().sum()
test_data_df.dropna(inplace=True)
le.fit(['S','Q','C'])
test_data_df['Embarked']=le.transform(test_data_df["Embarked"].astype(str))
scalar=MinMaxScaler(feature_range=(0,1))
test_data_df['Age']=scalar.fit_transform(test_data_df['Age'].values.reshape(-1,1))
X_testing=test_data_df.values
test_data=pd.read_csv(path_test)
train_data=pd.read_csv(path_train)
imputer=SimpleImputer()
train_data['Age']=imputer.fit_transform(train_data['Age'].values.reshape(-1,1))
train_data['SibSp']=imputer.fit_transform(train_data['SibSp'].values.reshape(-1,1))
train_data['Parch']=imputer.fit_transform(train_data['Parch'].values.reshape(-1,1))
train_data_df=train_data.drop(['PassengerId','Name', 'Ticket','Cabin', 'Fare'],axis=1).copy()
train_data_df=pd.concat([train_data_df,pd.get_dummies(train_data_df['Pclass'],prefix='Pclass')],axis=1)
le=LabelEncoder()
le.fit(['male','female'])
train_data_df['Sex']=le.transform(train_data_df['Sex'].astype(str))
train_data_df.isnull().sum()
train_data_df.dropna(inplace=True)
le.fit(['S','Q','C'])
train_data_df['Embarked']=le.transform(train_data_df["Embarked"].astype(str))
scalar=MinMaxScaler(feature_range=(0,1))
train_data_df['Age']=scalar.fit_transform(train_data_df['Age'].values.reshape(-1,1))
lb=LabelBinarizer()
train_data_df['Survived']=lb.fit_transform(train_data_df['Survived'])
X_training=train_data_df.drop("Survived",axis=1).values
Y_training=train_data_df[["Survived"]].values

learning_rate=0.001
training_epochs = 100
display_step=5

no_of_inputs=9
no_of_outputs=1

#define how many neurons we want in each layer of our NN
layer_1_node=50
layer_2_node=100
layer_3_node=50

#section one: Define the layers of the NN itself


#input layer
with tf.variable_scope("input"):
    X=tf.placeholder(tf.float32,shape=(None,no_of_inputs))
    
#layer1
with tf.variable_scope("layer_1"):
    weights= tf.get_variable(name="weights1",shape = [no_of_inputs,layer_1_node],initializer=tf.contrib.layers.xavier_initializer())
    biases= tf.get_variable(name="biases1",shape=[layer_1_node],initializer=tf.zeros_initializer())    
    layer_1_output = tf.nn.relu(tf.matmul(X, weights) + biases)
    
with tf.variable_scope("layer_2"):
    weights=tf.get_variable(name="weights2",shape=[layer_1_node,layer_2_node],initializer=tf.contrib.layers.xavier_initializer())
    biases=tf.get_variable(name="biases2",shape=[layer_2_node],initializer=tf.zeros_initializer())
    layer_2_output=tf.nn.relu(tf.matmul(layer_1_output,weights)+biases)
    
with tf.variable_scope("layer_3"):
    weights=tf.get_variable(name="weights3",shape=[layer_2_node,layer_3_node],initializer=tf.contrib.layers.xavier_initializer())
    biases=tf.get_variable(name='biases3',shape=[layer_3_node],initializer=tf.zeros_initializer())
    layer_3_output=tf.nn.relu(tf.matmul(layer_2_output,weights)+biases)

with tf.variable_scope("output"):
    weights=tf.get_variable(name='weights4',shape=[layer_3_node,no_of_outputs],initializer=tf.contrib.layers.xavier_initializer())
    biases=tf.get_variable(name="biases4",shape=[no_of_outputs],initializer=tf.zeros_initializer())
    predicted=tf.nn.relu(tf.matmul(layer_3_output,weights)+biases)
    
    test_data_df["survived"]=predicted
    
with tf.variable_scope("cost"):
    Y=tf.placeholder(tf.float32,shape=(None,1))
    cost=tf.reduce_mean(tf.squared_difference(predicted, Y))

with tf.variable_scope("train"):
    optimizer=tf.train.AdamOptimizer(learning_rate).minimize(cost)
    

with tf.variable_scope("loging"):
    tf.summary.scalar('current_cost',cost)
    summary=tf.summary.merge_all()
    
saver=tf.train.Saver()
    





with tf.Session() as session:
    
    session.run(tf.global_variables_initializer()) 
    
    training_writer=tf.summary.FileWriter("./logs/traning",session.graph)
    testing_writer=tf.summary.FileWriter("./logs/testing",session.graph)
    
    for epoch in range(training_epochs):
        session.run(optimizer,feed_dict={X:X_training,Y:Y_training})

        if epoch%5==0:
            traning_cost,training_summary= session.run([cost,summary],feed_dict={X:X_training,Y:Y_training})
            
            
            training_writer.add_summary(training_summary,epoch)
                        
            print(epoch,traning_cost)
    print("training is completed")
    final_training_cost= session.run(cost,feed_dict={X:X_training,Y:Y_training})
    print(final_training_cost) 
    save_path=saver.save(session,'./log/trained_model.ckpt')
    test_predict=session.run(predicted,feed_dict={X:X_testing})
    
binarizer=Binarizer(0.5)
test_prediction=binarizer.fit_transform(test_predict)
print(test_prediction)
