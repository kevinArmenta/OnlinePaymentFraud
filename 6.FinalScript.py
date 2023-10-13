#Final Script for Online Payment Fraud Project
#Kevin Armenta, 9/11/23


#------------------------------IMPORT ALL IMPORTANT PACKAGES--------------------------------------------------------
#General Management
import gc as gc
gc.enable()
from joblib import dump, load
from warnings import filterwarnings

#Data Handling
import pandas as pd, numpy as np 
#Time
import time
#Plotting
import matplotlib.pyplot as plt, seaborn as sns, scipy.stats, pylab
#TomekLinks and RandomUnderSampler
from imblearn.under_sampling import TomekLinks, RandomUnderSampler

#ML - train and test split
from sklearn.model_selection import train_test_split
#ML - Scalers
from sklearn import preprocessing
#ML - Hyperparameter optimization
import optuna
#ML - Metrics
from sklearn.metrics import f1_score, balanced_accuracy_score, recall_score, roc_auc_score
#ML - Final Model
from sklearn.naive_bayes import GaussianNB
#Visualize ML algorithm features
import shap

#Saving Data
import pickle



#-------------------------- FUNCTIONS ------------------------#
#Tranform numerical columns
def num_transform(df, name, newname, rootvalue=4):
    col = df[name] #Grab column
    df[newname] = col**(1/rootvalue) #Transform it and create new column
    df.drop(name,inplace=True,axis=1) #Drop original column
    return df

#Rearrange columns
def col_arrange(df, num):
    col_order = df.columns.tolist()
    col_order = col_order[-num:] + col_order[:-num]
    df = df[col_order]
    return df

#Transform name columns into usable format
def name_transform(df, colname, new_colname1, new_colname2):
    #grab the column from the dataframe
    name_df = df[colname].tolist()
    
    #Grab first letter of each entry
    name_first = [i[0] for i in name_df]
    
    #Grab the ID #
    name_ID = [i[1:] for i in name_df]
    
    #Put into df
    df['dummy_var'] = name_ID
    
    #Create new list
    #1 if 'C' & '0' if M
    first_bool = [1 if i=='C' else 0 for i in name_first]
    
    #Add the boolean list to the original dataframe
    df[new_colname1] = first_bool
    df[new_colname1] = df[new_colname1].astype('bool')
    
    #Create a pivot table of the IDs and grabt he ID#s
    name_pt = df['dummy_var'].value_counts()
    name_ptID = name_pt.index
    
    #Grab all the repeat IDs 
    name_repeat = name_pt[name_pt>1]
    ID_repeat = name_repeat.index
    
    #Loop and assign 0 if the ID repeats and 1 if the ID doesn't
    ID_bool = []
    for i in name_ID:
        if i in ID_repeat:
            ID_bool.append(0)
        else:
            ID_bool.append(1)
    
    #Add the ID_bool to the dataframe
    df[new_colname2] = ID_bool
    df[new_colname2] = df[new_colname2].astype('bool')
    
    #Drop the old column
    df.drop(colname,axis=1,inplace=True)
    df.drop('dummy_var',axis=1,inplace=True)
    
    return df

#Automaties one-hot encoding process for both training and test processes
def OHEncode(df, colname, encoder = None, train = True):
    if train:
        ohe = preprocessing.OneHotEncoder(handle_unknown="ignore",drop='first',sparse=False) #instantiate one hot encoder
        encoded_col = pd.DataFrame(data=ohe.fit_transform(df[colname].array.reshape(-1,1)), columns=ohe.get_feature_names_out()) #encode the type variable in the training data
        e_col = encoded_col.columns.tolist() #grab the column names from the encoded data
        #Convert the column types to boolean
        for col in e_col:
            encoded_col[col] = encoded_col[col].astype('bool')
        df = pd.concat([encoded_col,df],axis=1) #Add columns to dataframe
        df.drop(colname,axis=1,inplace=True) #Drop type column
        return df, ohe
    else:  
        encoded_col = pd.DataFrame(data=encoder.transform(df[colname].array.reshape(-1,1)), columns=encoder.get_feature_names_out()) #encode the type variable in the test data using the same one hot encoder
        e_col = encoded_col.columns.tolist() #Grab the column names from the encoded data 
        #Convert each column to a boolean
        for col in e_col:
            encoded_col[col] = encoded_col[col].astype('bool')
        df = pd.concat([encoded_col,df],axis=1) #Add columns to dataframe
        df.drop(colname,axis=1,inplace=True) #Drop type column
        return df

#Automate the Tomek Links algorithm for the training data
def Tomek(model_data, target, sample_strat = 'majority'):
    modeling_data_columns = model_data.columns.tolist()
    tlink = TomekLinks(sampling_strategy=sample_strat, n_jobs=2)
    X, y = tlink.fit_resample(model_data, target)
    tomek_modeling_data = pd.DataFrame(data=X, columns=modeling_data_columns) #Change to dataframe
    y = pd.DataFrame(y,columns=['isFraud']) #Change to dataframe
    return tomek_modeling_data, y



#Process a raw data file (train or test)*********
def data_processor(rawCSVfilename, test_split=0.01, dev_split=0.1, rstate=5):
    #Read in the data in CSV file format
    #Include the entire directory if its in a different file
    data = pd.read_csv(rawCSVfilename)

    #Get the data ready for transformations
    data = data.rename(columns={"oldbalanceOrg":"oldbalanceOrig"})
    data.drop('isFlaggedFraud',inplace=True,axis=1) #Remove isFlaggedFraud column
    #Define the column types
    dictionary_cols = {'step':'int64','type':'object','amount':'float64',
                      'nameOrig':'object','oldbalanceOrig':'float64','newbalanceOrig':'float64',
                      'nameDest':'object', 'oldbalanceDest':'float64', 'newbalanceDest':'float64',
                      'isFraud':'int64'}
    data = data.astype(dictionary_cols) #Change all the data types to be what we want

    #split the data
    train, test = train_test_split(data,test_size=test_split,random_state=rstate)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)  

    #--------------Process training data
    #------Numerical data
    train = num_transform(train,'amount','amount_4root')
    train = num_transform(train,'oldbalanceOrig','oldbalanceOrig_4root')
    train = num_transform(train,'newbalanceOrig','newbalanceOrig_4root')
    train = num_transform(train,'oldbalanceDest','oldbalanceDest_4root')
    train = num_transform(train,'newbalanceDest','newbalanceDest_4root')
    #Rearrange columns
    train = col_arrange(train, 5) #since we added 5 columns we need to re-arrange them and bring them to front
    
    #-------Categorical data
    #nameOrig
    train = name_transform(train,'nameOrig','ClientOrig','NO_ID_unique')
    #nameDest
    train = name_transform(train,'nameDest','ClientDest','ND_ID_unique')    
    #type
    train, ohe = OHEncode(train,'type') #pull out the ohe to use for the test data
    #step
    train.drop('step',axis=1,inplace=True)

    #Grab Modeling Train Data
    #Extract modeling data from training data (all vars except target variable)
    modeling_data_columns = [x for x in train.columns if x!= 'isFraud']
    modeling_data = train[modeling_data_columns]
    target_data = train['isFraud']

    #------------Tomek Links
    tomek_modeling_data, y = Tomek(modeling_data, target_data)

    #Split up the tomek data
    tomek_X_train, tomek_X_dev, tomek_y_train, tomek_y_dev = train_test_split(tomek_modeling_data, y, stratify=y, test_size=dev_split, random_state=rstate)


    #------------------Process testing data
    #Create the dataframe for the prediction variable
    test_y = test['isFraud']
    test_y = pd.DataFrame(test_y,columns=['isFraud']).astype('int')
    #Drop isFraud column
    test.drop('isFraud',axis=1,inplace=True)

    #------Numerical data
    test = num_transform(test,'amount','amount_4root')
    test = num_transform(test,'oldbalanceOrig','oldbalanceOrig_4root')
    test = num_transform(test,'newbalanceOrig','newbalanceOrig_4root')
    test = num_transform(test,'oldbalanceDest','oldbalanceDest_4root')
    test = num_transform(test,'newbalanceDest','newbalanceDest_4root')
    #Rearrange columns
    test = col_arrange(test, 5) #since we added 5 columns we need to re-arrange them and bring them to front

    #-------Categorical data
    #nameOrig
    test = name_transform(test,'nameOrig','ClientOrig','NO_ID_unique')
    #nameDest
    test = name_transform(test,'nameDest','ClientDest','ND_ID_unique') 
    #type
    test = OHEncode(test,'type',encoder=ohe,train=False)
    #step
    test.drop('step',axis=1,inplace=True)

    #Finalize the column order
    test = test[modeling_data_columns]


    #------------------RETURN THE PROCESSED DATA
    final_datasets_dict = {'tomek_X_train':tomek_X_train, 'tomek_X_dev':tomek_X_dev, 
                           'tomek_y_train':tomek_y_train, 'tomek_y_dev':tomek_y_dev,
                           'test':test, 'test_y':test_y}
    return final_datasets_dict


#Modeling the train set, testing on dev and test sets
def GNB_modeling(Xtrain,ytrain,Xdev,ydev,Xtest,ytest,varsmooth= 9.125860889745052*(10**-9)):
    #Create the Gaussian Naive Bayes model
    fmodel_tomek = GaussianNB(var_smoothing=varsmooth)
    #Fit the model to the training data
    fmodel_tomek.fit(Xtrain,ytrain.values.ravel())

    #Record metrics of model on dev set
    dev_pred = fmodel_tomek.predict(Xdev) #predictions on the dev set
    dev_recall = round(recall_score(ydev,dev_pred),3) #recall
    dev_f1 = round(f1_score(y_true=ydev, y_pred=dev_pred),3) #F1 score
    dev_bacc = round(balanced_accuracy_score(ydev, y_pred=dev_pred),3) #balanced accuracy

    #Record metrics of model on test set
    test_pred = fmodel_tomek.predict(Xtest)
    test_recall = round(recall_score(ytest, test_pred),3)
    test_f1 = round(f1_score(y_true=ytest, y_pred=test_pred),3)
    test_bacc = round(balanced_accuracy_score(ytest, y_pred=test_pred),3)
    
    #Save metrics
    metrics = [dev_recall,dev_f1,dev_bacc,test_recall,test_f1,test_bacc]

    final_dict = {'model':fmodel_tomek, 'dev set predictions':dev_pred, 
                  'test set predictions':test_pred, 'metrics':metrics}

    return final_dict


#SHAP analysis on test data
def SHAP_analysis(model, train, test, sample1=400, sample2=100):
    shap_data = shap.sample(train,nsamples=sample1) #This just samples data from the original training dataset
    explainer = shap.KernelExplainer(model.predict_proba,shap_data) #Create an explainer
    #SUPER LONG RUN-TIME!!!!!!!!
    shap_values_tomek = explainer.shap_values(test,nsamples=sample2) #Create the shap values based on the test data
    
    abs_mean_shap = np.mean(np.array([np.absolute(np.array(shap_values_tomek[0])), np.absolute(np.array(shap_values_tomek[1]))]), axis=0)

    #Get a dataframe of the shap values for each feature, for each sample. Then, create an average per feature
    shap_df = pd.DataFrame(data=abs_mean_shap, columns=test.columns) #IMPORTANT SHAP DATAFRAME WE WANT IN TABLEAU
    shap_avg_df = pd.DataFrame(shap_df.mean().to_dict(),index=[shap_df.index.values[-1]])

    #Reformat the dataframe
    shap_plot_df = shap_avg_df.T.reset_index() #IMPORTANT SHAP DATAFRAME WE WANT FOR TABLEAU
    shap_plot_df.columns = ['Feature', 'Mean ABS SHAP']

    return shap_df, shap_plot_df




#----------------------- PROCESSING DATA -------------------------#
six_datasets = data_processor('onlinefraud.csv')
#Pull datasets out
tomek_X_train = six_datasets['tomek_X_train']
tomek_X_dev = six_datasets['tomek_X_dev']
tomek_y_train = six_datasets['tomek_y_train']
tomek_y_dev = six_datasets['tomek_y_dev']
test_X = six_datasets['test']
test_y = six_datasets['test_y']

combined_test = pd.concat([test_X,test_y],axis=1,join='outer')
#combined_test.to_csv('test_data.csv')

#--------------------------- MODELING ----------------------------#
model_dict = GNB_modeling(Xtrain=tomek_X_train,ytrain=tomek_y_train,
                          Xdev=tomek_X_dev,ydev=tomek_y_dev,
                          Xtest=test_X,ytest=test_y)

fmodel_tomek = model_dict['model']
devset_pred = model_dict['dev set predictions']
testset_pred = model_dict['test set predictions']
metrics = model_dict['metrics']

testset_pred2 = pd.DataFrame(testset_pred,columns=['isFraud_prediction']).astype('int')
combined_test = pd.concat([combined_test, testset_pred2],axis=1,join='outer')
combined_test.to_csv('test_data2.csv')

#-------------------------- SHAP VALUES --------------------------#
# shap_df, shap_summary = SHAP_analysis(model=fmodel_tomek, train = tomek_X_train, test = test_X)
# #It all works!


# #Going to save with and without indices just in case
# shap_df.to_csv('shap_df.csv')
# shap_df.to_csv('shap_df2.csv',index=False)

# shap_summary.to_csv('shap_summary.csv')
# shap_summary.to_csv('shap_summary2.csv',index=False)