# DiphotonBDT training and evaluation code

#nescessary ones
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import glob
import os

import yaml
from yaml import Loader

import xgboost as xgb
from scipy.stats import chisquare

#plotting libraries
import mplhep, hist
import mplhep as hep
plt.style.use([mplhep.style.CMS])
from matplotlib import pyplot

import traininig_utils as utils
import ploting_utils as plot_utils
from sklearn import metrics

# defining some plot functions - should move this to ploting_utils file
def plot_BDT_output(predictions, test, path_to_plot):
    # plot signal and background separately
    plt.figure()
    plt.hist(predictions[test.get_label().astype(bool)],bins=np.linspace(0,1,50),
                histtype='step',color='midnightblue',label='signal')
    plt.hist(predictions[~(test.get_label().astype(bool))],bins=np.linspace(0,1,50),
                histtype='step',color='firebrick',label='background')
    # make the plot readable
    plt.xlabel('Prediction from BDT',fontsize=18)
    plt.ylabel('Events',fontsize=18)
    plt.legend(frameon=False)   
    plt.savefig( path_to_plot + 'BDT_outputs.png' ) 

def plot_ROC_curve( predictions, test,  test_data,test_labels , path_to_plot, evals_result ):
        
        # choose score cuts:
        cuts = np.linspace(0,0.95,200)
        nsignal = np.zeros(len(cuts))
        nbackground = np.zeros(len(cuts))
        for i,cut in enumerate(cuts):
            nsignal[i] = len(np.where(predictions[test.get_label().astype(bool)] > cut)[0])
            nbackground[i] = len(np.where(predictions[~(test.get_label().astype(bool))] > cut)[0])
            
        # plot efficiency vs. purity (ROC curve)
        plt.figure()
        plt.plot(nsignal/len(test_data[test_labels == 1]),nsignal/(nsignal + nbackground),'o-',color='blueviolet', label = 'AUC -' + str( (np.max(evals_result['train']['auc']) )))

        
        # make the plot readable
        plt.xlabel('Efficiency (nsignal/ntotal)',fontsize=18)
        plt.ylabel('Purity (nsignal/(nsignal + nbkg))',fontsize=18)
        plt.legend(frameon=False)
        plt.savefig( path_to_plot + 'ROC_curve.png' ) 

def plot_model_loss(evals_result, plot_path):
    
    train_losses = evals_result['train']['error']
    val_losses = evals_result['validation']['error']
    

    fig, ax = pyplot.subplots()
    ax.plot(train_losses, label='Train')
    ax.plot(val_losses, label='Test')
    ax.legend()
    plt.savefig( plot_path + 'loss_curve.png')

def plot_model_ams(evals_result, plot_path):
    
    train_losses = evals_result['train']['ams@0']
    val_losses = evals_result['validation']['ams@0']
    

    fig, ax = pyplot.subplots()
    ax.plot(train_losses, label='Train')
    ax.plot(val_losses, label='Test')
    ax.legend()
    plt.savefig( plot_path + 'ams.png')



#start of the main code
def main():

    path_to_plots = './results/'
    x_train, x_test, y_train, y_test, w_train, w_test, inputs_list = utils.load_data()

    train = xgb.DMatrix(data= x_train, weight = w_train ,label= y_train,
                    missing=-999.0, feature_names = inputs_list )

    test = xgb.DMatrix(data= x_test , weight = w_test , label= y_test,
                    missing=-999.0, feature_names = inputs_list )

   # Defining the parameters for the classifier and training
    param = {}

    # Booster parameters
    param['eta']              = 0.1 # learning rate
    param['max_depth']        = 6  # maximum depth of a tree
    param['subsample']        = 0.4 # fraction of events to train tree on
    param['colsample_bytree'] = 1.0 # fraction of features to train tree on

    # Learning task parameters
    param['objective']   = 'binary:logistic' # objective function
    param['eval_metric'] = 'error'           # evaluation metric for cross validation
    param = list(param.items()) + [('eval_metric', 'logloss')] + [('eval_metric', 'auc')] 

    # trying to make in multithreat!
    #param['nthread'] = 16 #by default xgboost already uses everything

    num_trees = 999  # number of trees to make

    evals_result = {}  # object to keep track of the losses
    evals = [(train, "train"), (test, "validation")]

    # perform the training: what is the default? pre stop or post prunning? - need to understand that
    booster = xgb.train(param,train,num_boost_round=num_trees, evals = evals, evals_result=evals_result, verbose_eval=10, early_stopping_rounds=20)
    booster.set_param({"device": "cuda:0"}) #setting it up to cuda!

    # now, evaluating the BDT
    predictions = booster.predict(test)

    # Printing the importance of every variables
    xgb.plot_importance(booster,grid=False)
    plt.savefig( path_to_plots + 'BDT_importance.png' ) #saving the plot

    # Plotting the background and signal BDT scores 
    plot_BDT_output( predictions, test, path_to_plots  )

    # Now, the ROC curve
    plot_ROC_curve( predictions, test, x_test,y_test, path_to_plots, evals_result  )

    #lets also plot the loss curve for the model
    plot_model_loss( evals_result , path_to_plots )
    #plot_model_ams( evals_result , path_to_plots )

    #last plot
    plot_utils.per_process_BDTscore(booster)

    #saving the model in .json format
    booster.save_model("diphoton_BDT.json")

if __name__ == "__main__":
    main()


