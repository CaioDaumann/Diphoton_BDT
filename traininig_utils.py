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

import xgboost
from scipy.stats import chisquare
from sklearn.model_selection import train_test_split

#plotting libraries
import mplhep, hist
import mplhep as hep
plt.style.use([mplhep.style.CMS])
from matplotlib import pyplot

import ploting_utils as plot_utils

dict_xSecs = {
    # in fb
    "ggH": 52.23e3 * 0.00227,
    "VBF": 4.078e3 * 0.00227,
    "VH": 2.4009e3 * 0.00227,
    "ttH": 0.5700e3 * 0.00227,
    "Diphoton": 89.14e3,
    "GJetPT20to40": 242.5e3,
    "GJetPT40": 919.1e3,
}
lumi = 20.7

signal_samples_list  = [ "GluGluHToGG_postEE_M125_2022","VBFtoGG_postEE_M125_2022","VHtoGG_postEE_M125_2022","ttHtoGG_postEE_M125_2022"]
bkg_samples_list     = [ "Diphoton", "GJetPt40toInf", "GJetPt20to40" ]

dict_names = {
 
    "GluGluHToGG_postEE_M125_2022" : "ggH",
    "VBFtoGG_postEE_M125_2022"     : "VBF",
    "VHtoGG_postEE_M125_2022"      : "VH",
    "ttHtoGG_postEE_M125_2022"     : "ttH",
    "Diphoton"                     : "Diphoton",
    "GJetPt40toInf"                : "GJetPT40",
    "GJetPt20to40"                 : "GJetPT20to40",


}

# This function normalize the weights to lumi, xsec, etc, ...
def normalize_weights():
    return 0

def load_data():
    
    general_path_to_data = "/net/scratch_cms3a/daumann/diphotonBDT_project/data/earlyRun3_ntuples_first_prelim_production/"

    # making the loop automatic for signal and background 
    signals = []
    for sample in signal_samples_list:

        files_signal  = glob.glob(  general_path_to_data + str( sample )  + "/nominal/*.parquet")
        signal        = [pd.read_parquet(f) for f in files_signal]
        signal        = pd.concat(signal,ignore_index=True)

        sum_genw_beforesel = 0
        for file in files_signal:
            sum_genw_beforesel += float(pq.read_table(file).schema.metadata[b'sum_genw_presel'])
        signal["weight"] *= (lumi * dict_xSecs[ dict_names[sample] ] / sum_genw_beforesel)
        signal = signal[  (signal.mass > 100) & (signal.mass < 180) ] 

        signals.append( signal )
    
    signal = pd.concat( signals )

    #loop over background samples now
    bkgs = []
    for sample in bkg_samples_list:

        files_bkg = glob.glob(  general_path_to_data + str( sample ) + "/nominal/*.parquet")
        background   = [pd.read_parquet(f) for f in files_bkg]
        background   = pd.concat(background,ignore_index=True)

        # now, lets normalize these weights here:
        sum_genw_beforesel = 0
        for file in files_bkg:
            sum_genw_beforesel += float(pq.read_table(file).schema.metadata[b'sum_genw_presel'])
        background["weight"] *= (lumi * dict_xSecs[ dict_names[sample] ] / sum_genw_beforesel)
        background = background[  ( background.mass > 100) & (background.mass < 180) ]

        bkgs.append( background )

    background = pd.concat( bkgs )

    #list of the variable that will be used as input to the BDT
    var_list = [

                "lead_pt",
                "lead_eta",
                "lead_phi",
                "lead_mvaID",
                "lead_energyErr",
                "sublead_pt",
                "sublead_eta",
                "sublead_phi",
                "sublead_mvaID",
                "sublead_energyErr",
                "fixedGridRhoAll",
                "cos_dphi",
                "sigma_m_over_m",
                "dZ",
                "nPV",
                "n_jets"]

    # some problems with sceta, phi and rho, so I took them out of the training!

    weight_label = ["weight"]

    # In order to mitigate the mass distribution sculpting, we will divide the pt of the photons by the mass
    signal["lead_pt"],signal["sublead_pt"] = signal["lead_pt"]/signal["mass"], signal["sublead_pt"]/signal["mass"]
    background["lead_pt"],background["sublead_pt"] = background["lead_pt"]/background["mass"], background["sublead_pt"]/background["mass"]

    # Cos delta phi variable definition 
    signal["cos_dphi"] = np.cos( signal["lead_phi"] - signal["sublead_phi"] )
    background["cos_dphi"] = np.cos( background["lead_phi"] - background["sublead_phi"] )

    #calculating some displacements
    for key in signal.keys():
        print( key )
    exit()


    # I am simply taking the abs of the weights for now ... (of course, this need to be changed xD)
    # should we increase the weights of the signal events? I am afraid they are too low and the network does not give enough importance to them ...
    signal_inputs, signal_weights       =  signal[var_list]     , np.abs(signal[weight_label])
    backgound_inputs, background_weights = background[var_list] , np.abs(background[weight_label])

    # now lets reescale the weights so we have the same sum of w for signal and bkg samples - weights are very important!
    signal_weights = (signal_weights/np.sum(signal_weights))*np.sum( background_weights )

    # now, we will validate it
    plot_utils.validation_plots( signal, background , signal[weight_label] , background[weight_label],var_list )

    # labels now:
    signal_labels      = np.ones( len(signal) )
    background_labels  = 0*np.ones( len(background) )

    print( 'Number of signal events: ', len( signal_labels ) , ' Number of background events: ', len( background_labels ) )

    train_data    = np.concatenate( [ signal_inputs  , backgound_inputs ], axis = 0 )
    train_labels  = np.concatenate( [ signal_labels  , background_labels ], axis = 0 )
    train_weights = np.concatenate( [ signal_weights , background_weights ], axis = 0 )

    #shuffle
    permutation = np.random.permutation(len(train_data))
    train_data    = train_data[permutation]
    train_labels  = train_labels[permutation]
    train_weights = train_weights[permutation]

    #now split the data into training and test sets, and also shuffle everything
    x_train, x_test, y_train, y_test, w_train, w_test = train_test_split(train_data, train_labels, train_weights, test_size=0.3, shuffle=True)

    return x_train, x_test, y_train, y_test, w_train, w_test, var_list