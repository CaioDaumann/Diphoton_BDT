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
import xgboost as xgb
from scipy.stats import chisquare
from sklearn.model_selection import train_test_split

#plotting libraries
import mplhep, hist
import mplhep as hep
plt.style.use([mplhep.style.CMS])
from matplotlib import pyplot

#perform validation plots to check if everything is okay with the samples ...
# probably I should separate the backgrounds here ... 
def validation_plots( data, background, data_weights, mc_weights ,  labels_list ):
    
    for key in labels_list:

        mean = np.mean(  data[key] )
        std  = np.std( data[key]  )

        data_hist              = hist.Hist(hist.axis.Regular(40, mean - 1.2*std,  mean + 1.2*std))
        mc_hist                = hist.Hist(hist.axis.Regular(40, mean - 1.2*std,  mean + 1.2*std))
        flow_hist              = hist.Hist(hist.axis.Regular(40, mean - 1.2*std,  mean + 1.2*std))        

        data_hist.fill( np.array( data[key] )       , weight = data_weights )
        mc_hist.fill(   np.array( background[key] ) , weight = mc_weights )
        #flow_hist.fill( flow_out )

        plott( data_hist , mc_hist, mc_hist , "./plots/" + str(key) + '.png', xlabel = str(key)  )



#main plot function!
def plott(data_hist,mc_hist,mc_rw_hist ,output_filename,xlabel,region=None, density = True  ):

    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, sharex=True)
    
    hep.histplot(
        mc_hist,
        label = 'Diphoton',
        yerr=False,
        density = density,
        color = "blue",
        linewidth=3,
        ax=ax[0],
        flow = 'none'
    )

    hep.histplot(
        mc_rw_hist,
        label = 'GJet',
        density = density,
        color = "red",
        linewidth=3,
        ax=ax[0],
        flow = 'none'
    )

    hep.histplot(
        data_hist,
        label   = "ggH x 100",
        yerr    = False,
        density = density,
        color   = "green",
        #linewidth=3,
        linewidth=3,
        ax=ax[0],
        flow = 'none'
    )



    ax[0].set_xlabel('')
    ax[0].margins(y=0.15)
    ax[0].set_ylim(0, 1.1*ax[0].get_ylim()[1])
    ax[0].tick_params(labelsize=22)

    #log scale for Iso variables
    if( "iso" in str(xlabel) ):
        ax[0].set_yscale('log')
        #ax[0].set_ylim(0.001,( np.max(data_hist)/1.5e6 ))
        ax[0].set_ylim(0.001, 10.05*ax[0].get_ylim()[1])
        

    # line at 1
    ax[1].axhline(1, 0, 1, label=None, linestyle='--', color="black", linewidth=1)#, alpha=0.5)

    #ratio
    data_hist_numpy = data_hist.to_numpy()
    mc_hist_numpy   = mc_hist.to_numpy()
    mc_hist_rw_numpy   = mc_rw_hist.to_numpy()

    integral_data = data_hist.sum() * (data_hist_numpy[1][1] - data_hist_numpy[1][0])
    integral_mc = mc_hist.sum() * (mc_hist_numpy[1][1] - mc_hist_numpy[1][0])

    #ratio betwenn normalizng flows prediction and data
    ratio = (data_hist_numpy[0] / integral_data) / (mc_hist_numpy[0] / integral_mc)
    ratio = np.nan_to_num(ratio)

    integral_mc_rw = mc_rw_hist.sum() * (mc_hist_rw_numpy[1][1] - mc_hist_rw_numpy[1][0])
    ratio_rw = (data_hist_numpy[0] / integral_data) / (mc_hist_rw_numpy[0] / integral_mc_rw)
    ratio_rw = np.nan_to_num(ratio_rw)

    # relative data errors
    errors_nom = (np.sqrt(data_hist_numpy[0])/data_hist_numpy[0]) 
    errors_nom = np.abs(np.nan_to_num(errors_nom))

    hep.histplot(
        ratio,
        bins=data_hist_numpy[1],
        label=None,
        color="blue",
        histtype='errorbar',
        yerr=errors_nom,
        markersize=12,
        elinewidth=3,
        alpha=1,
        ax=ax[1]
    )


    hep.histplot(
        ratio_rw,
        bins=data_hist_numpy[1],
        label=None,
        color="red",
        histtype='errorbar',
        yerr=errors_nom,
        markersize=12,
        elinewidth=3,
        alpha=1,
        ax=ax[1],
    )

    ax[0].set_ylabel("Density", fontsize=26)
    #ax[1].set_ylabel("Data / MC", fontsize=26)
    #ax.set_xlabel( str(xlabel), fontsize=26)
    ax[1].set_ylabel("Ratio", fontsize=26)
    ax[1].set_xlabel( str(xlabel) , fontsize=26)
    if region:
        if not "ZpT" in region:
            ax[0].text(0.05, 0.75, "Region: " + region.replace("_", "-"), fontsize=22, transform=ax[0].transAxes)
        else:
            ax[0].text(0.05, 0.75, "Region: " + region.split("_ZpT_")[0].replace("_", "-"), fontsize=22, transform=ax[0].transAxes)
            ax[0].text(0.05, 0.68, r"$p_\mathrm{T}(Z)$: " + region.split("_ZpT_")[1].replace("_", "-") + "$\,$GeV", fontsize=22, transform=ax[0].transAxes)
    ax[0].tick_params(labelsize=24)
    #ax.set_ylim(0., 1.1*ax.get_ylim()[1])
    ax[1].set_ylim(0.75, 1.25)

    ax[0].legend(
        loc="upper right", fontsize=21
    )

    #hep.cms.label(data=True, ax=ax[0], loc=0, label="Private Work", com=13.6, lumi=21.7)
    #plt.subplots_adjust(hspace=0.03)

    plt.tight_layout()

    fig.savefig(output_filename)

    return 0


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

signal_samples_list  = [ "GluGluHToGG_postEE_M125_2022"]
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


def per_process_BDTscore(model):

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

    general_path_to_data = "/net/scratch_cms3a/daumann/diphotonBDT_project/data/earlyRun3_ntuples_first_prelim_production/"
    sample_list = ["GluGluHToGG_postEE_M125_2022", "VBFtoGG_postEE_M125_2022", "VHtoGG_postEE_M125_2022", "ttHtoGG_postEE_M125_2022", "Diphoton", "GJetPt40toInf", "GJetPt20to40" ]
    keeper = []
    for sample in sample_list:

        files_signal  = glob.glob(  general_path_to_data + str( sample )  + "/nominal/*.parquet")
        signal        = [pd.read_parquet(f) for f in files_signal]
        signal        = pd.concat(signal,ignore_index=True)

        sum_genw_beforesel = 0
        for file in files_signal:
            sum_genw_beforesel += float(pq.read_table(file).schema.metadata[b'sum_genw_presel'])
        signal["weight"] *= (lumi * dict_xSecs[ dict_names[sample] ] / sum_genw_beforesel)
        signal = signal[  (signal.mass > 100) & (signal.mass < 180) ] 

        # In order to mitigate the mass distribution sculpting, we will divide the pt of the photons by the mass
        signal["lead_pt"],signal["sublead_pt"] = signal["lead_pt"]/signal["mass"], signal["sublead_pt"]/signal["mass"]

        # Cos delta phi variable definition 
        signal["cos_dphi"] = np.cos( signal["lead_phi"] - signal["sublead_phi"] )


        keeper.append( signal )

    hist_ggH          = hist.Hist(hist.axis.Regular(40, 0.0,  1.0))
    hist_Diphoton     = hist.Hist(hist.axis.Regular(40, 0.0,  1.0))
    hist_gjet         = hist.Hist(hist.axis.Regular(40, 0.0,  1.0))

    counter = 0
    for element in keeper:

        if( "GG" in sample_list[counter]  ):
            train = xgb.DMatrix(data=  element[var_list] , weight = np.abs(element[weight_label]) ,
                    missing=-999.0, feature_names = var_list )
            predictions = model.predict(train)

            hist_ggH.fill(  predictions , weight =  100*np.array( element[weight_label]  )  )

        if( "Diphoton" in sample_list[counter]  ):
            train = xgb.DMatrix(data=  element[var_list] , weight = np.abs(element[weight_label]) ,
                    missing=-999.0, feature_names = var_list )
            predictions = model.predict(train)

            hist_Diphoton.fill(  predictions , weight =  np.array( element[weight_label]  )  )

        if( "GJet" in sample_list[counter]  ):
            train = xgb.DMatrix(data=  element[var_list] , weight = np.abs(element[weight_label]) ,
                    missing=-999.0, feature_names = var_list )
            predictions = model.predict(train)

            hist_gjet.fill(  predictions , weight =  np.array( element[weight_label]  )  )


        counter = counter + 1

    plott( hist_ggH , hist_Diphoton, hist_gjet , "./results/bdt_per_process.png", xlabel = "BDT ouput", density = False  )

    return 0
