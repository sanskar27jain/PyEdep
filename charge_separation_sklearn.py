import sys, os

import numpy as np
import matplotlib.pyplot as plt

from ROOT import TFile, TTree

import pandas as pd

import warnings
warnings.filterwarnings("error") # so that we can use try-except on warnings raised by numpy 

class EventsTreeReader:

    def __init__(self, model_list_filename=".txt", out_filename="result.pdf", cv_score="accuracy", model="Logistic", f_option="light"):
        # Hopefully these are the newSpline files only 
        # self.discrNue = "~/PyEdepS/AR23_newSplines_writer_files/atm_nue_writer_out_AR23_Gv3.root"
        # self.discrAntinue = "~/PyEdepS/AR23_newSplines_writer_files/atm_antinue_writer_out_AR23_Gv3.root"

        self.discrNue     = "~/PyEdepS/AR23_newSplines_writer_files/L_thre/atm_nue_writer_out_with_L_thre_AR23_Gv3.root"
        self.discrAntinue = "~/PyEdepS/AR23_newSplines_writer_files/L_thre/atm_antinue_writer_out_with_L_thre_AR23_Gv3.root"

        self.contiNue = "~/PyEdepS/FlatEspectrum_nue_writer_out_AR23_newSplines.root"
        self.contiAntinue = "~/PyEdepS/FlatEspectrum_antinue_writer_out_AR23_newSplines.root"

        self.model_list_filename = model_list_filename
        self.out_filename = out_filename
        self.cv_score = cv_score

        ###### OLD CONTROL STRUCTURE ######
        self.model = model
        self.features = f_option
        # if "no_light" in self.out_filename:
        #     self.option = "no_light"
        #     print("Excluding L based on filename")

        # if "no_q500" in self.out_filename:
        #     self.option = "no_q500"
        #     print("Excluding Q_500 based on filename")

        # if "no_q75" in self.out_filename:
        #     self.option = "no_q75"
        #     print("Excluding Q_75 based on filename")

        # if "voting" in self.out_filename:
        #     self.option = "voting"

        # if "L_threshold" in self.out_filename:
        #     self.option = "L_threshold"
        #     print("Using L_75, L_500, Q based on filename")
        ############

        # In case somehow the plots directory does not exist in PyEdep directory (from event.py)
        if not os.path.exists("./plots"):
            os.makedirs("./plots")
            print("./plots directory did not exist. It was created.")

        self.GetEnergies()

    def GetEnergies(self):
        f_dN =  TFile.Open(self.discrNue)
        f_dA =  TFile.Open(self.discrAntinue)
        f_cN =  TFile.Open(self.contiNue)
        f_cA =  TFile.Open(self.contiAntinue)

        edepTreeDN = f_dN.Sim
        edepTreeDA = f_dA.Sim
        edepTreeCN = f_cN.Sim
        edepTreeCA = f_cA.Sim

        # For debugging purposes
        print("Number of entries in discrete E_nu nue edep-sim tree:     "    , edepTreeDN.GetEntries())
        print("Number of entries in discrete E_nu antinue edep-sim tree: "    , edepTreeDA.GetEntries())
        print("Number of entries in continuous E_nu nue edep-sim tree:     "    , edepTreeCN.GetEntries())
        print("Number of entries in continuous E_nu antinue edep-sim tree: "    , edepTreeCA.GetEntries())

        self.events_info = ({}, {}, {}, {})
        
        for edepTree, events_info in zip([edepTreeDN, edepTreeDA, edepTreeCN, edepTreeCA], self.events_info):
            events_info['E_nu']    = np.array([])
            events_info['E_avail'] = np.array([]) # A 1-row array. Every element in the array is the total E_avail of one specific event (entry).
            events_info['E_dep']   = np.array([])
                                      #no threshold, 75keV, 500keV
            events_info['Q']  = np.array([[], [], []])
                                                            #35, 100, 140, 180, 220 PEpMeV
            events_info['L']       = np.array([[], [],  [],  [],  []])

                                      #no threshold, 75keV, 500keV
            events_info['L_LY180_thre'] = np.array([[], [], []])

            events_info['nu_proc'] = np.array([])
            events_info['nu_pdg']  = np.array([], dtype=np.int64)

            events_info['nEvents'] = 0

            for entry in edepTree:
                events_info['E_nu']    = np.append(events_info['E_nu']   , entry.E_nu)
                events_info['E_avail'] = np.append(events_info['E_avail'], entry.E_avail)
                events_info['E_dep']   = np.append(events_info['E_dep']  , entry.E_depoTotal)
                Q_thresholds = [[entry.Q_depoTotal, entry.Q_depoTotal_th_75keV, entry.Q_depoTotal_th_500keV]]
                events_info['Q']  = np.append(events_info['Q'] , np.array(Q_thresholds).T, axis=1)
                L_PCEs = [[entry.L_depoTotal_avg_35PEpMeV, entry.L_depoTotal_avg_100PEpMeV, entry.L_depoTotal_avg_140PEpMeV, entry.L_depoTotal_avg_180PEpMeV, entry.L_depoTotal_avg_220PEpMeV]]
                events_info['L']  = np.append(events_info['L'] , np.array(L_PCEs).T,       axis=1) # Deposited energy in scintillation light assuming a light yield of _ photoelectrons per MeV for MIPs (assuming 0.5 kV/cm E field, ___ detector PCE, Birks Model)

                if hasattr(entry, 'L_depoTotal_avg_180PEpMeV_th_75keV'):
                    L_thresholds = [[entry.L_depoTotal_avg_180PEpMeV, 
                                     entry.L_depoTotal_avg_180PEpMeV_th_75keV,
                                     entry.L_depoTotal_avg_180PEpMeV_th_500keV]]
                    events_info['L_LY180_thre']  = np.append(events_info['L_LY180_thre'], np.array(L_thresholds).T, axis=1)

                events_info['nu_proc'] = np.append(events_info['nu_proc'], entry.nu_proc)
                events_info['nu_pdg'] = np.append(events_info['nu_pdg'], int(entry.nu_pdg))

                events_info['nEvents'] += 1

        ###### Light Threshold Test ######
        # E_nu = 500
        # info = self.events_info[0]
        # Q, Q_75, Q_500  = info['Q'][:, info['E_nu']==E_nu] 
        # L, L_75, L_500  = info['L_LY180_thre'][:, info['E_nu']==E_nu] 

        # bin_width = E_nu / 100

        # min_x, max_x = np.concatenate((Q, Q_75, Q_500, L, L_75, L_500)).min(), np.concatenate((Q, Q_75, Q_500, L, L_75, L_500)).max()
        # midrange_x, halfrange_x = (min_x + max_x) / 2, (max_x - min_x) / 2
        # xlim_min, xlim_max = midrange_x - (1.1 * halfrange_x), midrange_x + (1.1 * halfrange_x)
        # nearest_greater_multiple = xlim_max + bin_width - (xlim_max % bin_width)
        # bin_edges = np.linspace(0, nearest_greater_multiple, int(nearest_greater_multiple / bin_width) + 1)

        # fig, axs = plt.subplots(2, figsize=(6.4, 4.8*2))
        # axs[0].hist(Q,     bins=bin_edges, histtype="step", label=r"$Q$")
        # axs[0].hist(Q_75,  bins=bin_edges, histtype="step", label=r"$Q_{75}$") 
        # axs[0].hist(Q_500, bins=bin_edges, histtype="step", label=r"$Q_{500}$")

        # axs[1].hist(L,     bins=bin_edges, histtype="step", label=r"$L$")
        # axs[1].hist(L_75,  bins=bin_edges, histtype="step", label=r"$L_{75}$") 
        # axs[1].hist(L_500, bins=bin_edges, histtype="step", label=r"$L_{500}$")

        # for ax in axs:
        #     ax.set_xlim(xlim_min, xlim_max) 
        #     ax.set_xlabel("Energy [MeV]", fontsize="large")
        #     ax.set_ylabel("Number of events", fontsize="large")
        #     ax.legend(loc=("upper left" if E_nu > 200 else "best"),fontsize="large")
        # fig.suptitle(rf"$\nu$:{info['nu_pdg'][0]}, $E_{{\nu}}$ = {E_nu} MeV, bin width = {bin_width} MeV, {info['nEvents']} events", fontsize='small')
        # self.SaveAndClose()
        ############

        ###### OLD CONTROL STRUCTURE ######
        # if self.option == "voting":
        #     self.QvsLPlotsVotingModel()
        # elif self.option == "L_threshold":
        #     self.QvsLPlotsLightThreshold()
        # else:
        #     self.QvsLPlots(no_light=(self.option == "no_light"), no_q500=(self.option == "no_q500"), no_q75=(self.option == "no_q75"), scoring=self.cv_score)
        ###################################
                                                                                              #so that index of dataframe is not first entry of each tuple
        self.model_list = list( pd.read_csv(self.model_list_filename,delimiter=',').itertuples(index=False, name=None) )
                                                                                                            #so that it just returns normal tuples instead of 'named tuples'

        # self.model_list = [("Q75"       , "Logistic"),
        #                    ("Q75 L"     , "Logistic"),
        #                    ("Q75 Q500"  , "Logistic"),
        #                    ("Q75 Q500 L", "Logistic"),
        #                    ("L75"       , "Logistic"),
        #                    ("L75 Q"     , "Logistic"),
        #                    ("L75 L500"  , "Logistic"),
        #                    ("L75 L500 Q", "Logistic")]
                        #    ("Q75"       , "SVM"),
                        #    ("Q75 L"     , "SVM"),
                        #    ("Q75 Q500"  , "SVM"),
                        #    ("Q75 Q500 L", "SVM")]
        
        self.output_list = []

        for feature_str, model_nm in self.model_list:
            self.output_list.append( self.calcROCDiscreteE_nu(feature_str=feature_str, model_nm=model_nm) )

        ## sanity checks ##
        in_feats_col_selection = lambda df : [col_name for col_name in df.columns if col_name!="PredID"]
        # Make sure all models using the same test (input-features) set (and by extension same train set)
        for i in range(1, len(self.output_list)):
            od1, od2 = self.output_list[i-1:i+1]
            assert (od1['Test set'][in_feats_col_selection] == od2['Test set'][in_feats_col_selection]).all(axis=None) 
            
        for i, od in enumerate(self.output_list):
            # make sure best params and best model are consistent
            for k, v in od['Best params'].items(): 
                # One of the hyperparameters which we are trying to optimize for the AdaBDT is the "estimator" parameter, which (in this case) takes in a DecisionTreeClassifier object which in turn has varying values of its max_depth hyperparam. However, two different DecisionTreeClassifier objects, even if they share all parameters, are not == equal unless they are literally the same object in memory. So, the DecisionTreeClassifier object that gets stored in cvgs.best_params_ (which *is* the dictionary at cvgs.cv_results_['params'][cvgs.best_index_], containing the *literal* same objects) is a different object from the DT which gets created to be passed into the new copy of the AdaBDT being refit with the best params.  
                if k=='estimator':
                    assert v.get_params()['max_depth'] == od['Best model'].get_params()[k].get_params()['max_depth']
                else:
                    assert v == od['Best model'].get_params()[k] 
        
            # Make sure that features used by model is consistent with feature_str passed into fitting method
            for fstr in od['Best model'].feature_names_in_:
                assert fstr in self.model_list[i][0].split()
            # Test set Decision scores out-put in out_dict are consistent with scores computed by correspondingly out-put Best model on test set. 
            assert np.all( od['Best model'].decision_function(od['Test set'][self.model_list[i][0].split()]) == od['Decision scores'] )

        #### METRICS PLOT ####
        self.plotMetrics(option="ConfusionMatrix")
        self.plotMetrics(option="DiscriminatorHistogram")

    #---------------------------------------------------
    def plotMetrics(self, option="ConfusionMatrix"):
        subplot_mosaic = np.array([f'CM{i}' for i in range(len(self.output_list))])                                     #|
        subplot_mosaic = np.append( subplot_mosaic, np.array( ['.']*( 4 - (len(subplot_mosaic)%4) ) ) ).reshape(-1, 4)  #| -> Creates a nested list called subplot_mosaic which looks like [['.', 'CM0', 'CM1', 'CM2', 'CM3',],          
        subplot_mosaic = np.append(np.array([['.']*subplot_mosaic.shape[0]]).T, subplot_mosaic, axis=1)                 #|                                                                  ['.', 'CM4', 'CM5', 'CM6', 'CM7',],...
        subplot_mosaic = subplot_mosaic.tolist() + [['.']+['ROC']*4]                                                    #|                                                               ...['.', 'ROC', 'ROC', 'ROC', 'ROC' ]]   
        print("Subplot mosaic\n",subplot_mosaic)                                                             

        fig, ax_dict = plt.subplot_mosaic( mosaic=subplot_mosaic, 
                                           width_ratios=[0.1,1,1,1,1],  # Remember, the first column is all '.', for padding
                                           height_ratios=[1]*(len(subplot_mosaic)-1) + [3], # The ROC plot should be three times as tall as the CM/score histogram plots 
                                           figsize=( 16, 4*(len(subplot_mosaic)-1) + 12 ) ) # Each subplot-unit (mosaic array element) should basically be 4x4 
        # fig.set_layout_engine('tight')
        fig.set_layout_engine('constrained') # "Constrained layout is similar to Tight layout, but is substantially more flexible. 
                                             # It **handles colorbars placed on multiple Axes** ... and **Axes that span rows or columns (subplot_mosaic)**, striving to align spines from Axes in the same row or column." - mpl 3.11 user guide

        from sklearn.metrics import ConfusionMatrixDisplay
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize 
        
        for i, out_dict in enumerate(self.output_list):
            ## Confusion Matrices ##
            if option=="ConfusionMatrix":
                cmap_to_use = 'inferno'
                cmd = ConfusionMatrixDisplay.from_predictions(y_true=np.ravel(out_dict['Test set']["ID"]),
                                                              y_pred=np.ravel(out_dict['Test set']["PredID"]),
                                                              labels=(1, -1),     # The idea is that this along with the display_labels parameter ought to link the class-label of 1 to nue and the class-label of -1 to antinue, whenever it sees 1 or -1 in the lists of true or predicted labels y_true or y_pred. "List of labels to index the confusion matrix. This may be used to **reorder** or select a subset of labels. If None is given, those that appear at least once in y_true or y_pred are used in sorted order."
                                                              normalize=None,     # It will display the **actual** event counts on the plot, not proportions of the total event count (normalize='all')
                                                              display_labels=(r"$\nu_e$", r"$\bar{\nu_e}$"), # "Target names used for plotting. By default, _labels_ will be used if it is defined, otherwise the unique labels of y_true and y_pred will be used."
                                                              ax=ax_dict[f'CM{i}'],  # Plot it onto the Axes from the subplot_mosaic
                                                              colorbar=False,
                                                              cmap=cmap_to_use)
    
                colormap_normalizer = Normalize(vmin=0,vmax=len(out_dict['Test set'])) # linearly normalizes values between 0 and 4000 (size of test set) to between 0 and 1 for colormapping 

                print(f"Confusion matrix for model {self.model_list[i][1]} using features {self.model_list[i][0]}")
                print(cmd.confusion_matrix) # The fact that we are able to access a parameter of the constructor of ConfusionMatrixDisplay as an attribute of the object is unique to this and similar classes, I believe 

                assert cmd.confusion_matrix[0,0] == np.sum((out_dict['Test set']["ID"]== 1)&(out_dict['Test set']["PredID"]== 1))
                assert cmd.confusion_matrix[0,1] == np.sum((out_dict['Test set']["ID"]== 1)&(out_dict['Test set']["PredID"]==-1))
                assert cmd.confusion_matrix[1,0] == np.sum((out_dict['Test set']["ID"]==-1)&(out_dict['Test set']["PredID"]== 1))
                assert cmd.confusion_matrix[1,1] == np.sum((out_dict['Test set']["ID"]==-1)&(out_dict['Test set']["PredID"]==-1))

                cmd.im_.set(norm=colormap_normalizer)
                if (i+1) % 4 == 0:
                    cb = fig.colorbar(mappable=ScalarMappable(cmap=cmap_to_use, norm=colormap_normalizer),
                                      ax=[ ax_dict[f'CM{j}'] for j in range(i-3, i+1) ]) # Let the colorbar "steal" space from all 4 Axes in each row (all 4 become parent Axes) to be drawn. Remember the colorbar is still drawn on its own, new (since cax=None) Axes object, not on any of the 4 parent Axes it is stealing space from (which is what ax kwarg specifies) 
                    cb.set_label("Number of events")
                
            ## Discriminator (decision function score) distributions ##
            elif option=="DiscriminatorHistogram":
                self.plotDiscriminatorHistogram(ax=ax_dict[f'CM{i}'], 
                                                y_true=  out_dict['Test set']["ID"],
                                                y_scores=out_dict['Decision scores'],
                                                nbins=100,                                          
                                                rm_outliers=(self.model_list[i][1]=="AdaBDT"))  # AdaBDT decision-function-score distribution for true antinue events seem to have few **large** outliers!
                                                                                                # To be able to see the bulk distribution, only display the "median 98%" of the y_scores. 
            # Title subplots by model and features used
            ax_dict[f'CM{i}'].set_title(f"{self.model_list[i][1]} model, using {self.model_list[i][0]}")
            
            ## ROC curves ##
            fpr, tpr, thresholds = out_dict['ROC curve'] 
            ax_dict['ROC'].plot(fpr, tpr, label=f"{self.model_list[i][1]}, using {self.model_list[i][0]} (AUC = {out_dict['ROC AUC']:.3f})", color=f'C{i}')            
            
            # By default, the model uses a threshold of 0 for predictions
                                        # False positives (predicted nue, true antinue)                               # Actual negatives (antinue)
            fpr_at_zero_thres = np.sum((out_dict['Test set']['PredID']==1)&(out_dict['Test set']['ID']==-1)) / np.sum(out_dict['Test set']['ID']==-1)
                                        # True positives (predicted nue, true nue)                                    # Actual positives (nue)
            tpr_at_zero_thres = np.sum((out_dict['Test set']['PredID']==1)&(out_dict['Test set']['ID']== 1)) / np.sum(out_dict['Test set']['ID']== 1)
            ax_dict['ROC'].plot(fpr_at_zero_thres, tpr_at_zero_thres, 'o', color=f"C{i}")

            ax_dict['ROC'].set_xlabel(r"False positive rate (Fraction of true $\bar{\nu_e}$ labelled as $\nu_e$)", fontsize="xx-large")
            ax_dict['ROC'].set_ylabel(r"True positive rate (Fraction of true $\nu_e$ labelled as $\nu_e$)",        fontsize="xx-large")
            ax_dict['ROC'].set_title(r"ROC curve, positive label taken as $\nu_e$")
            # in case you forget, the point of the r is to tell python to interpret it as a "raw" string, i.e. to treat all "\" as is and NOT as escape characters.
        
        ax_dict['ROC'].plot([0, 1], [0, 1], "--", color="black", alpha=0.4, label="Random chance classifier")
        ax_dict['ROC'].legend(fontsize="x-large")

        # self.SaveAndClose()
        fig.savefig("./plots/" + self.out_filename[:-4] + option + ".pdf")
        print("Plot created")


    def calculateRes(energyDist):
        # rms = np.sqrt(np.mean(energyDist**2))    # RMS = sqrt(E[x^2]), it is not the same as the standard deviation which is sqrt(E[x^2] - (E[x])^2)
        # return rms / np.mean(energyDist)
        return np.std(energyDist) / np.mean(energyDist)

    def calculateIqr(energyDist):
        p25, p75 = np.percentile(energyDist, [25, 75])
        return (p75 - p25) / np.mean(energyDist)

    def calculateBias(energyDist, trueE):
        return np.mean((energyDist - trueE) / trueE)
    
    def plotDiscriminatorHistogram(self, ax, y_true, y_scores, nbins=100, rm_outliers=False):
        nue_scores     = y_scores[y_true== 1]
        antinue_scores = y_scores[y_true==-1]
        
        # AdaBDT decision-function-score distribution for true antinue events seem to have few **large** outliers!
        # To be able to see the bulk distribution, only display the "median 98%" of the y_scores. 
        if rm_outliers:
            pcntile1, pcntile99 = np.quantile(y_scores, (0.01,0.99))
            nue_in_mask     = (nue_scores     >= pcntile1)&(nue_scores     <= pcntile99)
            antinue_in_mask = (antinue_scores >= pcntile1)&(antinue_scores <= pcntile99)
            ax.hist(nue_scores[nue_in_mask],         bins=nbins, histtype='step', color='blue', label=r"True $\nu_e$ events"      +f"({np.sum(nue_in_mask    )}/{len(nue_scores    )} evts)")   
            ax.hist(antinue_scores[antinue_in_mask], bins=nbins, histtype='step', color='red',  label=r"True $\bar{\nu_e}$ events"+f"({np.sum(antinue_in_mask)}/{len(antinue_scores)} evts)")
        else:
            ax.hist(nue_scores,     bins=nbins, histtype='step', color='blue', label=r"True $\nu_e$ events"      )   
            ax.hist(antinue_scores, bins=nbins, histtype='step', color='red',  label=r"True $\bar{\nu_e}$ events")
        ax.set_xlabel(f"Decision Function Score ({nbins} bins)", fontsize='large')
        ax.set_ylabel("Number of events", fontsize='large')

        ax.legend()

    def SaveAndClose(self):
        #if len(self.event_energies.keys()) > 1:
        #    plt.savefig("./plots/" + self.out_filename[:-4] + "_" + str(int(E_nu)) + "MeV.pdf")
        #else:
        plt.savefig("./plots/" + self.out_filename)
        plt.clf()
        plt.close()
        print("Plot created.")

    #---------------------------------------------------
    def QvsLPlots(self, no_light=False, no_q500=False, no_q75=False, scoring="accuracy"):
        discrNuInfo, discrAntinuInfo, contiNuInfo, contiAntinuInfo = self.events_info
        
        # Q_thresholds_str = [0, 75, 500]
        Q_thresholds_str = [75, 500]
        # L_PCEs_str       = [35, 100, 140, 180, 220]
        # L_PCEs_str       = [35, 140, 220]
        L_PCEs_str       = [180]
        Qkey = {0:0, 75:1, 500:2}
        Lkey = {35:0, 100:1, 140:2, 180:3, 220:4}

        # In this program, the columns of the 2D arrays refer to different events.
        discrNuQ, discrAntinuQ = discrNuInfo['Q'], discrAntinuInfo['Q']
        discrNuL, discrAntinuL = discrNuInfo['L'], discrAntinuInfo['L']
        contiNuQ, contiAntinuQ = contiNuInfo['Q'], contiAntinuInfo['Q']
        contiNuL, contiAntinuL = contiNuInfo['L'], contiAntinuInfo['L']

        # fig, axs = plt.subplots(len(L_PCEs_str), len(Q_thresholds_str), figsize=(6.4*len(Q_thresholds_str), 4.8*len(L_PCEs_str)))
        fig, axs = plt.subplots(1, 1, figsize=(8, 6))
        fig.set_layout_engine('tight')
        fig.suptitle(f"Trained on {discrNuQ.shape[1]}+{discrAntinuQ.shape[1]} discr. E_nu evts. Tested on {contiNuQ.shape[1]}+{contiAntinuQ.shape[1]} cont. spectrum evts",
                      y=0.98, fontsize='small')

        # for r, axrow in enumerate(axs):
        #     for c, ax in enumerate(axrow):
        for r in [0]:
            for c, ax in [(0, axs)]: 

                print(f"Starting plot at row {r} column {c}.")
                
                # Q_index = Qkey[Q_thresholds_str[c]]
                Q_75_index = Qkey[75]
                Q_500_index = Qkey[500]
                L_index = Lkey[L_PCEs_str[r]]
                thisDNuQ75, thisDAntinuQ75   = discrNuQ[Q_75_index],  discrAntinuQ[Q_75_index]
                thisDNuQ500, thisDAntinuQ500 = discrNuQ[Q_500_index], discrAntinuQ[Q_500_index]
                thisDNuL, thisDAntinuL = discrNuL[L_index], discrAntinuL[L_index]
                thisCNuQ75, thisCAntinuQ75   = contiNuQ[Q_75_index],  contiAntinuQ[Q_75_index]
                thisCNuQ500, thisCAntinuQ500 = contiNuQ[Q_500_index], contiAntinuQ[Q_500_index]
                thisCNuL, thisCAntinuL = contiNuL[L_index], contiAntinuL[L_index]

                # Create a training data set out of the discrete E_nu case. 
                df_dnue = pd.DataFrame(data= {"Q75": thisDNuQ75, "Q500": thisDNuQ500, "L": thisDNuL, 
                                            "ID": np.array( [1]*len(thisDNuQ75) )})
                df_dantinue = pd.DataFrame(data= {"Q75": thisDAntinuQ75, "Q500": thisDAntinuQ500, "L": thisDAntinuL, 
                                                "ID": np.array( [-1]*len(thisDAntinuQ75) )})
                df_dtrain = pd.concat([df_dnue, df_dantinue], axis=0, ignore_index=True)

                # Create a testing data set out of the continuous E_nu case. 
                df_cnue = pd.DataFrame(data= {"Q75": thisCNuQ75, "Q500": thisCNuQ500, "L": thisCNuL, 
                                              "ID": np.array( [1]*len(thisCNuQ75) )})
                df_cantinue = pd.DataFrame(data= {"Q75": thisCAntinuQ75, "Q500": thisCAntinuQ500, "L": thisCAntinuL, 
                                                  "ID": np.array( [-1]*len(thisCAntinuQ75)) })
                df_ctest = pd.concat([df_cnue, df_cantinue], axis=0, ignore_index=True)

                from sklearn.linear_model import LogisticRegression
                from sklearn.svm import SVC
                from sklearn.ensemble import AdaBoostClassifier
                from sklearn.ensemble import GradientBoostingClassifier
                from sklearn.tree import DecisionTreeClassifier

                X_train = df_dtrain[["Q75", "Q500", "L"]]
                X_test  = df_ctest[["Q75", "Q500", "L"]]

                if no_light:
                    X_train = X_train.drop("L", axis='columns')
                    X_test  = X_test.drop("L", axis='columns')
                    print("L excluded")
                if no_q500:
                    X_train = X_train.drop("Q500", axis='columns')
                    X_test  = X_test.drop("Q500", axis='columns')
                    print("Q500 excluded")
                if no_q75:
                    X_train = X_train.drop("Q75", axis='columns')
                    X_test  = X_test.drop("Q75", axis='columns')
                    print("Q75 excluded")
                    
                y_train = df_dtrain[["ID"]]
                y_test  = df_ctest[["ID"]]

                print(f"Fitting Model...")

                param_grid = {} # for cross validation
                if (self.model == "Logistic"):
                    model = LogisticRegression()

                if (self.model == "SVM"):
                    model = SVC()
                    param_grid = {'C':[0.01, 0.1, 1, 10, 100], 'gamma':[1e-7, 1e-6, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}

                if (self.model == "AdaBDT"):
                    model = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2))
                    # param_grid = {'n_estimators':np.arange(10, 110, 10), 'max_depth':[1, 2, 3, 4, 5]}
                    param_grid = {'n_estimators':np.arange(5, 105, 5), 'estimator':[DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=2), DecisionTreeClassifier(max_depth=3), DecisionTreeClassifier(max_depth=4), DecisionTreeClassifier(max_depth=5)]}

                if (self.model == "AdaBoostDefault"):
                    model = AdaBoostClassifier()

                if (self.model == "GradientBoostingClassifier"):
                    model = GradientBoostingClassifier()
                    param_grid = {'n_estimators':np.arange(5, 130, 5), 'learning_rate':[0.001, 0.01, 0.1, 1, 10], 'max_depth':[1, 2, 3, 4, 5, 6]}

                #### CROSS VALIDATION ####
                from sklearn.model_selection import GridSearchCV, StratifiedKFold

                print("Scoring CV using", scoring)
                cvgs = GridSearchCV(model, param_grid=param_grid, scoring=scoring, refit=True, cv=StratifiedKFold(n_splits=5, shuffle=True)) # 5-fold stratified (preserves class ratio) k fold cv

                cvgs.fit(X_train, np.ravel(y_train))

                best_parameter_model = cvgs.best_estimator_
                print(f"Best {self.model} parameters found using 5-fold CV:", cvgs.best_params_)
                print("Mean accuracy score achieved by optimized-parameter model across the five CV folds", cvgs.best_score_)
                print("Accuracy score (correct pred/ total) achieved on training set by optimized-parameter model")
                print(best_parameter_model.score(X_train, np.ravel(y_train)))

                print("Model uses features:", best_parameter_model.feature_names_in_)

                # Next, return separately-cross-validated score on training set (i.e. nested CV), it is a more reasonable estimate of the generalized (independent of specific dataset) error?
                # Idk if it is needed in this case. People can see how well it generalizes using the separate test set. 

                # Score on test set 
                y_pred_test = pd.Series(best_parameter_model.predict(X_test))
                print("Accuracy score (correct pred/ total) achieved on test set by optimized-parameter model")
                acc = best_parameter_model.score(X_test, np.ravel(y_test))

                ##########################

                ### NO VALIDATION ####
                # model.fit(X_train, np.ravel(y_train))

                # print("Accuracy score (correct pred/ total) achieved on training set")
                # print(model.score(X_train, np.ravel(y_train)))

                # y_pred_test = pd.Series(model.predict(X_test))
                # print("Accuracy score (correct pred/ total) achieved on test set")
                # acc = model.score(X_test, np.ravel(y_test))

                #######################

                print(acc)

                df_ctest["PredID"] = y_pred_test

                print(f"Finding TP, FP, TN, FN...")
                predNueTrueNuePts         = df_ctest[ (df_ctest["PredID"]== 1)&(df_ctest["ID"]== 1) ]
                predNueTrueAntinuePts     = df_ctest[ (df_ctest["PredID"]== 1)&(df_ctest["ID"]==-1) ]
                predAntinueTrueNuePts     = df_ctest[ (df_ctest["PredID"]==-1)&(df_ctest["ID"]== 1) ]
                predAntinueTrueAntinuePts = df_ctest[ (df_ctest["PredID"]==-1)&(df_ctest["ID"]==-1) ]

                print(f"Plotting...")
                
                # The points are (L, Q) ordered pairs
                ax.plot(        predNueTrueNuePts["L" if not no_light else "Q500"],         predNueTrueNuePts["Q75"], ".", alpha=0.7, color="C0", markersize=4, markeredgewidth=0, label="IDed $\\nu_e$ | true $\\nu_e$   "                         )#+f"({ (len(predNueTrueNuePts)         / (contiNuQ.shape[1]+contiAntinuQ.shape[1]))*100 : .1f}% of pts)")
                ax.plot(    predAntinueTrueNuePts["L" if not no_light else "Q500"],     predAntinueTrueNuePts["Q75"], ".", alpha=0.7, color="C2", markersize=4, markeredgewidth=0, label="IDed $\\overline{\\nu_e}$ | true $\\nu_e$   "             )#+f"({ (len(predAntinueTrueNuePts)     / (contiNuQ.shape[1]+contiAntinuQ.shape[1]))*100 : .1f}% of pts)")
                ax.plot(predAntinueTrueAntinuePts["L" if not no_light else "Q500"], predAntinueTrueAntinuePts["Q75"], ".", alpha=0.7, color="C1", markersize=4, markeredgewidth=0, label="IDed $\\overline{\\nu_e}$ | true $\\overline{\\nu_e}$   " )#+f"({ (len(predAntinueTrueAntinuePts) / (contiNuQ.shape[1]+contiAntinuQ.shape[1]))*100 : .1f}% of pts)")
                ax.plot(    predNueTrueAntinuePts["L" if not no_light else "Q500"],     predNueTrueAntinuePts["Q75"], ".", alpha=0.7, color="C3", markersize=4, markeredgewidth=0, label="IDed $\\nu_e$ | true $\\overline{\\nu_e}$   "             )#+f"({ (len(predNueTrueAntinuePts)     / (contiNuQ.shape[1]+contiAntinuQ.shape[1]))*100 : .1f}% of pts)")

                # ax.plot([], [], ' ', label=f"Accuracy: {acc*100:.2f}%   ({(1-acc)*100:.2f}% misID)")
                ax.plot([], [], ' ', label=f"{(1-acc)*100:.2f}% of events misIDed\n({acc*100:.2f}% accuracy)")
                # ax.plot([], [], ' ', label=f"$\\nu_e$ Recall = #correctly IDed $\\nu_e$ / #actual $\\nu_e$ = {(len(predNueTrueNuePts) / (len(predAntinueTrueNuePts)+len(predNueTrueNuePts)))*100:.2f}%")
                # ax.plot([], [], ' ', label=f"$\\nu_e$ Precision = #correctly IDed $\\nu_e$ / #IDed $\\nu_e$ = {(len(predNueTrueNuePts) / (len(predNueTrueAntinuePts)+len(predNueTrueNuePts)))*100:.2f}%")
                # ax.plot([], [], ' ', label= "$\\overline{\\nu_e}$ Recall = #correctly IDed $\\overline{\\nu_e}$ / #actual $\\overline{\\nu_e}$ = " + f"{(len(predAntinueTrueAntinuePts) / (len(predAntinueTrueAntinuePts)+len(predNueTrueAntinuePts)))*100:.2f}%")
                # ax.plot([], [], ' ', label= "$\\overline{\\nu_e}$ Precision = #correctly IDed $\\overline{\\nu_e}$ / #IDed $\\overline{\\nu_e}$ = " + f"{(len(predAntinueTrueAntinuePts) / (len(predAntinueTrueAntinuePts)+len(predAntinueTrueNuePts)))*100:.2f}%")
                
                nue_recall        = (len(predNueTrueNuePts) / (len(predAntinueTrueNuePts)+len(predNueTrueNuePts)))
                nue_precision     = (len(predNueTrueNuePts) / (len(predNueTrueAntinuePts)+len(predNueTrueNuePts)))
                nue_product = nue_recall * nue_precision
                nue_f1 = (2 * nue_recall * nue_precision) / (nue_recall + nue_precision)

                antinue_recall    = (len(predAntinueTrueAntinuePts) / (len(predAntinueTrueAntinuePts)+len(predNueTrueAntinuePts)))
                antinue_precision = (len(predAntinueTrueAntinuePts) / (len(predAntinueTrueAntinuePts)+len(predAntinueTrueNuePts)))
                antinue_product = antinue_recall * antinue_precision
                antinue_f1 = (2 * antinue_recall * antinue_precision) / (antinue_recall + antinue_precision)

                print(f"nue Recall        = #correctly IDed nue     / #actual nue     = {nue_recall*100:.2f}%")
                print(f"nue Precision     = #correctly IDed nue     / #IDed   nue     = {nue_precision*100:.2f}%")
                print( "antinue Recall    = #correctly IDed antinue / #actual antinue = " + f"{antinue_recall*100:.2f}%")
                print( "antinue Precision = #correctly IDed antinue / #IDed   antinue = " + f"{antinue_precision*100:.2f}%")

                print(f"nue     product of recall and precision = {nue_product*100:.2f}%")
                print(f"nue     f1 score                        = {nue_f1*100:.2f}%")
                print(f"antinue product of recall and precision = {antinue_product*100:.2f}%")
                print(f"antinue f1 score                        = {antinue_f1*100:.2f}%")

                # ax.set_xlabel(f"$L$ ({L_PCEs_str[r]} avg PEpMeV)", fontsize="large")
                ax.set_xlabel("$L$" if not no_light else "$Q_{500}$", fontsize="large")
                ax.set_ylabel(r"$Q_{75}$", fontsize="large")
                # ax.set_title(f"$~~~~Q_{{75}}$ vs $L$ ({L_PCEs_str[r]} avg PEpMeV)", fontsize="small", loc='left', y=0.85)
                # ax.set_title(r"Model Inputs: $Q_{75}$, $Q_{500}$ & $L$", fontsize="small", loc='center')
                ax.set_title(f"Model Inputs: {[feature for feature in list(X_train.columns)]}", fontsize="small", loc='center')
                
                ax.legend(fontsize="medium", markerscale=3)

                print(f"Plot at row {r} column {c} completed.")

        self.SaveAndClose()

    #---------------------------------------------------
    def QvsLPlotsVotingModel(self):
        discrNuInfo, discrAntinuInfo, contiNuInfo, contiAntinuInfo = self.events_info
        
        # Q_thresholds_str = [0, 75, 500]
        # Q_thresholds_str = [75, 500]
        # L_PCEs_str       = [35, 100, 140, 180, 220]
        # L_PCEs_str       = [35, 140, 220]
        # L_PCEs_str       = [180]
        Qkey = {0:0, 75:1, 500:2}
        Lkey = {35:0, 100:1, 140:2, 180:3, 220:4}

        # In this program, the columns of the 2D arrays refer to different events.
        discrNuQ, discrAntinuQ = discrNuInfo['Q'], discrAntinuInfo['Q']
        discrNuL, discrAntinuL = discrNuInfo['L'], discrAntinuInfo['L']
        contiNuQ, contiAntinuQ = contiNuInfo['Q'], contiAntinuInfo['Q']
        contiNuL, contiAntinuL = contiNuInfo['L'], contiAntinuInfo['L']

        # fig, axs = plt.subplots(len(L_PCEs_str), len(Q_thresholds_str), figsize=(6.4*len(Q_thresholds_str), 4.8*len(L_PCEs_str)))
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        fig.set_layout_engine('tight')
        fig.suptitle(f"Trained on {discrNuQ.shape[1]}+{discrAntinuQ.shape[1]} discr. E_nu evts. Tested on {contiNuQ.shape[1]}+{contiAntinuQ.shape[1]} cont. spectrum evts",
                      y=0.98, fontsize='small')

        # for r, axrow in enumerate(axs):
        #     for c, ax in enumerate(axrow):
        # for r in [0]:
        #     for c, ax in [(0, axs)]: 

        # print(f"Starting plot at row {r} column {c}.")
        print(f"Starting plot")
        
        # Q_index = Qkey[Q_thresholds_str[c]]
        Q_75_index = Qkey[75]
        Q_500_index = Qkey[500]
        L_index = Lkey[180]
        thisDNuQ75, thisDAntinuQ75   = discrNuQ[Q_75_index],  discrAntinuQ[Q_75_index]
        thisDNuQ500, thisDAntinuQ500 = discrNuQ[Q_500_index], discrAntinuQ[Q_500_index]
        thisDNuL, thisDAntinuL = discrNuL[L_index], discrAntinuL[L_index]
        thisCNuQ75, thisCAntinuQ75   = contiNuQ[Q_75_index],  contiAntinuQ[Q_75_index]
        thisCNuQ500, thisCAntinuQ500 = contiNuQ[Q_500_index], contiAntinuQ[Q_500_index]
        thisCNuL, thisCAntinuL = contiNuL[L_index], contiAntinuL[L_index]

        # Create a training data set out of the discrete E_nu case. 
        df_dnue = pd.DataFrame(data= {"Q75": thisDNuQ75, "Q500": thisDNuQ500, "L": thisDNuL, 
                                    "ID": np.array( [1]*len(thisDNuQ75) )})
        df_dantinue = pd.DataFrame(data= {"Q75": thisDAntinuQ75, "Q500": thisDAntinuQ500, "L": thisDAntinuL, 
                                        "ID": np.array( [-1]*len(thisDAntinuQ75) )})
        df_dtrain = pd.concat([df_dnue, df_dantinue], axis=0, ignore_index=True)

        # Create a testing data set out of the continuous E_nu case. 
        df_cnue = pd.DataFrame(data= {"Q75": thisCNuQ75, "Q500": thisCNuQ500, "L": thisCNuL, 
                                        "ID": np.array( [1]*len(thisCNuQ75) )})
        df_cantinue = pd.DataFrame(data= {"Q75": thisCAntinuQ75, "Q500": thisCAntinuQ500, "L": thisCAntinuL, 
                                            "ID": np.array( [-1]*len(thisCAntinuQ75)) })
        df_ctest = pd.concat([df_cnue, df_cantinue], axis=0, ignore_index=True)

        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.tree import DecisionTreeClassifier

        X_train = df_dtrain[["Q75", "Q500", "L"]]
        X_test  = df_ctest[["Q75", "Q500", "L"]]

        y_train = df_dtrain[["ID"]]
        y_test  = df_ctest[["ID"]]

        print(f"Fitting Model...")

        param_grid = {} # for cross validation
        if (self.model == "Logistic"):
            model = LogisticRegression()

        if (self.model == "SVM"):
            model = SVC()
            param_grid = {'C':[0.01, 0.1, 1, 10, 100], 'gamma':[1e-7, 1e-6, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}

        if (self.model == "AdaBDT"):
            model = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2))
            param_grid = {'n_estimators':np.arange(5, 105, 5), 'estimator':[DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=2), DecisionTreeClassifier(max_depth=3), DecisionTreeClassifier(max_depth=4), DecisionTreeClassifier(max_depth=5)]}

        if (self.model == "AdaBoostDefault"):
            model = AdaBoostClassifier()

        if (self.model == "GradientBoostingClassifier"):
            model = GradientBoostingClassifier()
            param_grid = {'n_estimators':np.arange(5, 130, 5), 'learning_rate':[0.001, 0.01, 0.1, 1, 10], 'max_depth':[1, 2, 3, 4, 5, 6]}

        #### CROSS VALIDATION ####
        from sklearn.model_selection import GridSearchCV, StratifiedKFold

        cvgs = GridSearchCV(model, param_grid=param_grid, scoring='accuracy', refit=True, cv=StratifiedKFold(n_splits=5)) # 5-fold stratified (preserves class ratio) k fold cv
        # # cvgs = GridSearchCV(model, param_grid=param_grid, scoring='f1',     refit=True, cv=StratifiedKFold(n_splits=5)) # 5-fold stratified (preserves class ratio) k fold cv

        from sklearn.pipeline import Pipeline
        model_noL    = Pipeline( [ ('feature_drop', ManualFeatureSelector(features=['Q75', 'Q500'])), ('model', cvgs) ] )
        model_noQ500 = Pipeline( [ ('feature_drop', ManualFeatureSelector(features=['Q75', 'L'   ])), ('model', cvgs) ] )
        model_noQ75  = Pipeline( [ ('feature_drop', ManualFeatureSelector(features=['Q500', 'L'  ])), ('model', cvgs) ] )

        from sklearn.ensemble import VotingClassifier
        voting_model = VotingClassifier(estimators=[('noL',    model_noL), 
                                                    ('noQ500', model_noQ500),
                                                    ('noQ75',  model_noQ75)], voting='soft' if self.model != "SVM" else 'hard')


        voting_model.fit(X_train, np.ravel(y_train))
        print([classifier_pipeline.named_steps['model'].feature_names_in_ for classifier_pipeline in voting_model.estimators_])
        print(f"Best {self.model} parameters found using 5-fold CV:", [classifier_pipeline.named_steps['model'].best_params_ for classifier_pipeline in voting_model.estimators_])
        # print("Mean accuracy score achieved by optimized-parameter model across the five CV folds", cvgs.best_score_)
        print("Accuracy score (correct pred/ total) achieved on training set by optimized-parameter model")
        print(voting_model.score(X_train, np.ravel(y_train)))

        # Next, return separately-cross-validated score on training set (i.e. nested CV), it is a more reasonable estimate of the generalized (independent of specific dataset) error?
        # Idk if it is needed in this case. People can see how well it generalizes using the separate test set. 

        # # Score on test set 
        y_pred_test = pd.Series(voting_model.predict(X_test))
        print("Accuracy score (correct pred/ total) achieved on test set by optimized-parameter model")
        acc = voting_model.score(X_test, np.ravel(y_test))

        ##########################

        ### NO VALIDATION ####
        # from sklearn.pipeline import Pipeline
        # model_noL    = Pipeline( [ ('feature_drop', ManualFeatureSelector(features=['Q75', 'Q500'])), ('model', model) ] )
        # model_noQ500 = Pipeline( [ ('feature_drop', ManualFeatureSelector(features=['Q75', 'L'   ])), ('model', model) ] )
        # model_noQ75  = Pipeline( [ ('feature_drop', ManualFeatureSelector(features=['Q500', 'L'  ])), ('model', model) ] )

        # from sklearn.ensemble import VotingClassifier
        # voting_model = VotingClassifier(estimators=[('noL',    model_noL), 
        #                                             ('noQ500', model_noQ500),
        #                                             ('noQ75',  model_noQ75)], voting='soft' if self.model != "SVM" else 'hard')
        # # it seems probability estimation using SVM is actually computationally expensive
        # # It involves fitting a logistic regression model on the SVM scores (X) achieved by and class of (y) each data point
        # # (SVM score is the signed distance from hyperplane) which is then 5-fold cross-validated. 

        # voting_model.fit(X_train, np.ravel(y_train))
        # print("Models: ", voting_model.estimators_)
        # print([classifier_pipeline.named_steps['model'].feature_names_in_ for classifier_pipeline in voting_model.estimators_])

        # print("Accuracy score (correct pred/ total) achieved on training set")
        # print(voting_model.score(X_train, np.ravel(y_train)))

        # y_pred_test = pd.Series(voting_model.predict(X_test))
        # print("Accuracy score (correct pred/ total) achieved on test set")
        # acc = voting_model.score(X_test, np.ravel(y_test))

        #######################

        print(acc)

        df_ctest["PredID"] = y_pred_test

        print(f"Finding TP, FP, TN, FN...")
        predNueTrueNuePts         = df_ctest[ (df_ctest["PredID"]== 1)&(df_ctest["ID"]== 1) ]
        predNueTrueAntinuePts     = df_ctest[ (df_ctest["PredID"]== 1)&(df_ctest["ID"]==-1) ]
        predAntinueTrueNuePts     = df_ctest[ (df_ctest["PredID"]==-1)&(df_ctest["ID"]== 1) ]
        predAntinueTrueAntinuePts = df_ctest[ (df_ctest["PredID"]==-1)&(df_ctest["ID"]==-1) ]

        print(f"Plotting...")
        
        # The points are (L, Q) ordered pairs
        ax.plot(        predNueTrueNuePts["L"],         predNueTrueNuePts["Q75"], ".", alpha=0.7, color="C0", markersize=4, markeredgewidth=0, label="IDed $\\nu_e$ | true $\\nu_e$   "                         )#+f"({ (len(predNueTrueNuePts)         / (contiNuQ.shape[1]+contiAntinuQ.shape[1]))*100 : .1f}% of pts)")
        ax.plot(    predAntinueTrueNuePts["L"],     predAntinueTrueNuePts["Q75"], ".", alpha=0.7, color="C2", markersize=4, markeredgewidth=0, label="IDed $\\overline{\\nu_e}$ | true $\\nu_e$   "             )#+f"({ (len(predAntinueTrueNuePts)     / (contiNuQ.shape[1]+contiAntinuQ.shape[1]))*100 : .1f}% of pts)")
        ax.plot(predAntinueTrueAntinuePts["L"], predAntinueTrueAntinuePts["Q75"], ".", alpha=0.7, color="C1", markersize=4, markeredgewidth=0, label="IDed $\\overline{\\nu_e}$ | true $\\overline{\\nu_e}$   " )#+f"({ (len(predAntinueTrueAntinuePts) / (contiNuQ.shape[1]+contiAntinuQ.shape[1]))*100 : .1f}% of pts)")
        ax.plot(    predNueTrueAntinuePts["L"],     predNueTrueAntinuePts["Q75"], ".", alpha=0.7, color="C3", markersize=4, markeredgewidth=0, label="IDed $\\nu_e$ | true $\\overline{\\nu_e}$   "             )#+f"({ (len(predNueTrueAntinuePts)     / (contiNuQ.shape[1]+contiAntinuQ.shape[1]))*100 : .1f}% of pts)")

        # ax.plot([], [], ' ', label=f"Accuracy: {acc*100:.2f}%   ({(1-acc)*100:.2f}% misID)")
        ax.plot([], [], ' ', label=f"{(1-acc)*100:.2f}% of events misIDed\n({acc*100:.2f}% accuracy)")
        # ax.plot([], [], ' ', label=f"$\\nu_e$ Recall = #correctly IDed $\\nu_e$ / #actual $\\nu_e$ = {(len(predNueTrueNuePts) / (len(predAntinueTrueNuePts)+len(predNueTrueNuePts)))*100:.2f}%")
        # ax.plot([], [], ' ', label=f"$\\nu_e$ Precision = #correctly IDed $\\nu_e$ / #IDed $\\nu_e$ = {(len(predNueTrueNuePts) / (len(predNueTrueAntinuePts)+len(predNueTrueNuePts)))*100:.2f}%")
        # ax.plot([], [], ' ', label= "$\\overline{\\nu_e}$ Recall = #correctly IDed $\\overline{\\nu_e}$ / #actual $\\overline{\\nu_e}$ = " + f"{(len(predAntinueTrueAntinuePts) / (len(predAntinueTrueAntinuePts)+len(predNueTrueAntinuePts)))*100:.2f}%")
        # ax.plot([], [], ' ', label= "$\\overline{\\nu_e}$ Precision = #correctly IDed $\\overline{\\nu_e}$ / #IDed $\\overline{\\nu_e}$ = " + f"{(len(predAntinueTrueAntinuePts) / (len(predAntinueTrueAntinuePts)+len(predAntinueTrueNuePts)))*100:.2f}%")
        
        nue_recall        = (len(predNueTrueNuePts) / (len(predAntinueTrueNuePts)+len(predNueTrueNuePts)))
        nue_precision     = (len(predNueTrueNuePts) / (len(predNueTrueAntinuePts)+len(predNueTrueNuePts)))
        nue_product = nue_recall * nue_precision
        nue_f1 = (2 * nue_recall * nue_precision) / (nue_recall + nue_precision)

        antinue_recall    = (len(predAntinueTrueAntinuePts) / (len(predAntinueTrueAntinuePts)+len(predNueTrueAntinuePts)))
        antinue_precision = (len(predAntinueTrueAntinuePts) / (len(predAntinueTrueAntinuePts)+len(predAntinueTrueNuePts)))
        antinue_product = antinue_recall * antinue_precision
        antinue_f1 = (2 * antinue_recall * antinue_precision) / (antinue_recall + antinue_precision)

        print(f"nue Recall        = #correctly IDed nue     / #actual nue     = {nue_recall*100:.2f}%")
        print(f"nue Precision     = #correctly IDed nue     / #IDed   nue     = {nue_precision*100:.2f}%")
        print( "antinue Recall    = #correctly IDed antinue / #actual antinue = " + f"{antinue_recall*100:.2f}%")
        print( "antinue Precision = #correctly IDed antinue / #IDed   antinue = " + f"{antinue_precision*100:.2f}%")

        print(f"nue     product of recall and precision = {nue_product*100:.2f}%")
        print(f"nue     f1 score                        = {nue_f1*100:.2f}%")
        print(f"antinue product of recall and precision = {antinue_product*100:.2f}%")
        print(f"antinue f1 score                        = {antinue_f1*100:.2f}%")

        # ax.set_xlabel(f"$L$ ({L_PCEs_str[r]} avg PEpMeV)", fontsize="large")
        ax.set_xlabel("$L$", fontsize="large")
        ax.set_ylabel(r"$Q_{75}$", fontsize="large")
        # ax.set_title(f"$~~~~Q_{{75}}$ vs $L$ ({L_PCEs_str[r]} avg PEpMeV)", fontsize="small", loc='left', y=0.85)
        # ax.set_title(r"Model Inputs: $Q_{75}$, $Q_{500}$ & $L$", fontsize="small", loc='center')
        ax.set_title(f"Model Inputs: {[feature for feature in list(X_train.columns)]}", fontsize="small", loc='center')
        
        ax.legend(fontsize="medium", markerscale=3)

        # print(f"Plot at row {r} column {c} completed.")
        print(f"Plot completed.")

        self.SaveAndClose()

    def QvsLPlotsLightThreshold(self):
        discrNuInfo, discrAntinuInfo, contiNuInfo, contiAntinuInfo = self.events_info
        
        Qkey = {0:0, 75:1, 500:2}
        # Lkey = {35:0, 100:1, 140:2, 180:3, 220:4}
        L_threkey = {0:0, 75:1, 500:2}

        # In this program, the columns of the 2D arrays refer to different events.
        discrNuQ, discrAntinuQ = discrNuInfo['Q'], discrAntinuInfo['Q']
        discrNuL, discrAntinuL = discrNuInfo['L_LY180_thre'], discrAntinuInfo['L_LY180_thre']
        # contiNuQ, contiAntinuQ = contiNuInfo['Q'], contiAntinuInfo['Q']
        # contiNuL, contiAntinuL = contiNuInfo['L'], contiAntinuInfo['L']

        # fig, axs = plt.subplots(len(L_PCEs_str), len(Q_thresholds_str), figsize=(6.4*len(Q_thresholds_str), 4.8*len(L_PCEs_str)))
        fig, axs = plt.subplots(1, 1, figsize=(8, 6))
        fig.set_layout_engine('tight')
        # fig.suptitle(f"Trained on {discrNuQ.shape[1]}+{discrAntinuQ.shape[1]} discr. E_nu evts. Tested on {contiNuQ.shape[1]}+{contiAntinuQ.shape[1]} cont. spectrum evts",
        #               y=0.98, fontsize='small')
        fig.suptitle(f"Trained, validated and tested on {discrNuQ.shape[1]}+{discrAntinuQ.shape[1]} discr. E_nu evts.",
                      y=0.98, fontsize='small')

        # for r, axrow in enumerate(axs):
        #     for c, ax in enumerate(axrow):                
        Q_index, Q_75_index, Q_500_index =      Qkey[0],      Qkey[75],      Qkey[500]
        L_index, L_75_index, L_500_index = L_threkey[0], L_threkey[75], L_threkey[500]

        thisDNuL75,  thisDAntinuL75   = discrNuL[L_75_index],  discrAntinuL[L_75_index]
        thisDNuL500, thisDAntinuL500  = discrNuL[L_500_index], discrAntinuL[L_500_index]
        thisDNuQ, thisDAntinuQ        = discrNuQ[Q_index],     discrAntinuQ[Q_index]

        thisDNuQ75,  thisDAntinuQ75   = discrNuQ[Q_75_index],  discrAntinuQ[Q_75_index]
        thisDNuQ500, thisDAntinuQ500  = discrNuQ[Q_500_index], discrAntinuQ[Q_500_index]
        thisDNuL, thisDAntinuL        = discrNuL[L_index],     discrAntinuL[L_index]


        # Create a training data set out of the discrete E_nu case. 
        df_dnue = pd.DataFrame(data= {"L75": thisDNuL75, "L500": thisDNuL500, "Q": thisDNuQ, 
                                      "Q75": thisDNuQ75, "Q500": thisDNuQ500, "L": thisDNuL,
                                      "ID": np.array( [1]*len(thisDNuL75) )})
        df_dantinue = pd.DataFrame(data= {"L75": thisDAntinuL75, "L500": thisDAntinuL500, "Q": thisDAntinuQ, 
                                          "Q75": thisDAntinuQ75, "Q500": thisDAntinuQ500, "L": thisDAntinuL,
                                          "ID": np.array( [-1]*len(thisDAntinuL75) )})
        df_dtrain = pd.concat([df_dnue, df_dantinue], axis=0, ignore_index=True)

        # Create a testing data set out of the continuous E_nu case. 

        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.tree import DecisionTreeClassifier

        X_train = df_dtrain[["L75", "L500", "Q", "Q75", "Q500", "L"]]
        # X_test  = df_ctest[["Q75", "Q500", "L"]]
            
        y_train = df_dtrain[["ID"]]
        # y_test  = df_ctest[["ID"]]

        # Sanity check
        print( (X_train["L75"][:10000] == self.events_info[0]["L_LY180_thre"][1]).all() )
        print( (X_train["L75"][10000:] == self.events_info[1]["L_LY180_thre"][1]).all() )
        print( (X_train["L500"][:10000] == self.events_info[0]["L_LY180_thre"][2]).all() )
        print( (X_train["L500"][10000:] == self.events_info[1]["L_LY180_thre"][2]).all() )
        print( (X_train["Q"][:10000] == self.events_info[0]["Q"][0]).all() )
        print( (X_train["Q"][10000:] == self.events_info[1]["Q"][0]).all() )

        print( (X_train["Q75"][:10000] == self.events_info[0]["Q"][1]).all() )
        print( (X_train["Q75"][10000:] == self.events_info[1]["Q"][1]).all() )
        print( (X_train["Q500"][:10000] == self.events_info[0]["Q"][2]).all() )
        print( (X_train["Q500"][10000:] == self.events_info[1]["Q"][2]).all() )
        print( (X_train["L"][:10000] == self.events_info[0]["L_LY180_thre"][0]).all() )
        print( (X_train["L"][10000:] == self.events_info[1]["L_LY180_thre"][0]).all() )


        print(f"Fitting Model...")

        param_grid = {} # for cross validation
        if (self.model == "Logistic"):
            model = LogisticRegression()

        if (self.model == "SVM"):
            model = SVC()
            param_grid = {'C':[0.01, 0.1, 1, 10, 100], 'gamma':[1e-7, 1e-6, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}

        if (self.model == "AdaBDT"):
            model = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2))
            # param_grid = {'n_estimators':np.arange(10, 110, 10), 'max_depth':[1, 2, 3, 4, 5]}
            param_grid = {'n_estimators':np.arange(5, 105, 5), 'estimator':[DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=2), DecisionTreeClassifier(max_depth=3), DecisionTreeClassifier(max_depth=4), DecisionTreeClassifier(max_depth=5)]}

        if (self.model == "AdaBoostDefault"):
            model = AdaBoostClassifier()

        if (self.model == "GradientBoostingClassifier"):
            model = GradientBoostingClassifier()
            param_grid = {'n_estimators':np.arange(5, 130, 5), 'learning_rate':[0.001, 0.01, 0.1, 1, 10], 'max_depth':[1, 2, 3, 4, 5, 6]}

        #### CROSS VALIDATION ####
        # from sklearn.model_selection import GridSearchCV, StratifiedKFold

        # cvgs = GridSearchCV(model, param_grid=param_grid, scoring="accuracy", refit=True, cv=StratifiedKFold(n_splits=5, shuffle=True)) # 5-fold stratified (preserves class ratio) k fold cv

        # cvgs.fit(X_train, np.ravel(y_train))

        # best_parameter_model = cvgs.best_estimator_
        # print(f"Best {self.model} parameters found using 5-fold CV:", cvgs.best_params_)
        # print("Mean accuracy score achieved by optimized-parameter model across the five CV folds", cvgs.best_score_)
        # print("Accuracy score (correct pred/ total) achieved on training set by optimized-parameter model")
        # print(best_parameter_model.score(X_train, np.ravel(y_train)))

        # # Next, return separately-cross-validated score on training set (i.e. nested CV), it is a more reasonable estimate of the generalized (independent of specific dataset) error?
        # # Idk if it is needed in this case. People can see how well it generalizes using the separate test set. 

        # # Score on test set 
        # y_pred_test = pd.Series(best_parameter_model.predict(X_test))
        # print("Accuracy score (correct pred/ total) achieved on test set by optimized-parameter model")
        # acc = best_parameter_model.score(X_test, np.ravel(y_test))

        ##########################

        ### NO TEST SET ####
        from sklearn.model_selection import cross_validate
        from sklearn.metrics import accuracy_score, precision_score, recall_score, make_scorer
        from sklearn.model_selection import StratifiedKFold

        drop_list = [["Q75", "Q500", "L"], ["Q75", "Q500", "L", "Q"], ["Q75", "Q500", "L", "L500"], ["Q75", "Q500", "L", "L75"],
                     ["L75", "L500", "Q"], ["L75", "L500", "Q", "L"], ["L75", "L500", "Q", "Q500"], ["L75", "L500", "Q", "Q75"]]

        for drops in drop_list:
            X_train_proj = X_train.drop(drops, axis='columns')

            cv_results = cross_validate(model, X_train_proj, np.ravel(y_train), 
                                        scoring={"accuracy": make_scorer(accuracy_score), 
                                            "nue_precision":     make_scorer(lambda y, y_pred: precision_score(y, y_pred, average=None, labels=(1, -1))[0]),
                                            "antinue_precision": make_scorer(lambda y, y_pred: precision_score(y, y_pred, average=None, labels=(1, -1))[1]),
                                            "nue_recall":        make_scorer(lambda y, y_pred: recall_score(   y, y_pred, average=None, labels=(1, -1))[0]),
                                            "antinue_recall":    make_scorer(lambda y, y_pred: recall_score(   y, y_pred, average=None, labels=(1, -1))[1])},
                                        cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42))  # shuffle=True is crucial to make sure the cross-validation splitter is actually shuffling the training data set before partitioning it up into folds, one of which is used as test data (and the rest as training) each iteration   
            
            model.fit(X_train_proj, np.ravel(y_train))
            print("Model uses features:", model.feature_names_in_)

            print("Accuracy scores (correct pred/ total) achieved on diff splits of discrete evt sample")
            print(cv_results["test_accuracy"])
            print("Mean accuracy")
            print(cv_results["test_accuracy"].mean(), "+-", cv_results["test_accuracy"].std())
            print(f"{cv_results['test_accuracy'].mean()*100:.2f}%")
            print()
            
            print("nue Precision scores (correct pred nue/ total pred nue) achieved on diff splits of discrete evt sample")
            print(cv_results["test_nue_precision"])
            print("Mean nue_precision")
            print(cv_results["test_nue_precision"].mean(), "+-", cv_results["test_nue_precision"].std())
            print(f"{cv_results['test_nue_precision'].mean()*100:.2f}%")
            print()
            
            print("antinue Precision scores (correct pred antinue/ total pred antinue) achieved on diff splits of discrete evt sample")
            print(cv_results["test_antinue_precision"])
            print("Mean antinue_precision")
            print(cv_results["test_antinue_precision"].mean(), "+-", cv_results["test_antinue_precision"].std())
            print(f"{cv_results['test_antinue_precision'].mean()*100:.2f}%")
            print()
            
            print("nue Recall scores (correct pred nue/ total actual nue) achieved on diff splits of discrete evt sample")
            print(cv_results["test_nue_recall"])
            print("Mean nue_recall")
            print(cv_results["test_nue_recall"].mean(), "+-", cv_results["test_nue_recall"].std())
            print(f"{cv_results['test_nue_recall'].mean()*100:.2f}%")
            print()
            
            print("antinue Recall scores (correct pred antinue/ total actual antinue) achieved on diff splits of discrete evt sample")
            print(cv_results["test_antinue_recall"])
            print("Mean antinue_recall")
            print(cv_results["test_antinue_recall"].mean(), "+-", cv_results["test_antinue_recall"].std())
            print(f"{cv_results['test_antinue_recall'].mean()*100:.2f}%")
            print("\n\n")
        
        #######################

        # print(acc)

        # df_ctest["PredID"] = y_pred_test

        # print(f"Finding TP, FP, TN, FN...")
        # predNueTrueNuePts         = df_ctest[ (df_ctest["PredID"]== 1)&(df_ctest["ID"]== 1) ]
        # predNueTrueAntinuePts     = df_ctest[ (df_ctest["PredID"]== 1)&(df_ctest["ID"]==-1) ]
        # predAntinueTrueNuePts     = df_ctest[ (df_ctest["PredID"]==-1)&(df_ctest["ID"]== 1) ]
        # predAntinueTrueAntinuePts = df_ctest[ (df_ctest["PredID"]==-1)&(df_ctest["ID"]==-1) ]

        # print(f"Plotting...")
        
        # # The points are (L, Q) ordered pairs
        # ax.plot(        predNueTrueNuePts["L" if not no_light else "Q500"],         predNueTrueNuePts["Q75"], ".", alpha=0.7, color="C0", markersize=4, markeredgewidth=0, label="IDed $\\nu_e$ | true $\\nu_e$   "                         )#+f"({ (len(predNueTrueNuePts)         / (contiNuQ.shape[1]+contiAntinuQ.shape[1]))*100 : .1f}% of pts)")
        # ax.plot(    predAntinueTrueNuePts["L" if not no_light else "Q500"],     predAntinueTrueNuePts["Q75"], ".", alpha=0.7, color="C2", markersize=4, markeredgewidth=0, label="IDed $\\overline{\\nu_e}$ | true $\\nu_e$   "             )#+f"({ (len(predAntinueTrueNuePts)     / (contiNuQ.shape[1]+contiAntinuQ.shape[1]))*100 : .1f}% of pts)")
        # ax.plot(predAntinueTrueAntinuePts["L" if not no_light else "Q500"], predAntinueTrueAntinuePts["Q75"], ".", alpha=0.7, color="C1", markersize=4, markeredgewidth=0, label="IDed $\\overline{\\nu_e}$ | true $\\overline{\\nu_e}$   " )#+f"({ (len(predAntinueTrueAntinuePts) / (contiNuQ.shape[1]+contiAntinuQ.shape[1]))*100 : .1f}% of pts)")
        # ax.plot(    predNueTrueAntinuePts["L" if not no_light else "Q500"],     predNueTrueAntinuePts["Q75"], ".", alpha=0.7, color="C3", markersize=4, markeredgewidth=0, label="IDed $\\nu_e$ | true $\\overline{\\nu_e}$   "             )#+f"({ (len(predNueTrueAntinuePts)     / (contiNuQ.shape[1]+contiAntinuQ.shape[1]))*100 : .1f}% of pts)")

        # # ax.plot([], [], ' ', label=f"Accuracy: {acc*100:.2f}%   ({(1-acc)*100:.2f}% misID)")
        # ax.plot([], [], ' ', label=f"{(1-acc)*100:.2f}% of events misIDed\n({acc*100:.2f}% accuracy)")
        # # ax.plot([], [], ' ', label=f"$\\nu_e$ Recall = #correctly IDed $\\nu_e$ / #actual $\\nu_e$ = {(len(predNueTrueNuePts) / (len(predAntinueTrueNuePts)+len(predNueTrueNuePts)))*100:.2f}%")
        # # ax.plot([], [], ' ', label=f"$\\nu_e$ Precision = #correctly IDed $\\nu_e$ / #IDed $\\nu_e$ = {(len(predNueTrueNuePts) / (len(predNueTrueAntinuePts)+len(predNueTrueNuePts)))*100:.2f}%")
        # # ax.plot([], [], ' ', label= "$\\overline{\\nu_e}$ Recall = #correctly IDed $\\overline{\\nu_e}$ / #actual $\\overline{\\nu_e}$ = " + f"{(len(predAntinueTrueAntinuePts) / (len(predAntinueTrueAntinuePts)+len(predNueTrueAntinuePts)))*100:.2f}%")
        # # ax.plot([], [], ' ', label= "$\\overline{\\nu_e}$ Precision = #correctly IDed $\\overline{\\nu_e}$ / #IDed $\\overline{\\nu_e}$ = " + f"{(len(predAntinueTrueAntinuePts) / (len(predAntinueTrueAntinuePts)+len(predAntinueTrueNuePts)))*100:.2f}%")
        
        # nue_recall        = (len(predNueTrueNuePts) / (len(predAntinueTrueNuePts)+len(predNueTrueNuePts)))
        # nue_precision     = (len(predNueTrueNuePts) / (len(predNueTrueAntinuePts)+len(predNueTrueNuePts)))
        # nue_product = nue_recall * nue_precision
        # nue_f1 = (2 * nue_recall * nue_precision) / (nue_recall + nue_precision)

        # antinue_recall    = (len(predAntinueTrueAntinuePts) / (len(predAntinueTrueAntinuePts)+len(predNueTrueAntinuePts)))
        # antinue_precision = (len(predAntinueTrueAntinuePts) / (len(predAntinueTrueAntinuePts)+len(predAntinueTrueNuePts)))
        # antinue_product = antinue_recall * antinue_precision
        # antinue_f1 = (2 * antinue_recall * antinue_precision) / (antinue_recall + antinue_precision)

        # print(f"nue Recall        = #correctly IDed nue     / #actual nue     = {nue_recall*100:.2f}%")
        # print(f"nue Precision     = #correctly IDed nue     / #IDed   nue     = {nue_precision*100:.2f}%")
        # print( "antinue Recall    = #correctly IDed antinue / #actual antinue = " + f"{antinue_recall*100:.2f}%")
        # print( "antinue Precision = #correctly IDed antinue / #IDed   antinue = " + f"{antinue_precision*100:.2f}%")

        # print(f"nue     product of recall and precision = {nue_product*100:.2f}%")
        # print(f"nue     f1 score                        = {nue_f1*100:.2f}%")
        # print(f"antinue product of recall and precision = {antinue_product*100:.2f}%")
        # print(f"antinue f1 score                        = {antinue_f1*100:.2f}%")

        # # ax.set_xlabel(f"$L$ ({L_PCEs_str[r]} avg PEpMeV)", fontsize="large")
        # ax.set_xlabel("$L$" if not no_light else "$Q_{500}$", fontsize="large")
        # ax.set_ylabel(r"$Q_{75}$", fontsize="large")
        # # ax.set_title(f"$~~~~Q_{{75}}$ vs $L$ ({L_PCEs_str[r]} avg PEpMeV)", fontsize="small", loc='left', y=0.85)
        # # ax.set_title(r"Model Inputs: $Q_{75}$, $Q_{500}$ & $L$", fontsize="small", loc='center')
        # ax.set_title(f"Model Inputs: {[feature for feature in list(X_train.columns)]}", fontsize="small", loc='center')
        
        # ax.legend(fontsize="medium", markerscale=3)

        # print(f"Plot completed.")

        # self.SaveAndClose()

    def calcROCDiscreteE_nu(self, feature_str="Q75 Q500 L", model_nm="Logistic"):
        discrNuInfo, discrAntinuInfo, contiNuInfo, contiAntinuInfo = self.events_info

        Qkey      = {0:0, 75:1, 500:2}
        L_threkey = {0:0, 75:1, 500:2}

        # In this program, the columns of the 2D arrays refer to different events.
        discrNuQ, discrAntinuQ = discrNuInfo['Q'],            discrAntinuInfo['Q']
        discrNuL, discrAntinuL = discrNuInfo['L_LY180_thre'], discrAntinuInfo['L_LY180_thre']

        discrNuE_nu, discrAntinuE_nu = discrNuInfo['E_nu'], discrAntinuInfo['E_nu']

        # for r, axrow in enumerate(axs):
        #     for c, ax in enumerate(axrow):                
        Q_index, Q_75_index, Q_500_index =      Qkey[0],      Qkey[75],      Qkey[500]
        L_index, L_75_index, L_500_index = L_threkey[0], L_threkey[75], L_threkey[500]

        thisDNuL75,  thisDAntinuL75   = discrNuL[L_75_index],  discrAntinuL[L_75_index]
        thisDNuL500, thisDAntinuL500  = discrNuL[L_500_index], discrAntinuL[L_500_index]
        thisDNuQ,    thisDAntinuQ     = discrNuQ[Q_index],     discrAntinuQ[Q_index]

        thisDNuQ75,  thisDAntinuQ75   = discrNuQ[Q_75_index],  discrAntinuQ[Q_75_index]
        thisDNuQ500, thisDAntinuQ500  = discrNuQ[Q_500_index], discrAntinuQ[Q_500_index]
        thisDNuL,    thisDAntinuL     = discrNuL[L_index],     discrAntinuL[L_index]

        ## Create a training data set out of the discrete E_nu case. ##
        df_dnue = pd.DataFrame(data= {"L75": thisDNuL75, "L500": thisDNuL500, "Q": thisDNuQ, 
                                      "Q75": thisDNuQ75, "Q500": thisDNuQ500, "L": thisDNuL,
                                      "E_nu": discrNuE_nu,
                                      "ID": np.array( [1]*len(thisDNuL75) )})
        df_dantinue = pd.DataFrame(data= {"L75": thisDAntinuL75, "L500": thisDAntinuL500, "Q": thisDAntinuQ, 
                                          "Q75": thisDAntinuQ75, "Q500": thisDAntinuQ500, "L": thisDAntinuL,
                                          "E_nu": discrAntinuE_nu,
                                          "ID": np.array( [-1]*len(thisDAntinuL75) )})
        df_dall = pd.concat([df_dnue, df_dantinue], axis=0, ignore_index=True)

        # Sanity check
        assert (df_dall["L75"][:10000]  == self.events_info[0]["L_LY180_thre"][1]).all() 
        assert (df_dall["L75"][10000:]  == self.events_info[1]["L_LY180_thre"][1]).all() 
        assert (df_dall["L500"][:10000] == self.events_info[0]["L_LY180_thre"][2]).all() 
        assert (df_dall["L500"][10000:] == self.events_info[1]["L_LY180_thre"][2]).all() 
        assert (df_dall["Q"][:10000]    == self.events_info[0]["Q"][0]).all() 
        assert (df_dall["Q"][10000:]    == self.events_info[1]["Q"][0]).all() 

        assert (df_dall["Q75"][:10000]  == self.events_info[0]["Q"][1]).all()
        assert (df_dall["Q75"][10000:]  == self.events_info[1]["Q"][1]).all()
        assert (df_dall["Q500"][:10000] == self.events_info[0]["Q"][2]).all()
        assert (df_dall["Q500"][10000:] == self.events_info[1]["Q"][2]).all()
        assert (df_dall["L"][:10000]    == self.events_info[0]["L_LY180_thre"][0]).all()
        assert (df_dall["L"][10000:]    == self.events_info[1]["L_LY180_thre"][0]).all()

        assert (df_dall["E_nu"][:10000] == self.events_info[0]["E_nu"]).all()
        assert (df_dall["E_nu"][10000:] == self.events_info[1]["E_nu"]).all()
        assert (df_dall["ID"][:10000] ==  1).all()      # The above tests have "proven" that df_dall[:10000] are the nu events i.e self.events_info[0] and df_dall[10000:] are the antinu events i.e self.events_info[1] 
        assert (df_dall["ID"][10000:] == -1).all()      # so these lines are showing that the class labels were generated correctly.

        ## Create a testing data set with even E_nu distribution ##
        from sklearn.model_selection import train_test_split                                           
        df_dtrain, df_dtest = train_test_split( df_dall, test_size=0.2, random_state=42, shuffle=True, stratify=(df_dall["E_nu"]*df_dall["ID"]) ) # Since we want even distribution of E_nu and charge, treat the value of E_nu*charge (100, 200,...,1000 for nue, -100, -200,...,-1000 for antinue) as a "class label", and specify that we want equal number of rows having each unique "class label" value in both train and test partitions.  

        print("\ndf_dtest:")
        print(f"{'E_nu':<7}{'Count':<7}")
        test_enu_dist = np.unique(df_dtest["E_nu"], return_counts=True)
        for enu, ct in zip( test_enu_dist[0], test_enu_dist[1] ):
            print(f"{enu:<7}{ct:<7}")
        print(f"\n{'ID':<7}{'Count':<7}")
        print(f"{ 1:<7}{np.sum(df_dtest['ID']== 1):<7}")
        print(f"{-1:<7}{np.sum(df_dtest['ID']==-1):<7}")

        print("\ndf_dtrain:")
        print(f"{'E_nu':<7}{'Count':<7}")
        train_enu_dist = np.unique(df_dtrain["E_nu"], return_counts=True)
        for enu, ct in zip( train_enu_dist[0], train_enu_dist[1] ):
            print(f"{enu:<7}{ct:<7}")
        print(f"\n{'ID':<7}{'Count':<7}")
        print(f"{ 1:<7}{np.sum(df_dtrain['ID']== 1):<7}")
        print(f"{-1:<7}{np.sum(df_dtrain['ID']==-1):<7}")

        features_list = feature_str.split()
        X_train = df_dtrain[features_list]
        X_test  = df_dtest[features_list]
        
        y_train = df_dtrain[["ID"]]
        y_test  = df_dtest[["ID"]]

        ## Initialize the model of choice and hyperparameter grid to search over ##
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier
        # from sklearn.ensemble import GradientBoostingClassifier

        param_grid = {} # for cross validation
        if (model_nm == "Logistic"):
            model = LogisticRegression()
            param_grid = {'C': [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100], 'penalty':['l2']}   # default, l2 regularization: we only have 3 features max, we don't want to zero out any. 

        if (model_nm == "SVM"):
            model = SVC()
            param_grid = {'C':[0.01, 0.1, 1, 10, 100], 'gamma':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}

        if (model_nm == "AdaBDT"):
            # model = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2))
            model = AdaBoostClassifier() 
            param_grid = {'n_estimators':np.arange(5, 105, 5), 'estimator':[DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=2), DecisionTreeClassifier(max_depth=3), DecisionTreeClassifier(max_depth=4), DecisionTreeClassifier(max_depth=5)]}

        # if (model_nm == "GradientBoostingClassifier"):
        #     model = GradientBoostingClassifier()
        #     param_grid = {'n_estimators':np.arange(5, 130, 5), 'learning_rate':[0.001, 0.01, 0.1, 1, 10], 'max_depth':[1, 2, 3, 4, 5, 6]}

        ## Find the best hyperparameters on the train/validation (development) set using cross validation... ##
        print(f"Finding best parameters for {model_nm} model on training set using grid search and CV...")

        from sklearn.model_selection import GridSearchCV, StratifiedKFold
        import sklearn.metrics

        print("Scoring models during CV using", self.cv_score)

        cvgs = GridSearchCV(estimator=model, 
                            param_grid=param_grid, 
                            scoring=self.cv_score, 
                            refit=True, 
                            cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
                            n_jobs=4)  
        
        # The parallelization for GridSearchCV is implemented using joblib (statements involving Parallel(...)), specifically the "loky" backend, which will automatically limit the number of threads used by lower-level child processes (like those called by NumPy routines / some base-estimator fitting routines) 
        # Each child proccess is limited to max_threads = n_cpus // n_jobs (where n_jobs is the GridSearchCV parameter, which gets passed into the parallel = Parallel(...) constructor. Then the parallel loop is defined within --> with parallel: ... )   

        # How GridSearchCV  works is that it "clones" the Estimator object passed into the estimator= kwarg to use as the base_estimator -- creating an identical but separate (!=) Estimator object -- then it begins a parallelized loop where in each instance of the loop it creates another clone of the base_estimator and fits it on a combination of hyperparameters from the grid.
        # At the end, if refit=True it creates one more *clone* of the base_estimator object (self.best_estimator_ = clone(base_estimator)), sets the params of self.best_estimator to a **clone of the obtained self.best_params_ dictionary** (meaning any objects used as params for best_estimator are identical but != to objects in self.best_params_), and fits self.best_estimator on the whole training set.
        cvgs.fit(X_train, np.ravel(y_train))

        # Logistic Regression --> lower the C, stronger the regularization, simpler the model 
        # SVM --> lower the C, wider the margin (more lenient on no. of misclassified points), simpler the model/decision boundary
        #     --> lower the gamma ("inverse of rbf kernel radius"), simpler/"smoother" (larger the "scale" / "resolution" of) the model/decision boundary 
        #     --> apparently, model behavior is much more sensitive to gamma: for small enough gamma (complicated-enough boundary shape), making C larger (thinner margin) doesn't give additional benefit  
        
        # The "best model" is the one that obtains the highest test score (as per self.cv_score) on average across the 10-fold CV
        assert cvgs.cv_results_["mean_test_score"][cvgs.best_index_] == cvgs.best_score_ 
        print(f"Params of model with highest mean test {self.cv_score} score across the 10-fold CV on train set:", cvgs.best_params_)
        print(f"where mean test {self.cv_score} score was", cvgs.best_score_)

        print(f"{self.cv_score} score of this 'best model' on whole training set:", cvgs.score(X_train, y_train))
        print("Model uses features:", cvgs.feature_names_in_) 

        best_model = cvgs.best_estimator_

        # v Just for personal understanding of gridsearchCV 
        assert cvgs.best_params_ == cvgs.cv_results_['params'][cvgs.best_index_]  # make sure cvgs.best_params_ is the _literal same_ as the dictionary at the cvgs.best_index_th place of cvgs.cv_results['params']   
        # v make sure cvgs.best_estimator_ is the model with params equal to cvgs.best_params_   
        for k, v in cvgs.best_params_.items(): 
            # One of the hyperparameters which we are trying to optimize for the AdaBDT is the "estimator" parameter, which (in this case) takes in a DecisionTreeClassifier object which in turn has varying values of its max_depth hyperparam. However, two different DecisionTreeClassifier objects, even if they share all parameters, are not == equal unless they are literally the same object in memory. So, the DecisionTreeClassifier object that gets stored in cvgs.best_params_ (which *is* the dictionary at cvgs.cv_results_['params'][cvgs.best_index_], containing the *literal* same objects) is a different object from the DT which gets created to be passed into the new copy of the AdaBDT being refit with the best params.  
            if k=='estimator':   
                assert v.get_params()['max_depth'] == best_model.get_params()[k].get_params()['max_depth']
            else:
                assert v == best_model.get_params()[k]  
        
        if self.cv_score == 'roc_auc':                                                                        # y_true,  y_score
            assert cvgs.score(X_train, np.ravel(y_train)) == getattr(sklearn.metrics, f"{self.cv_score}_score")(y_train, best_model.decision_function(X_train))     #sklearn.metrics.roc_auc_score takes arguments (y_true, y_score)
            assert cvgs.score(X_train, np.ravel(y_train)) == getattr(sklearn.metrics, f"{self.cv_score}_score")(y_train, model.set_params(**cvgs.best_params_).fit(X_train, np.ravel(y_train)).decision_function(X_train)) # make sure cvgs.best_estimator_ is the model with params equal to cvgs.best_params_
        else:                                                                                                 # y_true,  y_pred
            assert cvgs.score(X_train, np.ravel(y_train)) == getattr(sklearn.metrics, f"{self.cv_score}_score")(y_train, best_model.predict(X_train))               #sklearn.metrics.<>_score takes arguments (y_true, y_pred) where <> is precision, accuracy or recall
            assert cvgs.score(X_train, np.ravel(y_train)) == getattr(sklearn.metrics, f"{self.cv_score}_score")(y_train, model.set_params(**cvgs.best_params_).fit(X_train, np.ravel(y_train)).predict(X_train))           # make sure cvgs.best_estimator_ is the model with params equal to cvgs.best_params_

        ## ...and assess performance on the external test (evaluation) set. ##
        y_pred_test = best_model.predict(X_test)
        df_dtest['PredID'] = y_pred_test
        assert (df_dtest['PredID'] == cvgs.predict(X_test)).all()

        accuracy              = np.sum((df_dtest['PredID']==df_dtest['ID']))          / len(df_dtest)
                                                # True positive nue                              # Actual nue 
        nue_recall            = np.sum((df_dtest['PredID']== 1)&(df_dtest['ID']== 1)) / np.sum((df_dtest['ID']    == 1))
        try:                                    # True positive nue                              # Predicted nue 
            nue_precision     = np.sum((df_dtest['PredID']== 1)&(df_dtest['ID']== 1)) / np.sum((df_dtest['PredID']== 1))
        except (RuntimeWarning, ZeroDivisionError):
            nue_precision = 0
                                                # True positive antinue                          # Actual antinue 
        antinue_recall        = np.sum((df_dtest['PredID']==-1)&(df_dtest['ID']==-1)) / np.sum((df_dtest['ID']    ==-1))
        try:                                    # True positive antinue                          # Predicted antinue
            antinue_precision = np.sum((df_dtest['PredID']==-1)&(df_dtest['ID']==-1)) / np.sum((df_dtest['PredID']==-1))
        except (RuntimeWarning, ZeroDivisionError):
            antinue_precision = 0
        
        assert accuracy         ==sklearn.metrics.accuracy_score( y_true=y_test, y_pred=y_pred_test)
        assert nue_recall       ==sklearn.metrics.recall_score(   y_true=y_test, y_pred=y_pred_test, average=None, labels=(1, -1))[0]
        assert antinue_recall   ==sklearn.metrics.recall_score(   y_true=y_test, y_pred=y_pred_test, average=None, labels=(1, -1))[1]
        assert nue_precision    ==sklearn.metrics.precision_score(y_true=y_test, y_pred=y_pred_test, average=None, labels=(1, -1), zero_division=0)[0] # If there is a zero division error i.e. when the model predicts 0 nue or 0 antinue, it will return *0* for the corresponding precision (given there are no positives, there are zero *true* positives, which is the more important info to convey)
        assert antinue_precision==sklearn.metrics.precision_score(y_true=y_test, y_pred=y_pred_test, average=None, labels=(1, -1), zero_division=0)[1] # This logic is written out manually above. 

        # assert nue_precision    ==sklearn.metrics.precision_score(y_true=y_test, y_pred=y_pred_test, average=None, labels=(1, -1), zero_division=np.nan)[0] # If there is a zero division error i.e. when the model predicts 0 nue or 0 antinue, it will return np.nan for the corresponding precision
        # ^^^ The zero_division = np.nan option was added in scikit-learn 1.3.0, we have scikit 1.2.1. I don't want to mess anyone else's potential work by upgrading using pip, so I'll use the zero_division=0 option. 

        print()        
        print(f"Accuracy          achieved by 'best model' on test set: {accuracy*100:.1f}%")
        print(f"Nue     recall    achieved by 'best model' on test set: {nue_recall*100:.1f}%")
        print(f"Nue     precision achieved by 'best model' on test set: {nue_precision*100:.1f}%")
        print(f"Antinue recall    achieved by 'best model' on test set: {antinue_recall*100:.1f}%")
        print(f"Antinue precision achieved by 'best model' on test set: {antinue_precision*100:.1f}%")
        print()

        discrim_scores_test = best_model.decision_function(X_test)
        # MAKE SURE THAT A POSITIVE DEC-FUNC SCORE CORRESPONDS TO A POSITIVE CLASS PREDICTION AND A NEGATIVE DEC-FUNC SCORE TO A NEGATIVE CLASS PREDICTION
        assert ((discrim_scores_test*y_pred_test) >= 0).all()

        ROC_curve = sklearn.metrics.roc_curve(y_true=y_test, 
                                              y_score=discrim_scores_test,
                                              pos_label=1) # ***Treating nue (ID: +1) here as the "positive"/"signal" class*** 
        ROC_auc = sklearn.metrics.roc_auc_score(y_true=y_test, 
                                                y_score=discrim_scores_test)
        
        print(f"Area under ROC curve achieved by 'best model' on test set: {ROC_auc*100:.1f}%")

        out_dict = {'ROC curve':       ROC_curve, 
                    'ROC AUC':         ROC_auc,
                    'Test set':        df_dtest,
                    'Decision scores': discrim_scores_test,
                    'Best model':      best_model,
                    'Best params':     cvgs.best_params_}

        return out_dict

from sklearn.base import BaseEstimator, TransformerMixin
class ManualFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, features=[]):
        self.features = features
    def fit(self, X, y):
        return self
    def transform(self, X, y=None):
        return X[list(self.features)]
        
# checks if we are running from command line
if __name__ == "__main__":

    n_args = len(sys.argv)

    if n_args < 2:
        print("need to specify models to run")
    elif n_args == 2:
        e = EventsTreeReader(model_list_filename=sys.argv[1])
    elif n_args == 3:
        e = EventsTreeReader(model_list_filename=sys.argv[1], out_filename=sys.argv[2])
    elif n_args == 4:
        e = EventsTreeReader(model_list_filename=sys.argv[1], out_filename=sys.argv[2], cv_score=sys.argv[3])
    else:
        e = EventsTreeReader(model_list_filename=sys.argv[1], out_filename=sys.argv[2], cv_score=sys.argv[3], model=sys.argv[4], f_option=sys.argv[5])


