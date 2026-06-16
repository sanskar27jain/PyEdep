import sys, os

import numpy as np
import matplotlib.pyplot as plt

from ROOT import TFile, TTree

class EventsTreeReader:

    def __init__(self, filename, out_filename="results.pdf", option=""):
        self.filename = filename
        self.out_filename = out_filename
        self.option = option

        # In case somehow the plots directory does not exist in PyEdep directory (from event.py)
        if not os.path.exists("./plots"):
            os.makedirs("./plots")
            print("./plots directory did not exist. It was created.")

        # from stack overflow
        import matplotlib as mpl
        mpl.rcParams['text.usetex'] = True
        mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}' #for \text command

        self.GetEnergies()

    def GetEnergies(self):
        f_in = TFile.Open(self.filename)
        edepTree = f_in.Sim

        # For debugging purposes
        print("Number of entries in edep-sim tree from writer: ", edepTree.GetEntries())
        # print("[e/μ, p, n, π+-, π0, γ, α, others] count in this event")

        self.event_energies = {}

        for entry in edepTree:
            E_nu = entry.E_nu
            if E_nu not in self.event_energies:
                self.event_energies[E_nu] = {}
                self.event_energies[E_nu]['E_avail'] = np.array([]) # A 1-row array. Every element in the array is the total E_avail of one specific event (entry).
                self.event_energies[E_nu]['E_dep'] = np.array([])
                self.event_energies[E_nu]['Q'] = np.array([])
                self.event_energies[E_nu]['Q_thre'] = np.array([])
                self.event_energies[E_nu]['L'] = np.array([])

                self.event_energies[E_nu]['L_th_75keV'] = np.array([])
                self.event_energies[E_nu]['L_th_500keV'] = np.array([])

                self.event_energies[E_nu]['Q_500'] = np.array([])
                self.event_energies[E_nu]['L_35'] = np.array([])

                self.event_energies[E_nu]['Q_e'] = np.array([])
                self.event_energies[E_nu]['Q_h'] = np.array([])

                # For R_cal visualization. May comment out later on.
                self.event_energies[E_nu]['L_e'] = np.array([])
                self.event_energies[E_nu]['L_h'] = np.array([])

                self.event_energies[E_nu]['E_dep_tracks'] = np.array([])
                # self.event_energies[E_nu]['E_dep_dots'] = np.array([]) ## debugging, requires change in event and writer.
                self.event_energies[E_nu]['Q_e_dots'] = np.array([])
                self.event_energies[E_nu]['Q_h_dots'] = np.array([])

                # Each column represents one of the events (entries), each row is for a different particle.
                self.event_energies[E_nu]['E_avail_list'] =            np.array([[], [], [], [], [], [], [], []]) # Collecting E_avail in each kind of particle (row) for each entry (column)
                self.event_energies[E_nu]['E_dep_list'] =              np.array([[], [], [], [], [], [], [], []]) # Collecting E_dep by each kind of particle (row) for each entry (column)
                self.event_energies[E_nu]['Q_dep_list_th75keV'] =      np.array([[], [], [], [], [], [], [], []])
                self.event_energies[E_nu]['Q_dep_list_dots_th75keV'] = np.array([[], [], [], [], [], [], [], []])
                self.event_energies[E_nu]['L_dep_list_180PEpMeV'] =    np.array([[], [], [], [], [], [], [], []])
                self.event_energies[E_nu]['N_par_list'] =              np.array([[], [], [], [], [], [], [], []])

                self.event_energies[E_nu]['L_dep_list_180PEpMeV_th_75keV']  = np.array([[], [], [], [], [], [], [], []])
                self.event_energies[E_nu]['L_dep_list_180PEpMeV_th_500keV'] = np.array([[], [], [], [], [], [], [], []])

                self.event_energies[E_nu]['E_dep_tracks_list'] =       np.array([[], [], [], [], [], [], [], []])

                self.event_energies[E_nu]['interaction'] = np.array([])

                self.event_energies[E_nu]['nEvents'] = 0
                self.event_energies[E_nu]['nu_pdg'] = np.array([], dtype=np.int64)

                self.event_energies[E_nu]['E_avail_gst'] = np.array([])
                self.event_energies[E_nu]['E_avail_pre_FSI_gst'] = np.array([])
                self.event_energies[E_nu]['N_par_list_gst'] = np.array([[], [], [], [], [], [], [], []])

                # Three more for the strange baryons
                self.event_energies[E_nu]['N_par_list_pre_FSI_gst'] = np.array([[], [], [], [], [], [], [], [], [], [], []])
                self.event_energies[E_nu]['interaction_gst'] = np.array([])

                self.event_energies[E_nu]['E_avail_list_dots'] = np.array([[], [], [], [], [], [], [], []])
                self.event_energies[E_nu]['N_par_list_extra'] = np.array([[], [], []])
                self.event_energies[E_nu]['N_pi+-_list_gst'] = np.array([[], []])


            self.event_energies[E_nu]['E_avail'] = np.append(self.event_energies[E_nu]['E_avail'], entry.E_avail)
            self.event_energies[E_nu]['E_dep'] = np.append(self.event_energies[E_nu]['E_dep'], entry.E_depoTotal)
            self.event_energies[E_nu]['Q'] = np.append(self.event_energies[E_nu]['Q'], entry.Q_depoTotal)
            self.event_energies[E_nu]['Q_thre'] = np.append(self.event_energies[E_nu]['Q_thre'], entry.Q_depoTotal_th_75keV) # 75 keV charge threshold
            self.event_energies[E_nu]['L'] = np.append(self.event_energies[E_nu]['L'], entry.L_depoTotal_avg_180PEpMeV) # Deposited energy in scintillation light assuming a light yield of 180 photoelectrons per MeV for MIPs (assuming 0.5 kV/cm E field, 0.8% APEX detector PCE, Birks Model)

            self.event_energies[E_nu]['Q_500'] = np.append(self.event_energies[E_nu]['Q_500'], entry.Q_depoTotal_th_500keV) # 500 keV charge threshold
            self.event_energies[E_nu]['L_35'] = np.append(self.event_energies[E_nu]['L_35'], entry.L_depoTotal_avg_35PEpMeV) # Deposited energy in scintillation light assuming a light yield of 35 photoelectrons per MeV for MIPs (assuming 0.5 kV/cm E field, 0.8% APEX detector PCE, Birks Model)

            # mu/e, pi0 and gamma (photons) make up the EM component. Using 75keV threshold.
            self.event_energies[E_nu]['Q_e'] = np.append(self.event_energies[E_nu]['Q_e'], np.sum([entry.Q_depoList_th_75keV[i] for i in [0, 4, 5]]))
            # p, n, pi+-, alpha, others (K, lambda, etc) make up hadronic component. Using 75keV threshold.
            self.event_energies[E_nu]['Q_h'] = np.append(self.event_energies[E_nu]['Q_h'], np.sum([entry.Q_depoList_th_75keV[i] for i in [1, 2, 3, 6, 7]]))

            self.event_energies[E_nu]['E_dep_tracks'] = np.append(self.event_energies[E_nu]['E_dep_tracks'], entry.E_depoTotal_track)
            # self.event_energies[E_nu]['E_dep_dots'] = np.append(self.event_energies[E_nu]['E_dep_dots'], entry.E_depoTotal_dots) ## debugging, requires change in event and writer
            self.event_energies[E_nu]['Q_e_dots'] = np.append(self.event_energies[E_nu]['Q_e_dots'], np.sum([entry.Q_depoList_dots_th_75keV[i] for i in [0, 4, 5]]))
            self.event_energies[E_nu]['Q_h_dots'] = np.append(self.event_energies[E_nu]['Q_h_dots'], np.sum([entry.Q_depoList_dots_th_75keV[i] for i in [1, 2, 3, 6, 7]]))

            self.event_energies[E_nu]['E_avail_list'] = np.append(self.event_energies[E_nu]['E_avail_list'], np.array([[entry.E_availList[i] for i in range(8)]]).T, axis=1)
            self.event_energies[E_nu]['E_dep_list'] = np.append(self.event_energies[E_nu]['E_dep_list'], np.array([[entry.E_depoList[i] for i in range(8)]]).T, axis=1)
            self.event_energies[E_nu]['Q_dep_list_th75keV'] = np.append(self.event_energies[E_nu]['Q_dep_list_th75keV'], np.array([[entry.Q_depoList_th_75keV[i] for i in range(8)]]).T, axis=1)
            self.event_energies[E_nu]['Q_dep_list_dots_th75keV'] = np.append(self.event_energies[E_nu]['Q_dep_list_dots_th75keV'], np.array([[entry.Q_depoList_dots_th_75keV[i] for i in range(8)]]).T, axis=1)
            self.event_energies[E_nu]['L_dep_list_180PEpMeV'] = np.append(self.event_energies[E_nu]['L_dep_list_180PEpMeV'], np.array([[entry.L_depoList_avg_180PEpMeV[i] for i in range(8)]]).T, axis=1)
            self.event_energies[E_nu]['N_par_list'] = np.append(self.event_energies[E_nu]['N_par_list'], np.array([[entry.N_parList[i] for i in range(8)]]).T, axis=1)

            self.event_energies[E_nu]['E_dep_tracks_list'] = np.append(self.event_energies[E_nu]['E_dep_tracks_list'], np.array([[entry.E_depoList_track[i] for i in range(8)]]).T, axis=1)
            #^^ index [1]: The energy deposited in any "track-like" depositions descended from primary proton.  

            # For R_cal visualization.
            self.event_energies[E_nu]['L_e'] = np.append(self.event_energies[E_nu]['L_e'], np.sum([entry.L_depoList_avg_180PEpMeV[i] for i in [0, 4, 5]]))
            self.event_energies[E_nu]['L_h'] = np.append(self.event_energies[E_nu]['L_h'], np.sum([entry.L_depoList_avg_180PEpMeV[i] for i in [1, 2, 3, 6, 7]]))

            #--------------------------------------------------------------------------------------
            # if self.event_energies[E_nu]['nEvents'] % 5 == 0: # for general testing purposes
            #     np.set_printoptions(precision=15)
            #     print([entry.E_availList[i] for i in range(8)]) # testing if self.event_energies[E_nu]['E_avail_list'] is being successfully populated
            #     print(np.array([[entry.E_availList[i] for i in range(8)]]))
            #     print(self.event_energies[E_nu]['E_avail_list'][:, self.event_energies[E_nu]['nEvents']], "\n")
            #     print(entry.nu_proc)
            #     print(self.event_energies[E_nu]['N_par_list'][:, self.event_energies[E_nu]['nEvents']], "event no ", self.event_energies[E_nu]['nEvents'])
            error_message = ""
            # error_message += f"{'E_AVAIL_OVER_E_NU ':<20}" if entry.E_avail > E_nu else f"{'':<20}"
            # error_message += f"{'E_DEP_OVER_E_NU ':<20}" if entry.E_depoTotal > E_nu else f"{'':<20}"
            # Below error seems to be gone in Genie v3, at least for 200 MeV numu
            error_message += f"{'E_DEP_OVER_E_AVAIL ':<20}" if entry.E_depoTotal > entry.E_avail else f"{'':<20}"
            error_message += f"{'NEUTRON E_DEP_OVER_E_AVAIL ':<30}" if entry.E_depoList[2] > entry.E_availList[2] else f"{'':<30}"
            #error_message += f"{'PROTON E_DEP_OVER_E_AVAIL ':<30}" if entry.E_depoList[1] > entry.E_availList[1] else f"{'':<30}"
            error_message += f"{'PROTON E_DEP_OVER_E_AVAIL ':<30}" if entry.E_depoList[1] - entry.E_availList[1] > 0.1 else f"{'':<30}"
            #error_message += f"{'E_DEP_IN_[68,70) ':<20}" if (entry.E_depoTotal >= 68 and entry.E_depoTotal < 70) else f"{'':<20}"
            #error_message += f"{'E_AVAIL_IN_[170,172) ':<23}" if (entry.E_avail >= 170 and entry.E_avail < 172) else f"{'':<23}"
            #error_message += f"{'N_NUCL==16 ':<15}" if (entry.N_parList[1]+entry.N_parList[2] == 16) else f"{'':<15}"
            #error_message += f"{'E_AVAIL_IN_[248.8,251.2] ':<26}" if (entry.E_avail >= 248.8 and entry.E_avail <= 251.2) else f"{'':<26}"
            if len(error_message.strip()) > 0:
                error_message = f"Event no {self.event_energies[E_nu]['nEvents']:<4}: " + error_message
#                print(error_message)
            #--------------------------------------------------------------------------------------

            self.event_energies[E_nu]['interaction'] = np.append(self.event_energies[E_nu]['interaction'], entry.nu_proc)

            self.event_energies[E_nu]['nEvents'] += 1
            self.event_energies[E_nu]['nu_pdg'] = np.append(self.event_energies[E_nu]['nu_pdg'], int(entry.nu_pdg))

            if hasattr(entry, "E_avail_gst"):
                self.event_energies[E_nu]['E_avail_gst']            = np.append(self.event_energies[E_nu]['E_avail_gst'], entry.E_avail_gst)
                self.event_energies[E_nu]['E_avail_pre_FSI_gst']    = np.append(self.event_energies[E_nu]['E_avail_pre_FSI_gst'], entry.E_avail_pre_FSI_gst)
                self.event_energies[E_nu]['N_par_list_gst']         = np.append(self.event_energies[E_nu]['N_par_list_gst'], np.array([[entry.N_parList_gst[i] for i in range(8)]]).T, axis=1)
                # To account for strange baryons at the primary vertices of antinue events, 11 places are used.
                self.event_energies[E_nu]['N_par_list_pre_FSI_gst'] = np.append(self.event_energies[E_nu]['N_par_list_pre_FSI_gst'], np.array([[entry.N_parList_pre_FSI_gst[i] for i in range(11)]]).T, axis=1)
                self.event_energies[E_nu]['interaction_gst']        = np.append(self.event_energies[E_nu]['interaction_gst'], entry.nu_proc_gst)
            
            if hasattr(entry, "E_availList_dots"):
                self.event_energies[E_nu]['E_avail_list_dots'] = np.append(self.event_energies[E_nu]['E_avail_list_dots'], np.array([[entry.E_availList_dots[i] for i in range(8)]]).T, axis=1)
                # ^^^ index [0]: Total available energy in primary electron minus all energy track-like deposited by primary electron + descendents.  
                self.event_energies[E_nu]['N_par_list_extra']  = np.append(self.event_energies[E_nu]['N_par_list_extra'], np.array([[entry.N_parList_extra[i] for i in range(3)]]).T, axis=1)
            
            if hasattr(entry, "N_pipmList_gst"):
                self.event_energies[E_nu]['N_pi+-_list_gst'] = np.append(self.event_energies[E_nu]['N_pi+-_list_gst'], np.array([[entry.N_pipmList_gst[i] for i in range(2)]]).T, axis=1)

            if hasattr(entry, "L_depoTotal_avg_180PEpMeV_th_75keV"):
                self.event_energies[E_nu]['L_th_75keV'] = np.append(self.event_energies[E_nu]['L_th_75keV'], entry.L_depoTotal_avg_180PEpMeV_th_75keV) # 75 keV light threshold
                self.event_energies[E_nu]['L_th_500keV'] = np.append(self.event_energies[E_nu]['L_th_500keV'], entry.L_depoTotal_avg_180PEpMeV_th_500keV) # 500 keV light threshold
    
                self.event_energies[E_nu]['L_dep_list_180PEpMeV_th_75keV'] = np.append(self.event_energies[E_nu]['L_dep_list_180PEpMeV_th_75keV'], 
                                                                                       np.array([[entry.L_depoList_avg_180PEpMeV_th_75keV[i] for i in range(8)]]).T, axis=1)
                self.event_energies[E_nu]['L_dep_list_180PEpMeV_th_500keV'] = np.append(self.event_energies[E_nu]['L_dep_list_180PEpMeV_th_500keV'], 
                                                                                       np.array([[entry.L_depoList_avg_180PEpMeV_th_500keV[i] for i in range(8)]]).T, axis=1)

        #--------------------------------------------------------------------------------------
        # Energy Reconstruction
        self.resolutions = {'E_rec_L1_res': {}, 'E_rec_Q1_res': {}, 'E_rec_Q2_res': {}, 'E_rec_Q3_res': {}, 'E_rec_(Q+L)1_res': {}, 'E_rec_(Q+L)2_res': {},
                            'L1_res_1p':    {}, 'Q1_res_1p':    {}, 'Q2_res_1p':    {}, 'Q3_res_1p':    {}, '(Q+L)1_res_1p':    {}, '(Q+L)2_res_1p':    {}}

        self.biases = {'E_rec_L1_b': {}, 'E_rec_Q1_b': {}, 'E_rec_Q2_b': {}, 'E_rec_Q3_b': {}, 'E_rec_(Q+L)1_b': {}, 'E_rec_(Q+L)2_b': {},
                       'L1_b_1p':    {}, 'Q1_b_1p':    {}, 'Q2_b_1p':    {}, 'Q3_b_1p':    {}, '(Q+L)1_b_1p':    {}, '(Q+L)2_b_1p':    {}}

        self.iqr = {'E_rec_L1_iqr': {}, 'E_rec_Q1_iqr': {}, 'E_rec_Q2_iqr': {}, 'E_rec_Q3_iqr': {}, 'E_rec_(Q+L)1_iqr': {}, 'E_rec_(Q+L)2_iqr': {},
                    'L1_iqr_1p':    {}, 'Q1_iqr_1p':    {}, 'Q2_iqr_1p':    {}, 'Q3_iqr_1p':    {}, '(Q+L)1_iqr_1p':    {}, '(Q+L)2_iqr_1p':    {}}

        self.paper_nue_scaling_factors = {"L1":        0.42,
                                          "Q1":        0.48,
                                          "Q2_e":      0.55,
                                          "Q2_h":      0.31,
                                          "Q3_e_dots": 0.47,
                                          "Q3_h_dots": 0.20}

        self.Genie_v3_nue_100MeV_to_1GeV_peak_scaling_factors = {"L1":        0.465,
                                                                 "Q1":        0.485,
                                                                 "Q2_e":      0.553,
                                                                 "Q2_h":      0.215,
                                                                 "Q3_e_dots": 0.475,
                                                                 "Q3_h_dots": 0.165,
                                                                 "(Q+L)1":    0.984,
                                                                 "(Q+L)2_e":  0.975,
                                                                 "(Q+L)2_h":  0.997}

        self.newSplines_nue_100MeV_to_1GeV_peak_scaling_factors = {"L1":        0.465,
                                                                   "Q1":        0.485,
                                                                   "Q2_e":      0.553,
                                                                   "Q2_h":      0.225,
                                                                   "Q3_e_dots": 0.475,
                                                                   "Q3_h_dots": 0.165,
                                                                   "(Q+L)1":    0.984,
                                                                   "(Q+L)2_e":  0.975,
                                                                   "(Q+L)2_h":  0.997}

        self.Genie_v3_nue_100MeV_to_1GeV_mean_scaling_factors = {"L1":        0.480,
                                                                 "Q1":        0.432,
                                                                 "Q2_e":      0.548,
                                                                 "Q2_h":      0.268,
                                                                 "Q3_e_dots": 0.461,
                                                                 "Q3_h_dots": 0.203,
                                                                 "(Q+L)1":    0.912,
                                                                 "(Q+L)2_e":  0.971,
                                                                 "(Q+L)2_h":  0.846}

        self.Genie_v3_nue_200MeV_to_1GeV_peak_scaling_factors = {"L1":        0.465,
                                                                 "Q1":        0.485,
                                                                 "Q2_e":      0.553,
                                                                 "Q2_h":      0.245,
                                                                 "Q3_e_dots": 0.475,
                                                                 "Q3_h_dots": 0.165,
                                                                 "(Q+L)1":    0.984,
                                                                 "(Q+L)2_e":  0.975,
                                                                 "(Q+L)2_h":  0.997}

        # For 1p only nue events -- FSI-less QES events. 28% of all nue events.
        self.Genie_v3_nue_100MeV_to_1GeV_1p0n0pi_scaling_factors = {"L1":        0.465,
                                                                    "Q1":        0.515,    # A higher scaling factor means a sharper resolution (division)
                                                                    "Q2_e":      0.553,
                                                                    "Q2_h":      0.215,
                                                                    "Q3_e_dots": 0.475,
                                                                    "Q3_h_dots": 0.215,
                                                                    "(Q+L)1":    0.984,
                                                                    "(Q+L)2_e":  0.975,
                                                                    "(Q+L)2_h":  1.003}

        self.Genie_v3_nue_200MeV_to_1GeV_1p0n0pi_scaling_factors = {"L1":        0.465,
                                                                    "Q1":        0.515,
                                                                    "Q2_e":      0.553,
                                                                    "Q2_h":      0.345,    # A higher scaling factor means Q2 won't overshoot as much anymore
                                                                    "Q3_e_dots": 0.475,
                                                                    "Q3_h_dots": 0.555,    # Not sure about this...
                                                                    "(Q+L)1":    0.984,
                                                                    "(Q+L)2_e":  0.975,
                                                                    "(Q+L)2_h":  1.003}

        self.Genie_v3_antinue_100MeV_to_1GeV_peak_scaling_factors = {"L1":        0.415,
                                                                     "Q1":        0.495,
                                                                     "Q2_e":      0.553,
                                                                     "Q2_h":      0.195,
                                                                     "Q3_e_dots": 0.475,
                                                                     "Q3_h_dots": 0.145,
                                                                     "(Q+L)1":    0.925,
                                                                     "(Q+L)2_e":  0.977,
                                                                     "(Q+L)2_h":  0.615}

        self.newSplines_antinue_100MeV_to_1GeV_peak_scaling_factors = {"L1":        0.415,
                                                                       "Q1":        0.485,
                                                                       "Q2_e":      0.553,
                                                                       "Q2_h":      0.185,
                                                                       "Q3_e_dots": 0.475,
                                                                       "Q3_h_dots": 0.155,
                                                                       "(Q+L)1":    0.935,
                                                                       "(Q+L)2_e":  0.976,
                                                                       "(Q+L)2_h":  0.535}

        ############################
        # self.event_energies.pop(100)    # To remove 100 MeV from the list
        ############################
        if self.option == "E_dep_more":
            self.ideal_res = {"L1":{}, "Q1":{}, "Q2":{}, "Q3":{}}
            self.ideal_b   = {"L1":{}, "Q1":{}, "Q2":{}, "Q3":{}}
            self.QL_res   = {"L":{}, "Q":{}}
        
        for E_nu in self.event_energies.keys():
            info = self.event_energies[E_nu].copy()
            if info['nu_pdg'][0] in [12, 14]:
                scaling_factors = self.newSplines_nue_100MeV_to_1GeV_peak_scaling_factors
                scaling_factors_1p = self.Genie_v3_nue_100MeV_to_1GeV_1p0n0pi_scaling_factors
                # scaling_factors = self.paper_nue_scaling_factors
                # self.E_a_to_E_nu_shift = 28.73      # 28.7269 MeV
                # self.E_a_to_E_nu_shift = 28.72      # For 200-1000 MeV
                self.E_a_to_E_nu_shift = 0          # NO SHIFT
            if info['nu_pdg'][0] in [-12, -14]:
                scaling_factors = self.newSplines_antinue_100MeV_to_1GeV_peak_scaling_factors
                # self.E_a_to_E_nu_shift = 31.38    # 31.3765 MeV
                self.E_a_to_E_nu_shift = 0    # NO SHIFT

            self.event_energies[E_nu]['E_rec_L1'] = (info['L'] / scaling_factors["L1"])                                               + self.E_a_to_E_nu_shift
            self.event_energies[E_nu]['E_rec_Q1'] = (info['Q_thre'] / scaling_factors["Q1"])                                          + self.E_a_to_E_nu_shift
            self.event_energies[E_nu]['E_rec_Q2'] = (info['Q_e'] / scaling_factors["Q2_e"]) + (info['Q_h'] / scaling_factors["Q2_h"]) + self.E_a_to_E_nu_shift
            self.event_energies[E_nu]['E_rec_Q3'] = info['E_dep_tracks'] + (info['Q_e_dots'] / scaling_factors["Q3_e_dots"]) + (info['Q_h_dots'] / scaling_factors["Q3_h_dots"]) + self.E_a_to_E_nu_shift
            self.event_energies[E_nu]['E_rec_(Q+L)1'] = ((info['Q_thre'] + info['L']) / scaling_factors["(Q+L)1"])                    + self.E_a_to_E_nu_shift
            self.event_energies[E_nu]['E_rec_(Q+L)2'] = ((info['Q_e'] + info['L_e']) / scaling_factors["(Q+L)2_e"]) + ((info['Q_h'] + info['L_h']) / scaling_factors["(Q+L)2_h"]) + self.E_a_to_E_nu_shift

            self.resolutions['E_rec_L1_res'][E_nu] = EventsTreeReader.calculateRes(self.event_energies[E_nu]['E_rec_L1'])
            self.resolutions['E_rec_Q1_res'][E_nu] = EventsTreeReader.calculateRes(self.event_energies[E_nu]['E_rec_Q1'])
            self.resolutions['E_rec_Q2_res'][E_nu] = EventsTreeReader.calculateRes(self.event_energies[E_nu]['E_rec_Q2'])
            self.resolutions['E_rec_Q3_res'][E_nu] = EventsTreeReader.calculateRes(self.event_energies[E_nu]['E_rec_Q3'])
            self.resolutions['E_rec_(Q+L)1_res'][E_nu] = EventsTreeReader.calculateRes(self.event_energies[E_nu]['E_rec_(Q+L)1'])
            self.resolutions['E_rec_(Q+L)2_res'][E_nu] = EventsTreeReader.calculateRes(self.event_energies[E_nu]['E_rec_(Q+L)2'])

            self.biases['E_rec_L1_b'][E_nu] = EventsTreeReader.calculateBias(self.event_energies[E_nu]['E_rec_L1'], E_nu)
            self.biases['E_rec_Q1_b'][E_nu] = EventsTreeReader.calculateBias(self.event_energies[E_nu]['E_rec_Q1'], E_nu)
            self.biases['E_rec_Q2_b'][E_nu] = EventsTreeReader.calculateBias(self.event_energies[E_nu]['E_rec_Q2'], E_nu)
            self.biases['E_rec_Q3_b'][E_nu] = EventsTreeReader.calculateBias(self.event_energies[E_nu]['E_rec_Q3'], E_nu)
            self.biases['E_rec_(Q+L)1_b'][E_nu] = EventsTreeReader.calculateBias(self.event_energies[E_nu]['E_rec_(Q+L)1'], E_nu)
            self.biases['E_rec_(Q+L)2_b'][E_nu] = EventsTreeReader.calculateBias(self.event_energies[E_nu]['E_rec_(Q+L)2'], E_nu)
            for rec in ['L1', 'Q1', 'Q2', 'Q3', '(Q+L)1', '(Q+L)2']:
                self.iqr['E_rec_'+rec+'_iqr'][E_nu] = EventsTreeReader.calculateIqr(self.event_energies[E_nu]['E_rec_'+rec])

            if info['nu_pdg'][0] > 0:
                mask_1p = np.array( [ (col ==  [1, 1, 0, 0, 0, 0, 0, 0]).all() for col in info['N_par_list'].T ] )    # Not separating pi+- for now
                self.E_a_to_E_nu_shift_1p = 28.86    # 28.86346 MeV
                self.event_energies[E_nu]['E_rec_L1_1p'] = (info['L'][mask_1p] / scaling_factors_1p["L1"])                   + self.E_a_to_E_nu_shift_1p
                self.event_energies[E_nu]['E_rec_Q1_1p'] = (info['Q_thre'][mask_1p] / scaling_factors_1p["Q1"])              + self.E_a_to_E_nu_shift_1p
                self.event_energies[E_nu]['E_rec_Q2_1p'] = (info['Q_e'][mask_1p] / scaling_factors_1p["Q2_e"]) + (info['Q_h'][mask_1p] / scaling_factors_1p["Q2_h"]) + self.E_a_to_E_nu_shift_1p
                self.event_energies[E_nu]['E_rec_Q3_1p'] = info['E_dep_tracks'][mask_1p] + (info['Q_e_dots'][mask_1p] / scaling_factors_1p["Q3_e_dots"]) + (info['Q_h_dots'][mask_1p] / scaling_factors_1p["Q3_h_dots"]) + self.E_a_to_E_nu_shift_1p
                self.event_energies[E_nu]['E_rec_(Q+L)1_1p'] = ((info['Q_thre'] + info['L'])[mask_1p] / scaling_factors_1p["(Q+L)1"]) + self.E_a_to_E_nu_shift_1p
                self.event_energies[E_nu]['E_rec_(Q+L)2_1p'] = ((info['Q_e'] + info['L_e'])[mask_1p] / scaling_factors_1p["(Q+L)2_e"]) + ((info['Q_h'] + info['L_h'])[mask_1p] / scaling_factors_1p["(Q+L)2_h"]) + self.E_a_to_E_nu_shift_1p

                self.resolutions['L1_res_1p'][E_nu] = EventsTreeReader.calculateRes(self.event_energies[E_nu]['E_rec_L1_1p'])
                self.resolutions['Q1_res_1p'][E_nu] = EventsTreeReader.calculateRes(self.event_energies[E_nu]['E_rec_Q1_1p'])
                self.resolutions['Q2_res_1p'][E_nu] = EventsTreeReader.calculateRes(self.event_energies[E_nu]['E_rec_Q2_1p'])
                self.resolutions['Q3_res_1p'][E_nu] = EventsTreeReader.calculateRes(self.event_energies[E_nu]['E_rec_Q3_1p'])
                self.resolutions['(Q+L)1_res_1p'][E_nu] = EventsTreeReader.calculateRes(self.event_energies[E_nu]['E_rec_(Q+L)1_1p'])
                self.resolutions['(Q+L)2_res_1p'][E_nu] = EventsTreeReader.calculateRes(self.event_energies[E_nu]['E_rec_(Q+L)2_1p'])

                self.biases['L1_b_1p'][E_nu] = EventsTreeReader.calculateBias(self.event_energies[E_nu]['E_rec_L1_1p'], E_nu)
                self.biases['Q1_b_1p'][E_nu] = EventsTreeReader.calculateBias(self.event_energies[E_nu]['E_rec_Q1_1p'], E_nu)
                self.biases['Q2_b_1p'][E_nu] = EventsTreeReader.calculateBias(self.event_energies[E_nu]['E_rec_Q2_1p'], E_nu)
                self.biases['Q3_b_1p'][E_nu] = EventsTreeReader.calculateBias(self.event_energies[E_nu]['E_rec_Q3_1p'], E_nu)
                self.biases['(Q+L)1_b_1p'][E_nu] = EventsTreeReader.calculateBias(self.event_energies[E_nu]['E_rec_(Q+L)1_1p'], E_nu)
                self.biases['(Q+L)2_b_1p'][E_nu] = EventsTreeReader.calculateBias(self.event_energies[E_nu]['E_rec_(Q+L)2_1p'], E_nu)
                for rec in ['L1', 'Q1', 'Q2', 'Q3', '(Q+L)1', '(Q+L)2']:
                    self.iqr[rec+'_iqr_1p'][E_nu] = EventsTreeReader.calculateIqr(self.event_energies[E_nu]['E_rec_'+rec+'_1p'])

                print(E_nu, "avg E_a - E_nu for 1p  evts:", np.mean(info['E_avail'][mask_1p] - E_nu))
            
            print(E_nu, "avg E_a - E_nu for all evts:", np.mean(info['E_avail'] - E_nu))
            print(E_nu, "avg E_a / E_nu for all evts:", np.mean(info['E_avail'] / E_nu))

            #--------------------------------------------------------------------------------------
            # Debugging / stats
            E_al = info['E_avail_list']
            E_a, E_a_pFS = info['E_avail_gst'], info['E_avail_pre_FSI_gst']
            E_dl = info['E_dep_list']
            N_pl = info['N_par_list']
            N_pl_pFS = info['N_par_list_pre_FSI_gst'] # New version with 11 places
            print(info["nu_pdg"][0])
            print(f"Percentage of available energy deposited by EM component over all {E_nu} MeV events: {(E_dl[[0, 4, 5]].sum() / E_al.sum()) * 100:.2f}%")
            print(f"Percentage of available energy allocated to protons      over all {E_nu} MeV events: {(E_al[1].sum()         / E_al.sum()) * 100:.2f}%")
            print(f"Percentage of available energy deposited by protons      over all {E_nu} MeV events: {(E_dl[1].sum()         / E_al.sum()) * 100:.2f}%")
            print(f"Percentage of available energy deposited by neutrons     over all {E_nu} MeV events: {(E_dl[2].sum()         / E_al.sum()) * 100:.2f}%")
            print(f"Percentage of available energy deposited by pi+-s        over all {E_nu} MeV events: {(E_dl[3].sum()         / E_al.sum()) * 100:.2f}%")
            #print(f"Percentage of available energy deposited by others       over all {E_nu} MeV events: {(E_dl[[6, 7]].sum()    / E_al.sum()) * 100:.2f}%")
            print()
            print(f"Percentage of hadronic E_a   in protons  over all {E_nu} MeV events: {(E_al[1].sum() / E_al[[1, 2, 3, 6, 7]].sum()) * 100:.2f}%")
            print(f"Percentage of hadronic E_a   in neutrons over all {E_nu} MeV events: {(E_al[2].sum() / E_al[[1, 2, 3, 6, 7]].sum()) * 100:.2f}%")
            print(f"Percentage of hadronic E_dep in protons  over all {E_nu} MeV events: {(E_dl[1].sum() / E_dl[[1, 2, 3, 6, 7]].sum()) * 100:.2f}%")
            print(f"Percentage of hadronic E_dep in neutrons over all {E_nu} MeV events: {(E_dl[2].sum() / E_dl[[1, 2, 3, 6, 7]].sum()) * 100:.2f}%")
            print(f"Average proton  multiplicity over all {E_nu} MeV events: {np.mean(N_pl[1])}")
            print(f"Average neutron multiplicity over all {E_nu} MeV events: {np.mean(N_pl[2])}")
            print()
            print(f"No. of events where E_avail exceeds {E_nu=} MeV: {len( info['E_avail'][ info['E_avail'] > E_nu ] )}")
            print(f"No. of events where E_dep exceeds {E_nu=} MeV:   {len( info['E_dep'][ info['E_avail'] > E_nu ] )}")
            print(f"No. of events where E_dep exceeds E_avail for {E_nu=} MeV: {len( info['E_avail'][ info['E_dep'] > info['E_avail'] ] )}")
            print()

            # FSI effects
            print(f"No. of events where FSI increases E_a for {E_nu=} MeV: {np.sum(E_a > E_a_pFS)}")
            print(f"No. of events where FSI decreases E_a for {E_nu=} MeV: {np.sum(E_a < E_a_pFS)}")
            print(f"No. of events where E_a is unchanged pre and post-FSI for {E_nu=} MeV: {np.sum(E_a == E_a_pFS)}")
            #print(f"Average proton  multiplicity over all {E_nu} MeV events with FSI (E_a =/= E_a_pFS): {np.mean(N_pl[1][E_a != E_a_pFS])}")
            #print(f"Average neutron multiplicity over all {E_nu} MeV events with FSI (E_a =/= E_a_pFS): {np.mean(N_pl[2][E_a != E_a_pFS])}")

            N_pl_pFS_tmp = N_pl_pFS.copy()    
            N_pl_pFS_tmp[7] = np.sum(N_pl_pFS_tmp[7:], axis=0)
            N_pl_pFS = N_pl_pFS_tmp[:8]

            no_fsi_N_pl_change = np.array([(N_pl_pFS[:,i] == N_pl[:,i]).all() for i in range(N_pl.shape[1])], dtype=np.bool_)
            print(f"No. of events where E_a is unchanged pre and post-FSI but N_pl changes for {E_nu=} MeV: {np.sum((E_a == E_a_pFS) & (~no_fsi_N_pl_change))}")

            print("\n Average differece between E_avail and E_avail_gst: ", np.mean(info["E_avail"] - info["E_avail_gst"]), "\n")

            if info['N_pi+-_list_gst'].size > 0 and not (info['N_pi+-_list_gst'].sum(axis=0) == info['N_par_list'][3]).all():
                print("The pi+, pi- counts from the gst file do not match the pi+- counts from the edepsim tree")
                sys.exit()

            #--------------------------------------------------------------------------------------
            # Call plotting functions

            if self.option == "E_dep_hist":
                self.plotE_depQLHistogram(E_nu)

            if self.option == "E_dep_scatter":
                self.plotE_availE_depScatter(E_nu)

            if self.option == "E_avail_pre_FSI":
                self.plotE_availPreFSIHistogram(E_nu)

            if self.option == "E_dep_FSI":
                self.plotE_depQLFSIInfoHistogram(E_nu)

            if self.option == "E_dep_particles":
                self.plotE_depParticlesHistogram(E_nu)

            if self.option == "E_dep_processes":
                self.plotE_depProcessesHistogram(E_nu)

            if self.option == "E_dep_fs":
                self.plotE_depFSHistogram(E_nu)

            if self.option == "E_dep_tplgy":
                self.plotE_depTplgy(E_nu)

            if self.option == "E_dep_more":
                res, bi, L_res, Q_th_res = self.plotE_depMoreInfo(E_nu, ep=False)
                for k, v in res.items():
                    self.ideal_res[k][E_nu] = v
                for k, v in bi.items():
                    self.ideal_b[k][E_nu] = v
                self.QL_res["L"][E_nu] = L_res
                self.QL_res["Q"][E_nu] = Q_th_res

            if self.option == "E_dep_1p":
                self.plotE_depMoreInfo(E_nu, ep=True)

            if self.option == "N_par":
                self.plotN_par(E_nu)

            if self.option == "E_rec_hist":
                eaview = scaling_factors == self.Genie_v3_nue_100MeV_to_1GeV_mean_scaling_factors
                self.plotE_recHistogram(E_nu, eaview=eaview)

            if self.option == "E_rec_hist_1p":
                self.plotE_recHistogram(E_nu, ep=True)

            if self.option == "multiplicity_scatter":
                self.plotMultiplicityE_availScatter(E_nu)

            if self.option == "particles_pre_FSI":
                self.plotPreFSIParticles(E_nu)

            #print(f"E_rec_L1 resolution: {self.resolutions['E_rec_L1_res'][E_nu]*100:.2f}%")
            #print(f"E_rec_Q1 resolution: {self.resolutions['E_rec_Q1_res'][E_nu]*100:.2f}%")
            #print(f"E_rec_Q2 resolution: {self.resolutions['E_rec_Q2_res'][E_nu]*100:.2f}%")
            #print(f"E_rec_Q3 resolution: {self.resolutions['E_rec_Q3_res'][E_nu]*100:.2f}%")

        if self.option == "E_rec_resolutions" or self.option == "E_dep_more" or self.option == "E_rec_hist":
            self.plotE_recResolutions()
        if self.option == "E_rec_resolutions_iqr":
            self.plotE_recResolutions(iqr=True)
        if self.option == "E_rec_resolutions_1p":
            self.plotE_recResolutions(ep=True)
        if self.option == "R_cal_hist":
            self.plotR_calHistogram()
        if self.option == "R_cal_hist_mean":
            self.plotR_calHistogram(peaks=False)
        if self.option == "R_cal_hist_1p":
            self.plotR_calHistogram(tplgy="1p0n0π")
        if self.option == "multiplicity_hist":
            self.plotMultiplicityHistogram()
        if self.option == "fsi_stats":
            self.plotFSIStats()
        if self.option == "E_dep_components_separately":
            self.plotE_depComponentsSeparately()
        if self.option == "missing_energy":
            self.plotMissingEnergy()

        if (self.option == "modified_Q3"):            
            self.modified_Q3()
        if (self.option == "particle_light_responses"):
            self.particle_light_responses()
        if (self.option == "particle_charge_responses"):
            self.particle_charge_responses()
        if (self.option == "particle_QL_responses"):
            self.particle_QL_responses()
        if (self.option == "E_avail_res"):
            self.E_avail_res()
        if (self.option == "E_avail_breakdown"):
            self.E_avail_breakdown()
        if (self.option == "missing_energy_by_particle"):
            self.missing_energy_by_particle()


        allE_a = np.concatenate([info['E_avail_gst'] for E_nu, info in self.event_energies.items()])
        allE_a_pFS = np.concatenate([info['E_avail_pre_FSI_gst'] for E_nu, info in self.event_energies.items()])
        print("No. of events where FSI increases E_a", np.sum(allE_a > allE_a_pFS))
        print("Average increase among those events", np.mean((allE_a - allE_a_pFS)[allE_a > allE_a_pFS]))
        print("No. of events where FSI decreases E_a", np.sum(allE_a < allE_a_pFS))
        print("No. of events where FSI does not change E_a", np.sum(allE_a == allE_a_pFS))        

        allE_al = np.concatenate([info['E_avail_list'] for E_nu, info in self.event_energies.items()], axis=1)
        allE_dl = np.concatenate([info['E_dep_list'] for E_nu, info in self.event_energies.items()], axis=1)
        print("Neutron R_dep", np.sum(allE_dl[2]) / np.sum(allE_al[2]))

        # Light Threshold Test
        # E_nu = 500
        # info = self.event_energies[E_nu]
        # Q, Q_75, Q_500, L, L_75, L_500 = (info['Q'], info['Q_thre'], info['Q_500'], 
        #                                   info['L'], info['L_th_75keV'], info['L_th_500keV'])
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

        # self.SaveAndClose(E_nu)

    #---------------------------------------------------
    def calculateRes(energyDist):
        # rms = np.sqrt(np.mean(energyDist**2))    # RMS = sqrt(E[x^2]), it is not the same as the standard deviation which is sqrt(E[x^2] - (E[x])^2)
        # return rms / np.mean(energyDist)
        return np.std(energyDist) / np.mean(energyDist)

    def calculateIqr(energyDist):
        p25, p75 = np.percentile(energyDist, [25, 75])
        return (p75 - p25) / np.mean(energyDist)

    def calculateBias(energyDist, trueE):
        return np.mean((energyDist - trueE) / trueE)

    def calculateBinEdges(min_x, max_x, bin_width):
        midrange_x, halfrange_x = (min_x + max_x) / 2, (max_x - min_x) / 2
        xlim_min, xlim_max = midrange_x - (1.1 * halfrange_x), midrange_x + (1.1 * halfrange_x)
        nearest_greater_multiple = xlim_max + bin_width - (xlim_max % bin_width)
        nearest_smaller_multiple = xlim_min - (xlim_min % bin_width)
        bin_edges = np.linspace(nearest_smaller_multiple, nearest_greater_multiple, int((nearest_greater_multiple-nearest_smaller_multiple) / bin_width) + 1)
        return bin_edges

    def SaveAndClose(self, E_nu):
        if len(self.event_energies.keys()) > 1:
            plt.savefig("./plots/" + self.out_filename[:-4] + "_" + str(int(E_nu)) + "MeV.pdf")
        else:
            plt.savefig("./plots/" + self.out_filename)

        plt.clf()
        plt.close()
        print("Plot created.")

    #---------------------------------------------------
    def plotE_depQLHistogram(self, E_nu):
        info = self.event_energies[E_nu]
        E_a, E_d, Q, Q_th, L, E_a_pFS = info['E_avail'], info['E_dep'], info['Q'], info['Q_thre'], info['L'], info['E_avail_pre_FSI_gst']
        bin_width = E_nu / 100
        # bin_edges = np.linspace(0, 1.25*E_nu, 126)

        # min_x, max_x = min(E_a.min(), E_d.min(), Q.min(), Q_th.min(), L.min()), max(E_a.max(), E_d.max(), Q.max(), Q_th.max(), L.max())
        min_x, max_x = np.concatenate((E_a, E_d, Q, Q_th, L, E_a_pFS)).min(), np.concatenate((E_a, E_d, Q, Q_th, L, E_a_pFS)).max()
        midrange_x, halfrange_x = (min_x + max_x) / 2, (max_x - min_x) / 2
        xlim_min, xlim_max = midrange_x - (1.1 * halfrange_x), midrange_x + (1.1 * halfrange_x)
        nearest_greater_multiple = xlim_max + bin_width - (xlim_max % bin_width)
        bin_edges = np.linspace(0, nearest_greater_multiple, int(nearest_greater_multiple / bin_width) + 1)

        plt.figure()
        # plt.hist(info['E_avail'], bins=int(np.ptp(info['E_avail']) / bin_width), histtype="step", label="E_avail")
        plt.axvline(E_nu, linestyle="--", alpha=0.8, color="grey",   label=r"$E_\nu$")
        plt.hist(E_a_pFS, bins=bin_edges, histtype="step", lw=2,   label=r"$E_\text{avail, pre-FSI}$", color='C9', linestyle=":")
        plt.hist(E_a,     bins=bin_edges, histtype="step", lw=2,   label=r"$E_\text{avail}$") 
        plt.hist(E_d,     bins=bin_edges, histtype="step", lw=2,   label=r"$E_\text{dep}$")

        # print(info['Q_e_dots'])
        # print(info['Q_h_dots'])

        # plt.hist(info['E_dep_tracks'], bins=int(np.ptp(info['E_dep_tracks']) / bin_width), histtype="step", label="E_dep_tracks")
        ## Debugging, needs change in event and writer to work ##
        # plt.hist(info['E_dep_dots'], bins=int(np.ptp(info['E_dep_dots']) / bin_width), histtype="step", label="E_dep_dots")
        # plt.hist(info['E_dep_dots'] + info['E_dep_tracks'], bins=int(np.ptp(info['E_dep_dots'] + info['E_dep_tracks']) / bin_width), histtype="step", label="E_dep_dots + E_dep_tracks")

        plt.xlim(xlim_min, xlim_max)
        plt.xlabel("Energy [MeV]", fontsize="large")
        plt.ylabel("Number of events", fontsize="large")
        plt.legend(loc=("upper left" if E_nu > 200 else "best"),fontsize="large")
        # plt.title(f"nu:{info['nu_pdg'][0]}, E_nu = {E_nu} MeV, bin width = {bin_width} MeV, {info['nEvents']} events", fontsize='small')

        self.SaveAndClose(E_nu)

        plt.figure()
        plt.hist(E_d,  bins=bin_edges,                 lw=2, histtype="step", color="C1", label=r"$E_\text{dep}$")
        plt.hist(Q,    bins=bin_edges, linestyle="--", lw=2, histtype="step", color="C3", label="$Q$")
        plt.hist(Q_th, bins=bin_edges,                 lw=2, histtype="step", color="C2", label=r"$Q_\text{thre}$")
        plt.hist(L,    bins=bin_edges,                 lw=2, histtype="step", color="C4", label=r"$L$")
        plt.xlim(xlim_min, xlim_max)
        plt.xlabel("Energy [MeV]",     fontsize="large")
        plt.ylabel("Number of events", fontsize="large")
        plt.legend(loc=("upper left" if E_nu > 200 else "best"), fontsize="large")

        if len(self.event_energies.keys()) > 1:
            plt.savefig("./plots/" + self.out_filename[:-4] + "_" + str(int(E_nu)) + "MeV_2.pdf")
        else:
            plt.savefig("./plots/" + self.out_filename[:-4] + "_2.pdf")

        plt.clf()
        plt.close()
        print("Plot created.")


    #---------------------------------------------------
    # This function was more of an exploration using the Gv3 200 MeV numu file to check if 1p only events would have a better chance for reconstruction
    def plotE_depQLFSIInfoHistogram(self, E_nu):
        info = self.event_energies[E_nu]
        E_a, N_pl = info['E_avail_gst'], info['N_par_list_gst']  # Post-FSI info from GST
        E_d, Q_th, L = info['E_dep'], info['Q_thre'], info['L'] # edepsim info, obviously post-FSI
        simple_event_mask = (N_pl[1] == 1) & (N_pl[2] == 0)     # There is only one post-FSI primary nucleon: a proton
        E_a_simple, E_d_simple, Q_th_simple, L_simple = (arr[simple_event_mask] for arr in [E_a, E_d, Q_th, L])
        #E_a_nonsimple, E_d_nonsimple, Q_th_nonsimple, L_nonsimple = (arr[~simple_event_mask] for arr in [E_a, E_d, Q_th, L])

        bin_width = E_nu / 100
        # bin_edges = np.linspace(0, 1.25*E_nu, 126)
        min_x, max_x = min(E_a.min(), E_d.min(), Q_th.min(), L.min()), max(E_a.max(), E_d.max(), Q_th.max(), L.max())
        midrange_x, halfrange_x = (min_x + max_x) / 2, (max_x - min_x) / 2
        xlim_min, xlim_max = midrange_x - (1.1 * halfrange_x), midrange_x + (1.1 * halfrange_x)
        nearest_greater_multiple = xlim_max + bin_width - (xlim_max % bin_width)
        bin_edges = np.linspace(0, nearest_greater_multiple, int(nearest_greater_multiple / bin_width) + 1)

        fig, axs = plt.subplots(4)
        fig.set_size_inches(6.4, 12.8)
        fig.set_layout_engine("tight")
        fig.suptitle(f"E_nu = {E_nu} MeV, bin width = {bin_width} MeV, {info['nEvents']} events. Simple: FS has 1 proton only", fontsize="medium")

        # axs[0].hist([E_a_simple, E_a_nonsimple], bins=bin_edges, histtype="stepfilled", stacked=True, label=["E_avail (simple)", "E_avail (other)"])
        # axs[1].hist([E_d_simple, E_d_nonsimple], bins=bin_edges, histtype="stepfilled", stacked=True, label=["E_dep (simple)", "E_dep (other)"])
        # axs[2].hist([Q_th_simple, Q_th_nonsimple], bins=bin_edges, histtype="stepfilled", stacked=True, label=["Q (dQ > 75keV) (simple)", "Q (other)"])
        # axs[3].hist([L_simple, L_nonsimple], bins=bin_edges, histtype="stepfilled", stacked=True, label=["L (simple)", "L (other)"])
        axs[0].hist(E_a_simple,  bins=bin_edges, histtype="stepfilled", label=f"E_avail (post-FSI: 1 proton only)\n Mean: {E_a_simple.mean():.6f}, Stdev: {E_a_simple.std():.6f}")
        axs[0].hist(E_a,         bins=bin_edges, histtype="step",       label="E_avail (all events)",            color="black")
        axs[1].hist(E_d_simple,  bins=bin_edges, histtype="stepfilled", label=f"E_dep (post-FSI: 1 proton only)\n Mean: {E_d_simple.mean():.6f}, Stdev: {E_d_simple.std():.6f}")
        axs[1].hist(E_d,         bins=bin_edges, histtype="step",       label="E_dep (all events)",              color="black")
        axs[2].hist(Q_th_simple, bins=bin_edges, histtype="stepfilled", label=f"Q (dQ > 75keV) (post-FSI: 1 proton only)\n Mean: {Q_th_simple.mean():.6f}, Stdev: {Q_th_simple.std():.6f}")
        axs[2].hist(Q_th,        bins=bin_edges, histtype="step",       label="Q (dQ > 75keV) (all events)",     color="black")
        axs[3].hist(L_simple,    bins=bin_edges, histtype="stepfilled", label=f"L (post-FSI: 1 proton only)\n Mean: {L_simple.mean():.6f}, Stdev: {L_simple.std():.6f}")
        axs[3].hist(L,           bins=bin_edges, histtype="step",       label="L (all events)",                  color="black")

        for ax in axs.flat:
            ax.set_xlim(xlim_min, xlim_max)
            ax.set_xlabel("Energy [MeV]", fontsize="small")
            ax.set_ylabel("Number of events", fontsize="small")
            ax.legend(fontsize="small")

        self.SaveAndClose(E_nu)

    #-------------------------------------------------
    def plotE_availPreFSIHistogram(self, E_nu):
        info = self.event_energies[E_nu]
        E_a, E_a_pFS = info['E_avail_gst'], info['E_avail_pre_FSI_gst']
        bin_width = E_nu / 100
        # bin_edges = np.linspace(0, 1.25*E_nu, 126)

        min_x, max_x = min(E_a.min(), E_a_pFS.min()), max(E_a.max(), E_a_pFS.max())
        midrange_x, halfrange_x = (min_x + max_x) / 2, (max_x - min_x) / 2
        xlim_min, xlim_max = midrange_x - (1.2 * halfrange_x), midrange_x + (1.1 * halfrange_x)
        nearest_greater_multiple = xlim_max + bin_width - (xlim_max % bin_width)
        bin_edges = np.linspace(0, nearest_greater_multiple, int(nearest_greater_multiple / bin_width) + 1)

        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5)
        fig.set_size_inches(6.4, 24.8)
        fig.set_layout_engine('tight')
        fig.suptitle(f"nu:{info['nu_pdg'][0]}, E_nu = {E_nu} MeV, bin width = {bin_width} MeV, {info['nEvents']} events. All info from GST.", y=0.99, fontsize='medium')

        hghts, _, _ = ax1.hist([E_a, E_a_pFS], bins=bin_edges, histtype="step", label=["E_avail", "E_avail before FSI"])
        ax1.set_xlim(xlim_min, xlim_max)
        ax1.set_xlabel("Energy [MeV]")
        ax1.set_ylabel("Number of events")
        ax1.set_title("E_avail distribution before and after FSI", fontsize="small", loc="left")
        ax1.legend()
        peak_left_edge_x = np.argmax(hghts[1]) * bin_width           # Getting the energy corresponding to the left edge of the peak in E_a_pFS
        peak_right_edge_x = (np.argmax(hghts[1]) + 1) * bin_width    # Getting the energy corresponding to the right edge of the peak in E_a_pFS
        pre_FSI_peak_mean = E_a_pFS[(E_a_pFS >= peak_left_edge_x) & (E_a_pFS < peak_right_edge_x)].mean()

        # zoom inset
        #if info['nu_pdg'][0] == 12:  # nue
        #    if E_nu < 300:
        #        axins = ax1.inset_axes([0.35, 0.2, 0.6, 0.6])    # [(Lower left corner x), (Lower left corner y), (Width), (Height)]
        #        axins.set_xlim(peak_right_edge_x, xlim_max)
        #    elif E_nu < 700:
        #        axins = ax1.inset_axes([0.51, 0.25, 0.42, 0.4])
        #        axins.set_xlim(peak_right_edge_x, xlim_max)
        #    if E_nu < 700:  # not the most efficient way of doing this, but it works
        #        axins.hist([E_a, E_a_pFS], bins=bin_edges, histtype="step")
        #        axins.set_ylim(0, 40)
        #        axins.tick_params(labelsize='x-small')
        #        axins.set_title("(Zoom in)", fontsize="small", y=1, loc="left")

        #if info['nu_pdg'][0] == -12: # antinue
        #    if E_nu < 300:
        #        axins = ax1.inset_axes([0.35, 0.2, 0.6, 0.6])
        #        axins.set_xlim(peak_right_edge_x, xlim_max)
        #    elif E_nu < 400:
        #        axins = ax1.inset_axes([0.51, 0.25, 0.42, 0.4])
        #        axins.set_xlim(peak_right_edge_x, xlim_max)
        #    else:
        #        axins = ax1.inset_axes([0.1, 0.27, 0.42, 0.4])
        #        axins.set_xlim(0, peak_left_edge_x)
        #    axins.hist([E_a, E_a_pFS], bins=bin_edges, histtype="step")
        #    axins.set_ylim(0, 40)
        #    axins.tick_params(labelsize='x-small')
        #    axins.set_title("(Zoom in)", fontsize="small", y=1, loc="left")

        proc_masks = [
            ('QES', (info['interaction_gst'] == 11)),
            ('RES', (info['interaction_gst'] == 12)),
            ('DIS', (info['interaction_gst'] == 13)),
            ('COH', (info['interaction_gst'] == 14)),
            ('MEC', (info['interaction_gst'] == 15)),
        ]

        E_a_breakdown = [E_a[mask] for (proc, mask) in proc_masks]
        E_a_pFS_breakdown = [E_a_pFS[mask] for (proc, mask) in proc_masks]
        E_a_pFS_labels = [f"{proc}, {len(E_a_pFS_breakdown[i])} events total" for i, (proc, mask) in enumerate(proc_masks)]
        ax2.hist(E_a_pFS_breakdown, bins=bin_edges, histtype="stepfilled", stacked=True, label=E_a_pFS_labels)
        ax2.hist(E_a_pFS, bins=bin_edges, histtype="step", color="black", label=f"All events E_avail (before FSI).\nMean val in peak={pre_FSI_peak_mean:.2f}")
        ax2.set_xlim(xlim_min, xlim_max)
        ax2.set_xlabel("Energy [MeV]")
        ax2.set_ylabel("Number of events")
        ax2.set_title("E_avail distribution before FSI broken down by interaction type", fontsize="small", loc="left")
        ax2.legend(fontsize='x-small')

        # zoom inset
        # if info['nu_pdg'][0] == 12:  # nue
        #     if E_nu < 300:
        #         axins2 = ax2.inset_axes([0.35, 0.2, 0.6, 0.6])    # [(Lower left corner x), (Lower left corner y), (Width), (Height)]
        #         axins2.set_xlim(peak_right_edge_x, xlim_max)
        #     elif E_nu < 700:
        #         axins2 = ax2.inset_axes([0.51, 0.25, 0.42, 0.4])
        #         axins2.set_xlim(peak_right_edge_x, xlim_max)
        #     if E_nu < 700:  # not the most efficient way of doing this, but it works
        #         axins2.hist(E_a_pFS_breakdown, bins=bin_edges, histtype="stepfilled", stacked=True)
        #         axins2.hist(E_a_pFS, bins=bin_edges, histtype="step", color="black")
        #         axins2.set_ylim(0, 40)
        #         axins2.tick_params(labelsize='x-small')
        #         axins2.set_title("(Zoom in)", fontsize="small", y=1, loc="left")

        # if info['nu_pdg'][0] == -12: # antinue
        #     if E_nu < 300:
        #         axins2 = ax2.inset_axes([0.35, 0.2, 0.6, 0.6])
        #         axins2.set_xlim(peak_right_edge_x, xlim_max)
        #     elif E_nu < 400:
        #         axins2 = ax2.inset_axes([0.51, 0.25, 0.42, 0.4])
        #         axins2.set_xlim(peak_right_edge_x, xlim_max)
        #     else:
        #         axins2 = ax2.inset_axes([0.1, 0.27, 0.42, 0.4])
        #         axins2.set_xlim(0, peak_left_edge_x)
        #     axins2.hist(E_a_pFS_breakdown, bins=bin_edges, histtype="stepfilled", stacked=True)
        #     axins2.hist(E_a_pFS, bins=bin_edges, histtype="step", color="black")
        #     axins2.set_ylim(0, 40)
        #     axins2.tick_params(labelsize='x-small')
        #     axins2.set_title("(Zoom in)", fontsize="small", y=1, loc="left")

        ax3.plot(np.linspace(E_a_pFS.min()+3, E_a_pFS.max(), 10), np.linspace(E_a_pFS.min()+3, E_a_pFS.max(), 10), '--', alpha=0.5, color='grey', label='E_avail_post_FSI = E_avail_pre_FSI')
        for i, (proc, mask) in enumerate(proc_masks):
            ax3.plot(E_a_pFS_breakdown[i], E_a_breakdown[i], '.', markersize=1, alpha=0.5, label=E_a_pFS_labels[i])
        ax3.set_xlabel("E_avail pre-FSI [MeV]")
        ax3.set_ylabel("E_avail post-FSI [MeV]")
        ax3.set_title("E_avail after FSI vs E_avail before FSI, broken down by interaction type", fontsize="small", loc="left")
        ax3.legend(fontsize='x-small')

        N_pl, N_pl_pFS = info['N_par_list_gst'], info['N_par_list_pre_FSI_gst']
        pre_FSI_states_masks = [
            # QES
            # ("e|μ,p",     np.array( [ (col == [1,1,0,0,0,0,0,0]).all() for col in N_pl_pFS.T ] )),
            ("e|μ,p",    np.array( [ (col == [1,1,0,0,0,0,0,0,0,0,0]).all() for col in N_pl_pFS.T ] )),
            ("e|μ,n",    np.array( [ (col == [1,0,1,0,0,0,0,0,0,0,0]).all() for col in N_pl_pFS.T ] )),
            ("e|μ,Λ",    np.array( [ (col == [1,0,0,0,0,0,0,0,1,0,0]).all() for col in N_pl_pFS.T ] )),
            ("e|μ,Σ0",   np.array( [ (col == [1,0,0,0,0,0,0,0,0,1,0]).all() for col in N_pl_pFS.T ] )),
            ("e|μ,Σ-",   np.array( [ (col == [1,0,0,0,0,0,0,0,0,0,1]).all() for col in N_pl_pFS.T ] )),
            # RES/DIS
            ("e|μ,p,π0", np.array( [ (col == [1,1,0,0,1,0,0,0,0,0,0]).all() for col in N_pl_pFS.T ] )),
            ("e|μ,p,π±", np.array( [ (col == [1,1,0,1,0,0,0,0,0,0,0]).all() for col in N_pl_pFS.T ] )),
            ("e|μ,p,Nπ", np.array( [ (col[0]==1) & (col[1]==1) & (col[2]==0) & (col[3]+col[4] > 1) for col in N_pl_pFS.T ] )),
            ("e|μ,n,π0", np.array( [ (col == [1,0,1,0,1,0,0,0,0,0,0]).all() for col in N_pl_pFS.T ] )),
            ("e|μ,n,π±", np.array( [ (col == [1,0,1,1,0,0,0,0,0,0,0]).all() for col in N_pl_pFS.T ] )),
            ("e|μ,n,Nπ", np.array( [ (col[0]==1) & (col[1]==0) & (col[2]==1) & (col[3]+col[4] > 1) for col in N_pl_pFS.T ] )),
            # COH - The Ar40 nucleus is also considered one of the primary particles in this case
            ("e|μ,π0",   np.array( [ (col == [1,0,0,0,1,0,0,1,0,0,0]).all() for col in N_pl_pFS.T ] )),
            ("e|μ,π±",   np.array( [ (col == [1,0,0,1,0,0,0,1,0,0,0]).all() for col in N_pl_pFS.T ] )),
            # MEC
            ("e|μ,2p",   np.array( [ (col == [1,2,0,0,0,0,0,0,0,0,0]).all() for col in N_pl_pFS.T ] )),
            ("e|μ,p,n",  np.array( [ (col == [1,1,1,0,0,0,0,0,0,0,0]).all() for col in N_pl_pFS.T ] )),
            ("e|μ,2n",   np.array( [ (col == [1,0,2,0,0,0,0,0,0,0,0]).all() for col in N_pl_pFS.T ] ))
        ]

        E_a_breakdown = [E_a[mask] for (state, mask) in pre_FSI_states_masks]
        E_a_pFS_breakdown = [E_a_pFS[mask] for (state, mask) in pre_FSI_states_masks]
        E_a_pFS_labels = [f"{state}, {len(E_a_pFS_breakdown[i])} events total" if len(E_a_pFS_breakdown[i]) > 0 else "" for i, (state, mask) in enumerate(pre_FSI_states_masks)]
        E_a_pFS_colors = ['lightcyan', 'aqua', 'deepskyblue', 'blue', 'navy', 'bisque', 'orange', 'peru', 'khaki', 'gold', 'goldenrod', 'pink', 'red', 'mediumpurple', 'purple', 'indigo']
        ax4.hist(E_a_pFS_breakdown, bins=bin_edges, histtype="stepfilled", stacked=True, label=E_a_pFS_labels, color=E_a_pFS_colors)
        ax4.hist(E_a_pFS, bins=bin_edges, histtype="step", color="black", label=f"All events E_avail (before FSI)")
        ax4.set_xlim(xlim_min, xlim_max)
        ax4.set_xlabel("Energy [MeV]")
        ax4.set_ylabel("Number of events")
        ax4.set_title("E_avail distribution before FSI broken down by pre-FSI particles", fontsize="small", loc="left")
        ax4.legend(fontsize='x-small')

        # zoom inset
        # if info['nu_pdg'][0] == 12:  # nue
        #     if E_nu < 300:
        #         axins4 = ax4.inset_axes([0.35, 0.2, 0.6, 0.6])    # [(Lower left corner x), (Lower left corner y), (Width), (Height)]
        #         axins4.set_xlim(peak_right_edge_x, xlim_max)
        #     elif E_nu < 700:
        #         axins4 = ax4.inset_axes([0.51, 0.25, 0.42, 0.4])
        #         axins4.set_xlim(peak_right_edge_x, xlim_max)
        #     if E_nu < 700:  # not the most efficient way of doing this, but it works
        #         axins4.hist(E_a_pFS_breakdown, bins=bin_edges, histtype="stepfilled", stacked=True, color=E_a_pFS_colors)
        #         axins4.hist(E_a_pFS, bins=bin_edges, histtype="step", color="black")
        #         axins4.set_ylim(0, 40)
        #         axins4.tick_params(labelsize='x-small')
        #         axins4.set_title("(Zoom in)", fontsize="small", y=1, loc="left")

        # if info['nu_pdg'][0] == -12: # antinue
        #     if E_nu < 300:
        #         axins4 = ax4.inset_axes([0.35, 0.2, 0.6, 0.6])
        #         axins4.set_xlim(peak_right_edge_x, xlim_max)
        #     elif E_nu < 400:
        #         axins4 = ax4.inset_axes([0.51, 0.25, 0.42, 0.4])
        #         axins4.set_xlim(peak_right_edge_x, xlim_max)
        #     else:
        #         axins4 = ax4.inset_axes([0.1, 0.27, 0.42, 0.4])
        #         axins4.set_xlim(0, peak_left_edge_x)
        #     axins4.hist(E_a_pFS_breakdown, bins=bin_edges, histtype="stepfilled", stacked=True, color=E_a_pFS_colors)
        #     axins4.hist(E_a_pFS, bins=bin_edges, histtype="step", color="black")
        #     axins4.set_ylim(0, 40)
        #     axins4.tick_params(labelsize='x-small')
        #     axins4.set_title("(Zoom in)", fontsize="small", y=1, loc="left")

        no_extra_nucl_mask = (N_pl[1] == N_pl_pFS[1]) & (N_pl[2] == N_pl_pFS[2])
        # no_change_mask = np.array([(N_pl[:, i] == N_pl_pFS[:8, i]).all() for i in range(N_pl_pFS.shape[1])])
        # pi_change_mask = (N_pl[3] != N_pl_pFS[3]) | (N_pl[4] != N_pl_pFS[4])
        extra_nucl_mask = (N_pl[1] > N_pl_pFS[1]) | (N_pl[2] > N_pl_pFS[2])
        E_a_pFS_noExNucl, E_a_noExNucl = E_a_pFS[no_extra_nucl_mask], E_a[no_extra_nucl_mask]
        # E_a_pFS_noChange, E_a_noChange = E_a_pFS[no_change_mask], E_a[no_change_mask]
        # E_a_pFS_piChange, E_a_piChange = E_a_pFS[pi_change_mask & ~extra_nucl_mask], E_a[pi_change_mask & ~extra_nucl_mask]
        E_a_pFS_ExNucl, E_a_ExNucl = E_a_pFS[extra_nucl_mask], E_a[extra_nucl_mask]
        no_extra_nucl_label = f"({len(E_a_pFS_noExNucl)}) events: proton & neutron mplicities unchanged by FSI"
        # no_change_label = f"({len(E_a_pFS_noChange)}) events: pre-FSI and post-FSI system have same number of all particles"
        # pi_change_label = f"({len(E_a_pFS_piChange)}) events: p,n mplicities are unchanged but change in pion numbers"
        extra_nucl_label = f"({len(E_a_pFS_ExNucl)}) events: proton or neutron mplicity increased by FSI"

        ax5.plot(np.linspace(E_a_pFS.min()+3, E_a_pFS.max(), 10), np.linspace(E_a_pFS.min()+3, E_a_pFS.max(), 10), '--', alpha=0.5, color='grey', label='E_avail_post_FSI = E_avail_pre_FSI')
        ax5.plot(E_a_pFS_noExNucl, E_a_noExNucl, '.b', markersize=1, alpha=0.5, label=no_extra_nucl_label)
        # ax5.plot(E_a_pFS_noChange, E_a_noChange, '.b', markersize=1, alpha=0.5, label=no_change_label)
        # ax5.plot(E_a_pFS_piChange, E_a_piChange, '.m', markersize=1, alpha=0.5, label=pi_change_label)
        # Be mindful of what is being drawn on top of what.
        # We want to put the extra nucleon case on top because we want to show that whenever there are extra protons or neutrons, E_avail increases.
        ax5.plot(E_a_pFS_ExNucl, E_a_ExNucl, '.C3', markersize=1, alpha=0.5, label=extra_nucl_label)
        ax5.set_xlabel("E_avail pre-FSI [MeV]")
        ax5.set_ylabel("E_avail post-FSI [MeV]")
        ax5.set_title("E_avail after FSI vs E_avail before FSI. See legend for breakdown.", fontsize="small", loc="left")
        ax5.legend(fontsize='x-small')

        #axins4 = ax4.inset_axes([0.65, 0.07, 0.32, 0.32])
        #axins4.plot(np.linspace(E_a_pFS.min()+0.5, E_a_pFS.max(), 10), np.linspace(E_a_pFS.min()+0.5, E_a_pFS.max(), 10), '--', alpha=0.5, color='grey', label='E_avail_post_FSI = E_avail_pre_FSI')
        #axins4.plot(E_a_pFS_noExNucl, E_a_noExNucl, '.b', markersize=1, alpha=0.5)
        #axins4.plot(E_a_pFS_ExNucl, E_a_ExNucl, '.C3', markersize=1, alpha=0.5)
        #axins4.set_xlim(169, 173)
        #axins4.set_ylim(169, 180)
        #axins4.tick_params(labelsize='xx-small')
        #axins4.set_title("(Zoom in)", fontsize="x-small", y=0.85, loc="left")

        self.SaveAndClose(E_nu)

    #---------------------------------------------------
    def plotE_availE_depScatter(self, E_nu):
        info = self.event_energies[E_nu]
        E_a, E_d, N_pl = info['E_avail'], info['E_dep'], info['N_par_list']
        E_a_mu, E_d_mu = info['E_avail_list'][0], info['E_dep_list'][0]

        fig, axs = plt.subplots(2)
        fig.set_size_inches(6.4, 9.6)
        fig.set_layout_engine('tight')
        fig.suptitle(f"E_nu = {E_nu} MeV, {info['nEvents']} events")
                               # (electron,       proton,         neutron,        pi+-,           pi0          )
        FS_masks = [
            ('e|μ,p',            ((N_pl[0] > 0) & (N_pl[1] > 0) & (N_pl[2]== 0) & (N_pl[3]== 0) & (N_pl[4]== 0))),
            ('e|μ,p,pi0',        ((N_pl[0] > 0) & (N_pl[1] > 0) & (N_pl[2]== 0) & (N_pl[3]== 0) & (N_pl[4] > 0))),
            ('e|μ,p,pi+-',       ((N_pl[0] > 0) & (N_pl[1] > 0) & (N_pl[2]== 0) & (N_pl[3] > 0) & (N_pl[4]== 0))),
            ('e|μ,p,pi0,pi+-',   ((N_pl[0] > 0) & (N_pl[1] > 0) & (N_pl[2]== 0) & (N_pl[3] > 0) & (N_pl[4] > 0))),
            ('e|μ,n',            ((N_pl[0] > 0) & (N_pl[1]== 0) & (N_pl[2] > 0) & (N_pl[3]== 0) & (N_pl[4]== 0))),
            ('e|μ,n,pi0',        ((N_pl[0] > 0) & (N_pl[1]== 0) & (N_pl[2] > 0) & (N_pl[3]== 0) & (N_pl[4] > 0))),
            ('e|μ,n,pi+-',       ((N_pl[0] > 0) & (N_pl[1]== 0) & (N_pl[2] > 0) & (N_pl[3] > 0) & (N_pl[4]== 0))),
            ('e|μ,n,pi0,pi+-',   ((N_pl[0] > 0) & (N_pl[1]== 0) & (N_pl[2] > 0) & (N_pl[3] > 0) & (N_pl[4] > 0))),
            ('e|μ,p,n',          ((N_pl[0] > 0) & (N_pl[1] > 0) & (N_pl[2] > 0) & (N_pl[3]== 0) & (N_pl[4]== 0))),
            ('e|μ,p,n,pi0',      ((N_pl[0] > 0) & (N_pl[1] > 0) & (N_pl[2] > 0) & (N_pl[3]== 0) & (N_pl[4] > 0))),
            ('e|μ,p,n,pi+-',     ((N_pl[0] > 0) & (N_pl[1] > 0) & (N_pl[2] > 0) & (N_pl[3] > 0) & (N_pl[4]== 0))),
            ('e|μ,p,n,pi0,pi+-', ((N_pl[0] > 0) & (N_pl[1] > 0) & (N_pl[2] > 0) & (N_pl[3] > 0) & (N_pl[4] > 0))),
            ('e|μ,pi0',          ((N_pl[0] > 0) & (N_pl[1]== 0) & (N_pl[2]== 0) & (N_pl[3]== 0) & (N_pl[4] > 0))),
            ('e|μ,pi+-',         ((N_pl[0] > 0) & (N_pl[1]== 0) & (N_pl[2]== 0) & (N_pl[3] > 0) & (N_pl[4]== 0))),
        ]
        FS_colors = ('lightcoral', 'coral', 'crimson', 'maroon', 'palegreen', 'yellowgreen', 'olive', 'darkolivegreen', 'lightskyblue', 'turquoise', 'dodgerblue', 'darkblue', 'violet', 'darkviolet')
        E_a_breakdown, E_d_breakdown = [E_a[mask] for FS, mask in FS_masks], [E_d[mask] for FS, mask in FS_masks]
        E_a_mu_breakdown, E_d_mu_breakdown = [E_a_mu[mask] for FS, mask in FS_masks], [E_d_mu[mask] for FS, mask in FS_masks]
        labels = [f"{FS} events, {len(E_a_breakdown[i])} events total" if len(E_a_breakdown[i]) > 0 else "" for i, (FS, mask) in enumerate(FS_masks)]

        axs[0].plot(np.linspace(E_d.min(), E_a.max(), 10), np.linspace(E_d.min(), E_a.max(), 10), '--', color='grey', alpha=0.5, label='E_dep = E_avail')
        axs[1].plot(np.linspace(E_d_mu.min()+100, E_a_mu.max(), 10), np.linspace(E_d_mu.min(), E_a_mu.max()-100, 10), '--', color='darkgrey', alpha=0.5, label='E_dep = E_avail - 100')
        for i in range(len(FS_masks)):
            axs[0].plot(E_a_breakdown[i], E_d_breakdown[i], '.', markersize=1, color=FS_colors[i], label=labels[i])
            axs[1].plot(E_a_mu_breakdown[i], E_d_mu_breakdown[i], '.', markersize=1, color=FS_colors[i], label=labels[i])
        axs[0].set_title("(All particles)", x=0.2, y=0.7)
        axs[1].set_title("(Muon, w/descendents)", x=0.2, y=0.7)
        for ax in axs.flat:
            ax.set_xlabel("E_avail [MeV]")
            ax.set_ylabel("E_dep [MeV]")
            ax.legend(fontsize="small")

        if len(self.event_energies.keys()) > 1:
            plt.savefig("./plots/" + self.out_filename[:-4] + "_" + str(int(E_nu)) + "MeV.pdf")
        else:
            plt.savefig("./plots/" + self.out_filename)

        plt.clf()
        plt.close()
        print("Plot created.")

    #---------------------------------------------------
    def plotE_depParticlesHistogram(self, E_nu):
        info = self.event_energies[E_nu]
        E_al = info['E_avail_list']
        E_dl = info['E_dep_list']
        N_parl = info['N_par_list']
        bin_width = E_nu / 100

        # Attempting a new binning method to ensure both equal width and aligned bins, as well as to avoid the zero range error.
        # Assuming all values of E_avail and E_dep will be within 1.25 times the true neutrino energy
        # The number of elements in bin_edges would be num = ((stop - start)/bin_width) + 1 = ((1.25*E_nu)/(E_nu/100)) + 1 = (1.25*100) + 1 = 126
        # bin_edges = np.linspace(0, 1.25*E_nu, 126)
        max_x = max(E_al.sum(axis=0).max(), E_dl.sum(axis=0).max())
        xlim_max = 1.05 * max_x
        nearest_greater_multiple = xlim_max + bin_width - (xlim_max % bin_width)
        bin_edges = np.linspace(0, nearest_greater_multiple, int(nearest_greater_multiple / bin_width) + 1)

        # fig, axs = plt.subplots(4, 3, sharex=True) # All plots in a column share an x-axis. Automatically removes x tick labels from upper plots.
        fig, axs = plt.subplots(4, 3)                # Since we have passed in the same array of bin edges into each histogram plot, all subplots
                                                     # will automatically have the same x range. This way we keep the tick labels.
        fig.suptitle(f"E_nu = {E_nu} MeV, bin width = {bin_width} MeV, {info['nEvents']} events", y=0.99)
        fig.set_size_inches(12.6, 16.9)
        fig.set_layout_engine('tight')

        axs[0, 0].set_title("Electron/Muon", fontsize='medium', pad=3.0)
        # bin_edges includes the left edge of 1st bin and right edge of last bin
        axs[0, 0].hist(E_al[0], bins=bin_edges, histtype="step", label="E_avail")
        axs[0, 0].hist(E_dl[0], bins=bin_edges, histtype="step", label="E_dep")

        axs[0, 1].set_title("Total", fontsize='medium', pad=3.0)
        axs[0, 1].hist(E_al.sum(axis=0), bins=bin_edges, histtype="step", label="E_avail")
        axs[0, 1].hist(E_dl.sum(axis=0), bins=bin_edges, histtype="step", label="E_dep")
        # axs[1, 1].hist(info['E_avail'], bins=bin_edges, histtype="step", label="E_avail")
        # axs[1, 1].hist(info['E_dep'], bins=bin_edges, histtype="step", label="E_dep")

        E_a_proton_wp = E_al[1, N_parl[1]>0]         # E_avail of protons for events where least 1 proton is present
        E_d_proton_wp = E_dl[1, N_parl[1]>0]         # E_dep of protons for events where least 1 proton is present
        axs[0, 2].set_title(f"Proton, {len(E_a_proton_wp)} events", fontsize='medium', pad=3.0)
        axs[0, 2].hist(E_a_proton_wp, bins=bin_edges, histtype="step", label="E_avail")
        axs[0, 2].hist(E_d_proton_wp, bins=bin_edges, histtype="step", label="E_dep")

        axs[1, 0].set_title("e+p", fontsize='medium', pad=3.0)
        axs[1, 0].hist(E_al[0:2].sum(axis=0), bins=bin_edges, histtype="step", label="E_avail")
        axs[1, 0].hist(E_dl[0:2].sum(axis=0), bins=bin_edges, histtype="step", label="E_dep")

        axs[1, 1].set_title("e+p+n", fontsize='medium', pad=3.0)
        axs[1, 1].hist(E_al[0:3].sum(axis=0), bins=bin_edges, histtype="step", label="E_avail")
        axs[1, 1].hist(E_dl[0:3].sum(axis=0), bins=bin_edges, histtype="step", label="E_dep")

        E_a_neutron_wp = E_al[2, N_parl[2]>0]         # E_avail of neutrons for events where least 1 neutron is present
        E_d_neutron_wp = E_dl[2, N_parl[2]>0]         # E_dep of neutrons for events where least 1 neutron is present
        axs[1, 2].set_title(f"Neutron, {len(E_a_neutron_wp)} events", fontsize='medium', pad=3.0)
        axs[1, 2].hist(E_a_neutron_wp, bins=bin_edges, histtype="step", label="E_avail")
        axs[1, 2].hist(E_d_neutron_wp, bins=bin_edges, histtype="step", label="E_dep")

        E_a_pi0_wp = E_al[4, N_parl[4]>0]         # E_avail of pi0s for events where least 1 pi0 is present
        E_d_pi0_wp = E_dl[4, N_parl[4]>0]         # E_dep of pi0s for events where least 1 pi0 is present
        axs[2, 0].set_title(f"pi0, {len(E_a_pi0_wp)} events", fontsize='medium', pad=3.0)
        axs[2, 0].hist(E_a_pi0_wp, bins=bin_edges, histtype="step", label="E_avail")
        axs[2, 0].hist(E_d_pi0_wp, bins=bin_edges, histtype="step", label="E_dep")

        axs[2, 1].set_title("e+p+pi(0/+/-)", fontsize='medium', pad=3.0)
        axs[2, 1].hist(E_al[[0,1,3,4]].sum(axis=0), bins=bin_edges, histtype="step", label="E_avail")
        axs[2, 1].hist(E_dl[[0,1,3,4]].sum(axis=0), bins=bin_edges, histtype="step", label="E_dep")

        axs[2, 2].set_title("e+n", fontsize='medium', pad=3.0)
        axs[2, 2].hist(E_al[[0,2]].sum(axis=0), bins=bin_edges, histtype="step", label="E_avail")
        axs[2, 2].hist(E_dl[[0,2]].sum(axis=0), bins=bin_edges, histtype="step", label="E_dep")

        #axs[2, 1].set_title("p+n", fontsize='medium', pad=3.0)
        #axs[2, 1].hist(E_al[1:3].sum(axis=0), bins=bin_edges, histtype="step", label="E_avail")
        #axs[2, 1].hist(E_dl[1:3].sum(axis=0), bins=bin_edges, histtype="step", label="E_dep")

        E_a_pipm_wp = E_al[3, N_parl[3]>0]         # E_avail of pi+-s for events where least 1 pi+- is present
        E_d_pipm_wp = E_dl[3, N_parl[3]>0]         # E_dep of pi+-s for events where least 1 pi+- is present
        axs[3, 0].set_title(f"pi+-, {len(E_a_pipm_wp)} events", fontsize='medium', pad=3.0)
        axs[3, 0].hist(E_a_pipm_wp, bins=bin_edges, histtype="step", label="E_avail")
        axs[3, 0].hist(E_d_pipm_wp, bins=bin_edges, histtype="step", label="E_dep")

        axs[3, 1].set_title("e+n+pi(0/+/-)", fontsize='medium', pad=3.0)
        axs[3, 1].hist(E_al[[0,2,3,4]].sum(axis=0), bins=bin_edges, histtype="step", label="E_avail")
        axs[3, 1].hist(E_dl[[0,2,3,4]].sum(axis=0), bins=bin_edges, histtype="step", label="E_dep")

        E_a_gao_wp = E_al[5:, (N_parl[5]>0) | (N_parl[6]>0) | (N_parl[7]>0)] # E_avail of gamma, alpha and others when at least one is present
        E_d_gao_wp = E_dl[5:, (N_parl[5]>0) | (N_parl[6]>0) | (N_parl[7]>0)]
        axs[3, 2].set_title(f"Gamma+Alpha+Others, {len(E_a_gao_wp[0])} events where at least 1 is present", fontsize='small', pad=3.0)
        axs[3, 2].hist(E_a_gao_wp.sum(axis=0), bins=bin_edges, histtype="step", label="E_avail")
        axs[3, 2].hist(E_d_gao_wp.sum(axis=0), bins=bin_edges, histtype="step", label="E_dep")

        #E_a_g_wp = E_al[5, N_parl[5]>0]
        #E_d_g_wp = E_dl[5, N_parl[5]>0]
        #E_a_ao_wp = E_al[6:, (N_parl[6]>0) | (N_parl[7]>0)] # E_avail of alpha and others when at least one is present
        #E_d_ao_wp = E_dl[6:, (N_parl[6]>0) | (N_parl[7]>0)]
        #axs[3, 1].set_title(f"Gamma, {len(E_a_g_wp)} events", fontsize='medium', pad=3.0)
        #axs[3, 1].hist(E_a_g_wp, bins=bin_edges, histtype="step", label="E_avail")
        #axs[3, 1].hist(E_d_g_wp, bins=bin_edges, histtype="step", label="E_dep")
        #axs[3, 2].set_title(f"Alpha+Others, {len(E_a_ao_wp[0])} events where at least 1 is present", fontsize='small', pad=3.0)
        #axs[3, 2].hist(E_a_ao_wp.sum(axis=0), bins=bin_edges, histtype="step", label="E_avail")
        #axs[3, 2].hist(E_d_ao_wp.sum(axis=0), bins=bin_edges, histtype="step", label="E_dep")

        for i in range(len(axs.flat)):
            ax = axs.flat[i]
            ax.tick_params('x', labelsize='xx-small')
            if i > (len(axs.flat) - len(axs[0, :]) - 1):    # if ax is in the last row
                ax.set_xlabel("Energy [MeV]")
            if i % len(axs[0, :]) == 0:                     # is ax is in the first column
                ax.set_ylabel("No. of events")

        fig.legend(handles=axs[0, 0].get_legend_handles_labels()[0])

        if len(self.event_energies.keys()) > 1:
            plt.savefig("./plots/" + self.out_filename[:-4] + "_" + str(int(E_nu)) + "MeV.pdf")
        else:
            plt.savefig("./plots/" + self.out_filename)

        plt.clf()
        plt.close()
        print("Plot created.")

    #---------------------------------------------------
    def plotE_depProcessesHistogram(self, E_nu):
        info = self.event_energies[E_nu]
        E_a, E_d = info['E_avail'], info['E_dep']
        bin_width = E_nu / 100
        # bin_edges = np.linspace(0, 1.25*E_nu, 126)

        min_x, max_x = min(np.min(E_a), np.min(E_d)), max(np.max(E_a), np.max(E_d))
        midrange_x, halfrange_x = (min_x + max_x) / 2, (max_x - min_x) / 2
        xlim_min, xlim_max = midrange_x - (1.1 * halfrange_x), midrange_x + (1.1 * halfrange_x)
        nearest_greater_multiple = xlim_max + bin_width - (xlim_max % bin_width)
        bin_edges = np.linspace(0, nearest_greater_multiple, int(nearest_greater_multiple / bin_width) + 1)

        proc_masks = [
            ('QES', (info['interaction'] == 11)),
            ('RES', (info['interaction'] == 12)),
            ('DIS', (info['interaction'] == 13)),
            ('COH', (info['interaction'] == 14)),
            ('MEC', (info['interaction'] == 15)),
        ]

        fig, axs = plt.subplots(2, sharex=True)    # E_avail and E_dep will share an x-axis. Tick labels are removed from E_avail (upper Axes)
        fig.suptitle(f"E_nu = {E_nu} MeV, bin width = {bin_width} MeV, {info['nEvents']} events")
        fig.set_size_inches(6.8, 9.6)
        fig.set_layout_engine('tight')

        # Stacked filled histogram showing the breakdown of E_avail between the different processes #
        E_a_breakdown = [E_a[mask] for (proc, mask) in proc_masks]
        E_a_labels = [f"{proc}, {len(E_a_breakdown[i])} events total" for i, (proc, mask) in enumerate(proc_masks)]
        axs[0].hist(E_a_breakdown, bins=bin_edges, histtype="stepfilled", stacked=True, label=E_a_labels)
        axs[0].hist(E_a, bins=bin_edges, histtype="step", color="black", label="All events E_avail")
        axs[0].set_ylabel("No. of events")
        axs[0].set_title("E_avail")
        axs[0].legend(loc="upper left", fontsize="x-small")

        # Stacked filled histogram showing the breakdown of E_dep between the different processes #
        E_d_breakdown = [E_d[mask] for (proc, mask) in proc_masks]
        E_d_labels = [f"{proc}, {len(E_d_breakdown[i])} events total" for i, (proc, mask) in enumerate(proc_masks)]
        axs[1].hist(E_d_breakdown, bins=bin_edges, histtype="stepfilled", stacked=True, label=E_d_labels)
        axs[1].hist(E_d, bins=bin_edges, histtype="step", color="black", label="All events E_dep")
        axs[1].set_xlim(xlim_min, xlim_max)
        axs[1].set_xlabel("Energy [MeV]")
        axs[1].set_ylabel("No. of events")
        axs[1].set_title("E_dep")
        axs[1].legend(loc="upper left", fontsize="x-small")

        if len(self.event_energies.keys()) > 1:
            plt.savefig("./plots/" + self.out_filename[:-4] + "_" + str(int(E_nu)) + "MeV.pdf")
        else:
            plt.savefig("./plots/" + self.out_filename)

        plt.clf()
        plt.close()
        print("Plot created.")

    #---------------------------------------------------
    def plotE_depFSHistogram(self, E_nu):
        info = self.event_energies[E_nu]
        E_a, E_d, N_pl = info['E_avail'], info['E_dep'], info['N_par_list']
        N_pl_s = info['N_par_list_extra']
        #E_a, E_d, N_pl = info['E_avail_gst'], info['E_dep'], info['N_par_list_gst']
        bin_width = E_nu / 100
        # bin_edges = np.linspace(0, 1.25*E_nu, 126)

        min_x, max_x = min(np.min(E_a), np.min(E_d)), max(np.max(E_a), np.max(E_d))
        midrange_x, halfrange_x = (min_x + max_x) / 2, (max_x - min_x) / 2
        xlim_min, xlim_max = midrange_x - (1.1 * halfrange_x), midrange_x + (1.1 * halfrange_x)
        nearest_greater_multiple = xlim_max + bin_width - (xlim_max % bin_width)
        bin_edges = np.linspace(0, nearest_greater_multiple, int(nearest_greater_multiple / bin_width) + 1)

        # Pion production can happen w/out nucleon production in the case of COH scattering
                                # (electron,       proton,         neutron,        pi+-,           pi0,            lambda,          sigma0,          sigma-)
        FS_masks = [
            ('e|μ,Λ',            ((N_pl[0] > 0) & (N_pl[1]== 0) & (N_pl[2]== 0) & (N_pl[3]== 0) & (N_pl[4]== 0) & (N_pl_s[0]==1) & (N_pl_s[1]==0) & (N_pl_s[2]==0))),
            ('e|μ,Σ0',           ((N_pl[0] > 0) & (N_pl[1]== 0) & (N_pl[2]== 0) & (N_pl[3]== 0) & (N_pl[4]== 0) & (N_pl_s[0]==0) & (N_pl_s[1]==1) & (N_pl_s[2]==0))),
            ('e|μ,Σ-',           ((N_pl[0] > 0) & (N_pl[1]== 0) & (N_pl[2]== 0) & (N_pl[3]== 0) & (N_pl[4]== 0) & (N_pl_s[0]==0) & (N_pl_s[1]==0) & (N_pl_s[2]==1))),
            ('e|μ,p',            ((N_pl[0] > 0) & (N_pl[1] > 0) & (N_pl[2]== 0) & (N_pl[3]== 0) & (N_pl[4]== 0))),
            ('e|μ,p,pi0',        ((N_pl[0] > 0) & (N_pl[1] > 0) & (N_pl[2]== 0) & (N_pl[3]== 0) & (N_pl[4] > 0))),
            ('e|μ,p,pi+-',       ((N_pl[0] > 0) & (N_pl[1] > 0) & (N_pl[2]== 0) & (N_pl[3] > 0) & (N_pl[4]== 0))),
            ('e|μ,p,pi0,pi+-',   ((N_pl[0] > 0) & (N_pl[1] > 0) & (N_pl[2]== 0) & (N_pl[3] > 0) & (N_pl[4] > 0))),
            ('e|μ,n',            ((N_pl[0] > 0) & (N_pl[1]== 0) & (N_pl[2] > 0) & (N_pl[3]== 0) & (N_pl[4]== 0))),
            ('e|μ,n,pi0',        ((N_pl[0] > 0) & (N_pl[1]== 0) & (N_pl[2] > 0) & (N_pl[3]== 0) & (N_pl[4] > 0))),
            ('e|μ,n,pi+-',       ((N_pl[0] > 0) & (N_pl[1]== 0) & (N_pl[2] > 0) & (N_pl[3] > 0) & (N_pl[4]== 0))),
            ('e|μ,n,pi0,pi+-',   ((N_pl[0] > 0) & (N_pl[1]== 0) & (N_pl[2] > 0) & (N_pl[3] > 0) & (N_pl[4] > 0))),
            ('e|μ,p,n',          ((N_pl[0] > 0) & (N_pl[1] > 0) & (N_pl[2] > 0) & (N_pl[3]== 0) & (N_pl[4]== 0))),
            ('e|μ,p,n,pi0',      ((N_pl[0] > 0) & (N_pl[1] > 0) & (N_pl[2] > 0) & (N_pl[3]== 0) & (N_pl[4] > 0))),
            ('e|μ,p,n,pi+-',     ((N_pl[0] > 0) & (N_pl[1] > 0) & (N_pl[2] > 0) & (N_pl[3] > 0) & (N_pl[4]== 0))),
            ('e|μ,p,n,pi0,pi+-', ((N_pl[0] > 0) & (N_pl[1] > 0) & (N_pl[2] > 0) & (N_pl[3] > 0) & (N_pl[4] > 0))),
            ('e|μ,pi0',          ((N_pl[0] > 0) & (N_pl[1]== 0) & (N_pl[2]== 0) & (N_pl[3]== 0) & (N_pl[4] > 0))),
            ('e|μ,pi+-',         ((N_pl[0] > 0) & (N_pl[1]== 0) & (N_pl[2]== 0) & (N_pl[3] > 0) & (N_pl[4]== 0))),
        ]
        FS_colors = ('lightyellow', 'yellow', 'gold', 'lightcoral', 'coral', 'crimson', 'maroon', 'palegreen', 'yellowgreen', 'olive', 'darkolivegreen', 'lightskyblue', 'turquoise', 'dodgerblue', 'darkblue', 'violet', 'darkviolet')

        fig, axs = plt.subplots(2, sharex=True)    # E_avail and E_dep will share an x-axis. Tick labels are removed from E_avail (upper Axes)
        fig.suptitle(f"E_nu = {E_nu} MeV, bin width = {bin_width} MeV, {info['nEvents']} events.")
        fig.set_size_inches(6.8, 9.6)
        fig.set_layout_engine('tight')

        # Stacked filled histogram showing the breakdown of E_avail between the different final states (I hope I am using that term correctly)
        E_a_breakdown = [E_a[mask] for (FS, mask) in FS_masks]
        E_a_labels = [f"{FS} post-FSI, {len(E_a_breakdown[i])} events total" if len(E_a_breakdown[i]) > 0 else "" for i, (FS, mask) in enumerate(FS_masks)]
        axs[0].hist(E_a_breakdown, bins=bin_edges, histtype="stepfilled", stacked=True, color=FS_colors, label=E_a_labels)
        axs[0].hist(E_a, bins=bin_edges, histtype="step", color="black", label="All events E_avail")
        axs[0].set_ylabel("No. of events")
        axs[0].set_title("E_avail")
        axs[0].legend(loc="upper left", fontsize="x-small")

        # Stacked filled histogram showing the breakdown of E_dep between the different final states
        E_d_breakdown = [E_d[mask] for (FS, mask) in FS_masks]
        E_d_labels = [f"{FS} post-FSI, {len(E_d_breakdown[i])} events total" if len(E_d_breakdown[i]) > 0 else "" for i, (FS, mask) in enumerate(FS_masks)]
        axs[1].hist(E_d_breakdown, bins=bin_edges, histtype="stepfilled", stacked=True, color=FS_colors, label=E_d_labels)
        axs[1].hist(E_d, bins=bin_edges, histtype="step", color="black", label="All events E_dep")
        axs[1].set_xlim(xlim_min, xlim_max)
        # axs[1].set_xlim(150, 200) # For analysis
        axs[1].set_xlabel("Energy [MeV]")
        axs[1].set_ylabel("No. of events")
        axs[1].set_title("E_dep")
        axs[1].legend(loc="upper left", fontsize="x-small")

        if len(self.event_energies.keys()) > 1:
            plt.savefig("./plots/" + self.out_filename[:-4] + "_" + str(int(E_nu)) + "MeV.pdf")
        else:
            plt.savefig("./plots/" + self.out_filename)

        plt.clf()
        plt.close()
        print("Plot created.")

    #---------------------------------------------------
    def plotE_depTplgy(self, E_nu):
        info = self.event_energies[E_nu]
        N_pl = info["N_par_list"]
        N_pil = info["N_pi+-_list_gst"]
        E_a, E_d, Q_th, L = info["E_avail"], info["E_dep"], info["Q_thre"], info["L"]

        bin_width = E_nu / 100

        min_x, max_x = min(np.min(E_a), np.min(E_d), np.min(Q_th), np.min(L)), max(np.max(E_a), np.max(E_d), np.max(Q_th), np.max(L))
        midrange_x, halfrange_x = (min_x + max_x) / 2, (max_x - min_x) / 2
        xlim_min, xlim_max = midrange_x - (1.1 * halfrange_x), midrange_x + (1.1 * halfrange_x)
        nearest_greater_multiple = xlim_max + bin_width - (xlim_max % bin_width)
        bin_edges = np.linspace(0, nearest_greater_multiple, int(nearest_greater_multiple / bin_width) + 1)

        N_pl = np.concatenate( (N_pl[0:3], N_pil, N_pl[4:]), axis=0)
        # print("Rows of new N_par_list (this should be 9):", len(N_pl))

        # Assuming for now pi+ and pi- can be separated
        # Right now this is 'nue-centered'
        tplgies = [                           # e|μ, p,                n,            π+,  π-,  π0,  γ,   α,   etc
            ("1p0n0π",  np.array( [ (col ==    [1,   1,                0,            0,   0,   0,   0,   0,   0  ]).all() for col in N_pl.T ] ) ),
            ("1p1n0π",  np.array( [ (col ==    [1,   1,                1,            0,   0,   0,   0,   0,   0  ]).all() for col in N_pl.T ] ) ),
            ("1pXn0π",  np.array( [ (col[0:2]==[1,   1]).all()&(col[2]>0)&(col[3:]==[0,   0,   0,   0,   0,   0  ]).all() for col in N_pl.T ] ) ),
            ("2p0n0π",  np.array( [ (col ==    [1,   2,                0,            0,   0,   0,   0,   0,   0  ]).all() for col in N_pl.T ] ) ),
            ("2pXn0π",  np.array( [ (col[0:2]==[1,   1]).all()&(col[2]>0)&(col[3:]==[0,   0,   0,   0,   0,   0  ]).all() for col in N_pl.T ] ) ),
            ("1p0n1π+", np.array( [ (col ==    [1,   1,                0,            1,   0,   0,   0,   0,   0  ]).all() for col in N_pl.T ] ) )
        ]

        #fig, axs = plt.subplots(len(tplgies)+1, 2, figsize=(12.8, 4.8*(len(tplgies)+1)))
        fig, axs = plt.subplots(len(tplgies)+1, figsize=(12.8, 4.8*(len(tplgies)+1)))
        fig.set_layout_engine("tight")
        fig.suptitle(f"nu:{info['nu_pdg'][0]}, E_nu = {E_nu} MeV, bin width = {bin_width} MeV, {info['nEvents']} events", y=0.99)
        #for r, ax_row in enumerate(axs):
            #for c, ax in enumerate(ax_row):
            #    E = [E_a, E_d][c]
            #    if r > 0:
            #        ax.hist(E[tplgies[r-1][1]], bins=bin_edges, histtype="stepfilled", label=f"{tplgies[r-1][0]} evts ({len(E[tplgies[r-1][1]])}/{info['nEvents']}) {['E_avail', 'E_dep'][c]}")
            #    else:
            #        ax.hist(E,                  bins=bin_edges, histtype="step",       label=f"All evts {['E_avail', 'E_dep'][c]}")
            #    ax.set_xlim(xlim_min, xlim_max)
            #    ax.set_xlabel(["E_avail [MeV]", "E_dep [MeV"][c])
            #    ax.set_ylabel("No. of evts")
            #    ax.legend()
        for r, ax in enumerate(axs):
            if r > 0:
                 title = f"{tplgies[r-1][0]} evts ({len(E_a[tplgies[r-1][1]])}/{info['nEvents']})"
                 mask = tplgies[r-1][1]
            else:
                 title = "All evts"
                 mask = np.ones((len(E_a),), dtype=np.bool_)
            ax.set_title(title, loc='left')
            ax.hist(E_a[mask],  bins=bin_edges, histtype="step", label="E_avail")
            ax.hist(E_d[mask],  bins=bin_edges, histtype="step", label="E_dep")
            ax.hist(Q_th[mask], bins=bin_edges, histtype="step", label="Q (dQ > 75keV)")
            ax.hist(L[mask],    bins=bin_edges, histtype="step", label="L")

            ax.set_xlim(xlim_min, xlim_max)
            ax.set_xlabel("Energy [MeV]")
            ax.set_ylabel("No. of evts")
            ax.legend()

        self.SaveAndClose(E_nu)

    #---------------------------------------------------
    def plotE_depMoreInfo(self, E_nu, ep=False):
        info = self.event_energies[E_nu]
        N_pl = info["N_par_list"]
        E_a, E_d, Q_th, L = info["E_avail"], info["E_dep"], info["Q_thre"], info["L"]
        E_al, E_dl, E_al_dots = info["E_avail_list"], info["E_dep_list"], info['E_avail_list_dots']
        if ep:
            mask = np.array( [ (col[0]==1)&(col[1]==1)&(col[2:]==0).all() for col in N_pl.T ] )
        else:
            mask = np.ones((info['nEvents'],), dtype=np.bool_)

        bin_width = E_nu / 100

        min_x, max_x = min( np.concatenate( (E_a[mask], E_d[mask], Q_th[mask], L[mask]) ) ), max( np.concatenate( (E_a[mask], E_d[mask], Q_th[mask], L[mask]) ) )
        midrange_x, halfrange_x = (min_x + max_x) / 2, (max_x - min_x) / 2
        xlim_min, xlim_max = midrange_x - (1.1 * halfrange_x), midrange_x + (1.1 * halfrange_x)
        nearest_greater_multiple = xlim_max + bin_width - (xlim_max % bin_width)
        bin_edges = np.linspace(0, nearest_greater_multiple, int(nearest_greater_multiple / bin_width) + 1)

        #fig, axs = plt.subplots(len(tplgies)+1, 2, figsize=(12.8, 4.8*(len(tplgies)+1)))
        fig, axs = plt.subplots(6, figsize=(12.8, 4.8*5))
        fig.set_layout_engine("tight")
        if ep:
            fig.suptitle(f"nu:{info['nu_pdg'][0]}, E_nu = {E_nu} MeV, bin width = {bin_width} MeV, {np.sum(mask)} / {info['nEvents']} 1p0n0pi events", y=0.99)
        else:
            fig.suptitle(f"nu:{info['nu_pdg'][0]}, E_nu = {E_nu} MeV, bin width = {bin_width} MeV, {info['nEvents']} events", y=0.99)

        # Overall energy distributions
        axs[0].set_title("Considering all final state particles", loc='left')
        axs[0].hist(E_a[mask],  bins=bin_edges, histtype="step", label="E_avail")
        axs[0].hist(E_d[mask],  bins=bin_edges, histtype="step", label="E_dep")
        L_res = EventsTreeReader.calculateRes(info['L'][mask])
        Q_th_res = EventsTreeReader.calculateRes(info['Q_thre'][mask])
        axs[0].hist(Q_th[mask], bins=bin_edges, histtype="step", label=f"Q (dQ > 75keV). 'Res'={Q_th_res*100:.1f}%")
        axs[0].hist(L[mask],    bins=bin_edges, histtype="step", label=f"L (180PEpMeV). 'Res'={L_res*100:.1f}%")
        # Hadronic component energy distributions
        axs[1].set_title("Hadronic component", loc='left')
        axs[1].hist(E_al[[1, 2, 3, 6, 7]].sum(axis=0)[mask], bins=bin_edges, histtype="step", label="E_avail")
        axs[1].hist(E_dl[[1, 2, 3, 6, 7]].sum(axis=0)[mask], bins=bin_edges, histtype="step", label="E_dep")
        axs[1].hist(info['Q_h'][mask],                       bins=bin_edges, histtype="step", label="Q (dQ > 75keV)")
        axs[1].hist(info['L_h'][mask],                       bins=bin_edges, histtype="step", label="L (180PEpMeV)")
        # EM component energy distributions
        axs[2].set_title("EM component", loc='left')
        axs[2].hist(E_al[[0, 4, 5]].sum(axis=0)[mask], bins=bin_edges, histtype="step", label="E_avail")
        axs[2].hist(E_dl[[0, 4, 5]].sum(axis=0)[mask], bins=bin_edges, histtype="step", label="E_dep")
        axs[2].hist(info['Q_e'][mask],                 bins=bin_edges, histtype="step", label="Q (dQ > 75keV)")
        axs[2].hist(info['L_e'][mask],                 bins=bin_edges, histtype="step", label="L (180PEpMeV)")
        # Charge calorimetric responses
        rbw = 0.01
        rbe = np.linspace(0, 1.1, int(1.1 / 0.01) + 1)
        axs[3].set_title("Charge calorimetric responses", loc='left')
        RcQ = (Q_th / E_a)[mask]
        RcQe = (info['Q_e'] / E_al[[0, 4, 5]].sum(axis=0))[mask]
        RcQh = (info['Q_h'] / E_al[[1, 2, 3, 6, 7]].sum(axis=0))[mask]
        RcQed = (info['Q_e_dots'] / E_al_dots[[0, 4, 5]].sum(axis=0))[mask & (info['Q_e_dots'] != 0)]
        RcQhd = (info['Q_h_dots'] / E_al_dots[[1, 2, 3, 6, 7]].sum(axis=0))[mask & (info['Q_h_dots'] != 0)]

        RcQ_bh, _,  RcQ_art  = axs[3].hist(  RcQ,   bins=rbe, histtype="step", label="Overall R_cal_charge")
        RcQe_bh, _, RcQe_art = axs[3].hist(  RcQe,  bins=rbe, histtype="step", label="EM R_cal_charge")
        RcQh_bh, _, RcQh_art = axs[3].hist(  RcQh,  bins=rbe, histtype="step", label="Hadronic R_cal_charge")
        RcQed_bh, _, RcQed_art = axs[3].hist(RcQed, bins=rbe, histtype="step", label=f"EM R_cal_charge_dots. {len(RcQed)} evts w/ Q_e_dots != 0")
        RcQhd_bh, _, RcQhd_art = axs[3].hist(RcQhd, bins=rbe, histtype="step", label=f"Hadronic R_cal_charge_dots. {len(RcQhd)} evts w/ Q_h_dots != 0")

        RcQ_peak   = EventsTreeReader.calculateMeanValInPeak(data=RcQ,   bin_heights=RcQ_bh,   bin_width=rbw, bin_left_limit=0)
        RcQe_peak  = EventsTreeReader.calculateMeanValInPeak(data=RcQe,  bin_heights=RcQe_bh,  bin_width=rbw, bin_left_limit=0)
        RcQh_peak  = EventsTreeReader.calculateMeanValInPeak(data=RcQh,  bin_heights=RcQh_bh,  bin_width=rbw, bin_left_limit=0)
        RcQed_peak = EventsTreeReader.calculateMeanValInPeak(data=RcQed, bin_heights=RcQed_bh, bin_width=rbw, bin_left_limit=0)
        RcQhd_peak = EventsTreeReader.calculateMeanValInPeak(data=RcQhd, bin_heights=RcQhd_bh, bin_width=rbw, bin_left_limit=0)
        RcQ_art[0].set_label(  RcQ_art[0].get_label()   + f" Peak value = {RcQ_peak:.2f}")
        RcQe_art[0].set_label( RcQe_art[0].get_label()  + f" Peak value = {RcQe_peak:.2f}")
        RcQh_art[0].set_label( RcQh_art[0].get_label()  + f" Peak value = {RcQh_peak:.2f}")
        RcQed_art[0].set_label(RcQed_art[0].get_label() + f" Peak value = {RcQed_peak:.2f}")
        RcQhd_art[0].set_label(RcQhd_art[0].get_label() + f" Peak value = {RcQhd_peak:.2f}")

        # Light calorimetric responses
        axs[4].set_title("Light calorimetric responses", loc='left')
        RcL  = (L / E_a)[mask]
        RcLe = (info['L_e'] / E_al[[0, 4, 5]].sum(axis=0))[mask]
        RcLh = (info['L_h'] / E_al[[1, 2, 3, 6, 7]].sum(axis=0))[mask]
        RcL_bh, _,  RcL_art  = axs[4].hist(RcL,  bins=rbe, histtype="step", label="Overall R_cal_light")
        RcLe_bh, _, RcLe_art = axs[4].hist(RcLe, bins=rbe, histtype="step", label="EM R_cal_light")
        RcLh_bh, _, RcLh_art = axs[4].hist(RcLh, bins=rbe, histtype="step", label="Hadronic R_cal_light")
        RcL_peak   = EventsTreeReader.calculateMeanValInPeak(data=RcL,   bin_heights=RcL_bh,   bin_width=rbw, bin_left_limit=0)
        RcLe_peak  = EventsTreeReader.calculateMeanValInPeak(data=RcLe,  bin_heights=RcLe_bh,  bin_width=rbw, bin_left_limit=0)
        RcLh_peak  = EventsTreeReader.calculateMeanValInPeak(data=RcLh,  bin_heights=RcLh_bh,  bin_width=rbw, bin_left_limit=0)
        RcL_art[0].set_label(  RcL_art[0].get_label()   + f" Peak value = {RcL_peak:.2f}")
        RcLe_art[0].set_label( RcLe_art[0].get_label()  + f" Peak value = {RcLe_peak:.2f}")
        RcLh_art[0].set_label( RcLh_art[0].get_label()  + f" Peak value = {RcLh_peak:.2f}")

        # Ideal Energy Reconstruction
        scaling_factors = {"L1":        round(RcL_peak,   2),
		           "Q1":        round(RcQ_peak,   2),
                           "Q2_e":      round(RcQe_peak,  2),
                           "Q2_h":      round(RcQh_peak,  2),
                           "Q3_e_dots": round(RcQed_peak, 2),
                           "Q3_h_dots": round(RcQhd_peak, 2)}
        #E_a_to_E_nu_shift = 29     # We don't have that much precision anyways.
        axs[5].set_title(f"Ideal case energy reconstruction. Shift: {self.E_a_to_E_nu_shift} MeV", loc='left')
        L1 = (info['L'] / scaling_factors["L1"])                                               + self.E_a_to_E_nu_shift
        Q1 = (info['Q_thre'] / scaling_factors["Q1"])                                          + self.E_a_to_E_nu_shift
        Q2 = (info['Q_e'] / scaling_factors["Q2_e"]) + (info['Q_h'] / scaling_factors["Q2_h"]) + self.E_a_to_E_nu_shift
        Q3 = info['E_dep_tracks'] + (info['Q_e_dots'] / scaling_factors["Q3_e_dots"]) + (info['Q_h_dots'] / scaling_factors["Q3_h_dots"]) + self.E_a_to_E_nu_shift

        label_l = ["L1", "Q1",    "Q2", "Q3"  ]
        color_l = ["C3", "black", "C2", "blue"]
        rec_l = [L1, Q1, Q2, Q3]
        resolutions = {}
        biases = {}
        for i, E_rec in enumerate(rec_l):
            resolutions[label_l[i]] = EventsTreeReader.calculateRes(E_rec)
            biases[label_l[i]] = EventsTreeReader.calculateBias(E_rec, E_nu)
            axs[5].hist(E_rec[mask], bins=bin_edges, histtype="step", color=color_l[i], label=label_l[i] + f". Resolution:{resolutions[label_l[i]]*100:.1f}%. R.Bias:{biases[label_l[i]]*100:.1f}%")
        axs[5].axvline(E_nu, color="indigo", ls="--", alpha=0.4, label="Actual E_nu")

        for ax in axs[[0,1,2,5]]:
            ax.set_xlim(0, xlim_max)
            ax.set_xlabel("Energy [MeV]")
            ax.set_ylabel("No. of evts")
            ax.legend(fontsize="small")
        for ax in axs[[3,4]]:
            ax.set_xlabel("Calorimeter response. Bin width = 0.01")
            ax.set_ylabel("No. of evts")
            ax.legend(fontsize="small")

        self.SaveAndClose(E_nu)
        return resolutions, biases, L_res, Q_th_res

#---------------------------------------------------
    def plotE_depComponentsSeparately(self):

        E_nu_vals = np.array(list(self.event_energies.keys()))
        E_nu_ct = len(E_nu_vals)

        variables = ["All_particles", "EM", "Hadronic", "Dots", "Tracks"]

        figs = []
        axs = []

        for variable in variables:
            subplots = plt.subplots(E_nu_ct, figsize=(9.6, 4*E_nu_ct))
            figs.append(subplots[0])
            axs.append(subplots[1])
            figs[-1].set_layout_engine("tight")
            figs[-1].suptitle(f"nu:{self.event_energies[E_nu_vals[0]]['nu_pdg'][0]}, E_nu = {E_nu_vals[0]} to {E_nu_vals[-1]} MeV. {variable}", y=0.995)

        for i, E_nu in enumerate(E_nu_vals):
            info = self.event_energies[E_nu]
            E_a, E_d, Q_th, L = info["E_avail"], info["E_dep"], info["Q_thre"], info["L"]
            Q_e, Q_h = info["Q_e"], info["Q_h"]
            Q_e_dots, Q_h_dots, E_d_trck = info["Q_e_dots"], info["Q_h_dots"], info["E_dep_tracks"]
            E_al, E_dl, E_al_dots = info["E_avail_list"], info["E_dep_list"], info['E_avail_list_dots']

            bin_width = E_nu / 100

            # All particles
            min_x, max_x = min( np.concatenate( (E_a, E_d, Q_th, L) ) ), max( np.concatenate( (E_a, E_d, Q_th, L) ) )
            bin_edges = EventsTreeReader.calculateBinEdges(min_x, max_x, bin_width)
            axs[0][i].hist(E_a,  bins=bin_edges, histtype="step", label="E_avail")
            axs[0][i].hist(E_d,  bins=bin_edges, histtype="step", label="E_dep")
            L_res = EventsTreeReader.calculateRes(L)
            Q_th_res = EventsTreeReader.calculateRes(Q_th)
            axs[0][i].hist(Q_th, bins=bin_edges, histtype="step", label=f"Q (dQ > 75keV). 'Res'={Q_th_res*100:.1f}%")
            axs[0][i].hist(L,    bins=bin_edges, histtype="step", label=f"L (180PEpMeV). 'Res'={L_res*100:.1f}%")

            # Hadronic component energy distributions
            E_ah, E_dh = E_al[[1, 2, 3, 6, 7]].sum(axis=0), E_dl[[1, 2, 3, 6, 7]].sum(axis=0)
            min_x, max_x = min( np.concatenate( (E_ah, E_dh, Q_h) ) ), max( np.concatenate( (E_ah, E_dh, Q_h) ) )
            bin_edges = EventsTreeReader.calculateBinEdges(min_x, max_x, bin_width)
            axs[1][i].hist(E_ah, bins=bin_edges, histtype="step", label="E_avail")
            axs[1][i].hist(E_dh, bins=bin_edges, histtype="step", label="E_dep")
            axs[1][i].hist(Q_h,  bins=bin_edges, histtype="step", label="Q (dQ > 75keV)")

            # EM component energy distributions
            E_ae, E_de = E_al[[0, 4, 5]].sum(axis=0), E_dl[[0, 4, 5]].sum(axis=0)
            min_x, max_x = min( np.concatenate( (E_ae, E_de, Q_e) ) ), max( np.concatenate( (E_ae, E_de, Q_e) ) )
            bin_edges = EventsTreeReader.calculateBinEdges(min_x, max_x, bin_width)
            axs[2][i].hist(E_ae, bins=bin_edges, histtype="step", label="E_avail")
            axs[2][i].hist(E_de, bins=bin_edges, histtype="step", label="E_dep")
            axs[2][i].hist(Q_e,  bins=bin_edges, histtype="step", label="Q (dQ > 75keV)")
        
            # Dots
            E_ae_d = E_al_dots[[0, 4, 5]].sum(axis=0)
            E_ah_d = E_al_dots[[1, 2, 3, 6, 7]].sum(axis=0)
            min_x, max_x = min( np.concatenate( (E_ae_d, E_ah_d, Q_e_dots, Q_h_dots) ) ), max( np.concatenate( (E_ae_d, E_ah_d, Q_e_dots, Q_h_dots) ) )
            bin_edges = EventsTreeReader.calculateBinEdges(min_x, max_x, bin_width)
            axs[3][i].hist(E_ae_d, bins=bin_edges, histtype="step", label="E_avail EM dots")
            axs[3][i].hist(E_ah_d, bins=bin_edges, histtype="step", label="E_avail Hadronic dots")
            axs[3][i].hist(Q_e_dots, bins=bin_edges, histtype="step", label="Q EM dots")
            axs[3][i].hist(Q_h_dots, bins=bin_edges, histtype="step", label="Q Hadronic dots")
            
            # Tracks
            axs[4][i].hist(E_d_trck, bins=bin_edges, histtype="step", label="E_dep tracks")       

            for j in range(len(axs)):
                axs[j][i].set_title(f"E_nu = {E_nu} MeV", fontsize="small")
                axs[j][i].legend(fontsize="small")

        for fig, variable in zip(figs, variables): 
            fig.savefig("./plots/" + self.out_filename[:-4] + "_" + variable + ".pdf")
            fig.clf()
            print("Plots created.")


#---------------------------------------------------
    def plotMissingEnergy(self):

        E_nu_vals = np.array(list(self.event_energies.keys()))
        E_nu_ct = len(E_nu_vals)

        variables = ["Q", "L", "Hadronic_and_EM_L", "E_dep"]

        figs = []
        axs = []

        for variable in variables:
            subplots = plt.subplots(E_nu_ct, 2 if variable != "E_dep" else 1, figsize=(6.4*(2 if variable != "E_dep" else 1), 4.8*E_nu_ct))
            figs.append(subplots[0])
            axs.append(subplots[1])
            figs[-1].set_layout_engine("tight")
            figs[-1].suptitle(f"nu:{self.event_energies[E_nu_vals[0]]['nu_pdg'][0]}, E_nu = {E_nu_vals[0]} to {E_nu_vals[-1]} MeV. {variable}", y=0.995)

        for i, E_nu in enumerate(E_nu_vals):
            info = self.event_energies[E_nu]

            Q_75, Q_500 = info["Q_thre"], info["Q_500"]
            L_180, L_35 = info["L"], info["L_35"]

            L_e, L_h = info["L_e"], info["L_h"]

            E_d = info["E_dep"]

            bin_width = E_nu / 100

            N_pl = info["N_par_list"]
            n_no_chpi = (N_pl[2]>0)&(N_pl[3]==0)
            chpi_no_n = (N_pl[2]==0)&(N_pl[3]>0)
            chpi_n = (N_pl[2]>0)&(N_pl[3]>0)
            no_chpi_no_n = (N_pl[2]==0)&(N_pl[3]==0)

            labels = [r"No $n$ or $\pi\pm$", r"$n$, no $\pi\pm$", r"$\pi\pm$, no n", r"Both $n$ and $\pi\pm$"]
            masks = [no_chpi_no_n, n_no_chpi, chpi_no_n, chpi_n]

            min_x, max_x = min(Q_75), max(Q_75)
            bin_edges = EventsTreeReader.calculateBinEdges(min_x, max_x, bin_width)
            axs[0][i, 0].set_title(f"E_nu = {E_nu} MeV. dQ > 75 keV threshold. {len(Q_75)} evts. Bin width = {bin_width} MeV.", fontsize="small")
            axs[0][i, 0].hist(Q_75, bins=bin_edges, histtype="step", label="All events")
            axs[0][i, 0].hist(Q_75[no_chpi_no_n], bins=bin_edges, histtype="step", label=r"No $n$ or $\pi\pm$")
            axs[0][i, 0].hist(Q_75[n_no_chpi], bins=bin_edges, histtype="step", label=r"$n$, no $\pi\pm$")
            axs[0][i, 0].hist(Q_75[chpi_no_n], bins=bin_edges, histtype="step", label=r"$\pi\pm$, no n")
            axs[0][i, 0].hist(Q_75[chpi_n], bins=bin_edges, histtype="step", label=r"Both $n$ and $\pi\pm$")
            axs[0][i, 0].set_xlabel("$Q$ ($dQ>75$keV) [MeV]")

            min_x, max_x = min(Q_500), max(Q_500)
            bin_edges = EventsTreeReader.calculateBinEdges(min_x, max_x, bin_width)
            axs[0][i, 1].set_title(f"E_nu = {E_nu} MeV. dQ > 500 keV threshold. {len(Q_500)} evts. Bin width = {bin_width} MeV.", fontsize="small")
            axs[0][i, 1].hist(Q_500, bins=bin_edges, histtype="step", label="All events")
            axs[0][i, 1].hist(Q_500[no_chpi_no_n], bins=bin_edges, histtype="step", label=r"No $n$ or $\pi\pm$")
            axs[0][i, 1].hist(Q_500[n_no_chpi], bins=bin_edges, histtype="step", label=r"$n$, no $\pi\pm$")
            axs[0][i, 1].hist(Q_500[chpi_no_n], bins=bin_edges, histtype="step", label=r"$\pi\pm$, no $n$")
            axs[0][i, 1].hist(Q_500[chpi_n], bins=bin_edges, histtype="step", label=r"Both $n$ and $\pi\pm$")
            axs[0][i, 1].set_xlabel("$Q$ ($dQ>500$keV) [MeV]")

            min_x, max_x = min(L_35), max(L_35)
            bin_edges = EventsTreeReader.calculateBinEdges(min_x, max_x, bin_width)
            axs[1][i, 0].set_title(f"E_nu = {E_nu} MeV. 35 PE/MeV PCE. {len(L_35)} evts. Bin width = {bin_width} MeV.", fontsize="small")
            axs[1][i, 0].hist(L_35, bins=bin_edges, histtype="step", label="All events")
            axs[1][i, 0].hist(L_35[no_chpi_no_n], bins=bin_edges, histtype="step", label=r"No $n$ or $\pi\pm$")
            axs[1][i, 0].hist(L_35[n_no_chpi], bins=bin_edges, histtype="step", label=r"$n$, no $\pi\pm$")
            axs[1][i, 0].hist(L_35[chpi_no_n], bins=bin_edges, histtype="step", label=r"$\pi\pm$, no $n$")
            axs[1][i, 0].hist(L_35[chpi_n], bins=bin_edges, histtype="step", label=r"Both $n$ and $\pi\pm$")
            axs[1][i, 0].set_xlabel("$L$ (35 PE/MeV PCE) [MeV]")
            
            min_x, max_x = min(L_180), max(L_180)
            bin_edges = EventsTreeReader.calculateBinEdges(min_x, max_x, bin_width)
            axs[1][i, 1].set_title(f"E_nu = {E_nu} MeV. 180 PE/MeV PCE. {len(L_180)} evts. Bin width = {bin_width} MeV.", fontsize="small")
            axs[1][i, 1].hist(L_180, bins=bin_edges, histtype="step", label="All events")
            axs[1][i, 1].hist(L_180[no_chpi_no_n], bins=bin_edges, histtype="step", label=r"No $n$ or $\pi\pm$")
            axs[1][i, 1].hist(L_180[n_no_chpi], bins=bin_edges, histtype="step", label=r"$n$, no $\pi\pm$")
            axs[1][i, 1].hist(L_180[chpi_no_n], bins=bin_edges, histtype="step", label=r"$\pi\pm$, no $n$")
            axs[1][i, 1].hist(L_180[chpi_n], bins=bin_edges, histtype="step", label=r"Both $n$ and $\pi\pm$")
            axs[1][i, 1].set_xlabel("$L$ (180 PE/MeV PCE) [MeV]")

            min_x, max_x = min(L_h), max(L_h)
            bin_edges = EventsTreeReader.calculateBinEdges(min_x, max_x, bin_width)
            axs[2][i, 0].set_title(f"E_nu = {E_nu} MeV. 180 PE/MeV PCE. {len(L_h)} evts. Bin width = {bin_width} MeV.", fontsize="small")
            axs[2][i, 0].hist(L_h, bins=bin_edges, histtype="step", label="All events")
            axs[2][i, 0].hist(L_h[no_chpi_no_n], bins=bin_edges, histtype="step", label=r"No $n$ or $\pi\pm$")
            axs[2][i, 0].hist(L_h[n_no_chpi], bins=bin_edges, histtype="step", label=r"$n$, no $\pi\pm$")
            axs[2][i, 0].hist(L_h[chpi_no_n], bins=bin_edges, histtype="step", label=r"$\pi\pm$, no $n$")
            axs[2][i, 0].hist(L_h[chpi_n], bins=bin_edges, histtype="step", label=r"Both $n$ and $\pi\pm$")
            axs[2][i, 0].set_xlabel("Hadronic $L$ (180 PE/MeV PCE) [MeV]")

            min_x, max_x = min(L_e), max(L_e)
            bin_edges = EventsTreeReader.calculateBinEdges(min_x, max_x, bin_width)
            axs[2][i, 1].set_title(f"E_nu = {E_nu} MeV. 180 PE/MeV PCE. {len(L_e)} evts. Bin width = {bin_width} MeV.", fontsize="small")
            axs[2][i, 1].hist(L_e, bins=bin_edges, histtype="step", label="All events")
            axs[2][i, 1].hist(L_e[no_chpi_no_n], bins=bin_edges, histtype="step", label=r"No $n$ or $\pi\pm$")
            axs[2][i, 1].hist(L_e[n_no_chpi], bins=bin_edges, histtype="step", label=r"$n$, no $\pi\pm$")
            axs[2][i, 1].hist(L_e[chpi_no_n], bins=bin_edges, histtype="step", label=r"$\pi\pm$, no $n$")
            axs[2][i, 1].hist(L_e[chpi_n], bins=bin_edges, histtype="step", label=r"Both $n$ and $\pi\pm$")
            axs[2][i, 1].set_xlabel("EM $L$ (180 PE/MeV PCE) [MeV]")

            min_x, max_x = min(E_d), max(E_d)
            bin_edges = EventsTreeReader.calculateBinEdges(min_x, max_x, bin_width)
            axs[3][i].set_title(f"E_nu = {E_nu} MeV. {len(E_d)} evts. Bin width = {bin_width} MeV.", fontsize="small")
            axs[3][i].hist(E_d, bins=bin_edges, histtype="step", label="All events", lw=2)
            axs[3][i].hist([E_d[mask] for mask in masks], bins=bin_edges, histtype="stepfilled", stacked=True, label=labels, lw=2)
            # axs[3][i].hist(E_d[no_chpi_no_n], bins=bin_edges, histtype="step", label=r"No $n$ or $\pi\pm$", lw=2)
            # axs[3][i].hist(E_d[n_no_chpi], bins=bin_edges, histtype="step", label=r"$n$, no $\pi\pm$", lw=2)
            # axs[3][i].hist(E_d[chpi_no_n], bins=bin_edges, histtype="step", label=r"$\pi\pm$, no $n$", lw=2)
            # axs[3][i].hist(E_d[chpi_n], bins=bin_edges, histtype="step", label=r"Both $n$ and $\pi\pm$", lw=2)
            axs[3][i].set_xlabel(r"$E_\text{dep}$ [MeV]", fontsize="large")
            axs[3][i].set_ylabel('Number of events', fontsize="large")
            axs[3][i].legend(fontsize="large", loc="upper left")

            for j in range(len(axs)-1):
                axs[j][i, 0].legend(fontsize="medium")
                axs[j][i, 1].legend(fontsize="medium")
                axs[0][i, 0].set_ylabel('Number of events')
                axs[0][i, 1].set_ylabel('Number of events')

        for fig, variable in zip(figs, variables): 
            fig.savefig("./plots/" + self.out_filename[:-4] + "_" + variable + ".pdf")
            fig.clf()
            print("Plots created.")

    # ------ Q3 analysis ------ #
    def modified_Q3(self):
        # Exclude primary protons(+desc) from tracks
        E_dep_tracks_wout_proton = np.zeros(10000)
        E_avail_EM_dots = np.zeros(10000)
        E_avail_had_dots_and_proton = np.zeros(10000)
        Q_dep_EM_dots = np.zeros(10000)
        Q_dep_had_dots_and_proton = np.zeros(10000)

        # Exclude primary electron(+desc) from tracks
        E_dep_tracks_wout_electron = np.zeros(10000)
        E_avail_EM_dots_and_electron = np.zeros(10000)
        E_avail_had_dots = np.zeros(10000)
        Q_dep_EM_dots_and_electron = np.zeros(10000)
        Q_dep_had_dots = np.zeros(10000)

        E_dep_tracks_list_allE = np.zeros((8, 10000))
        Q_dep_dots_list_allE = np.zeros((8, 10000))

        for i, E_nu in enumerate(self.event_energies.keys()):
            info = self.event_energies[E_nu]

            # Exclude primary protons(+desc) from tracks.
            E_dep_tracks_wout_proton[1000*i:1000*(i+1)] = info['E_dep_tracks_list'][[0,2,3,4,5,6,7]].sum(axis=0)
            
            E_avail_EM_dots[1000*i:1000*(i+1)] = info['E_avail_list_dots'][[0,4,5]].sum(axis=0)
            E_avail_had_dots_and_proton[1000*i:1000*(i+1)] = (info['E_avail_list_dots'][[2, 3, 6, 7]].sum(axis=0) + info['E_avail_list'][1]) # We don't want to double count available energy of dot-like depositions attr. to protons

            Q_dep_EM_dots[1000*i:1000*(i+1)] = info['Q_e_dots']
            Q_dep_had_dots_and_proton[1000*i:1000*(i+1)] = (info['Q_dep_list_dots_th75keV'][[2, 3, 6, 7]].sum(axis=0) + info['Q_dep_list_th75keV'][1]) # We don't want to double count Qdep of dot-like depositions attr. to protons

            # Exclude primary electron(+desc) from tracks.
            E_dep_tracks_wout_electron[1000*i:1000*(i+1)] = info['E_dep_tracks_list'][1:].sum(axis=0)
            
            E_avail_had_dots[1000*i:1000*(i+1)] = info['E_avail_list_dots'][[1,2,3,6,7]].sum(axis=0)
            E_avail_EM_dots_and_electron[1000*i:1000*(i+1)] = (info['E_avail_list_dots'][[4,5]].sum(axis=0) + info['E_avail_list'][0])

            Q_dep_had_dots[1000*i:1000*(i+1)] = info['Q_h_dots']
            Q_dep_EM_dots_and_electron[1000*i:1000*(i+1)] = (info['Q_dep_list_dots_th75keV'][[4,5]].sum(axis=0) + info['Q_dep_list_th75keV'][0])

            # For stats
            E_dep_tracks_list_allE[:, 1000*i:1000*(i+1)] = info['E_dep_tracks_list']
            Q_dep_dots_list_allE[:, 1000*i:1000*(i+1)] = info['Q_dep_list_dots_th75keV']

        # Proton exclusion
        R_cal_charge_e_dots = Q_dep_EM_dots / E_avail_EM_dots
        R_cal_charge_h_dots_and_proton = Q_dep_had_dots_and_proton / E_avail_had_dots_and_proton
        # Electron exclusion
        R_cal_charge_h_dots = Q_dep_had_dots / E_avail_had_dots
        R_cal_charge_e_dots_and_electron = Q_dep_EM_dots_and_electron / E_avail_EM_dots_and_electron

        bin_width = 0.01
        bin_edges = np.linspace(0, 1.5, 151)         

        # Proton exclusion
        R_c_e_dots_bh, _ = np.histogram(R_cal_charge_e_dots, bins=bin_edges)
        R_c_h_dots_p_bh, _ = np.histogram(R_cal_charge_h_dots_and_proton, bins=bin_edges)
        Q_EM_dots_scaling_factor = EventsTreeReader.calculateMeanValInPeak(R_cal_charge_e_dots, R_c_e_dots_bh, bin_width)  
        print(f"{Q_EM_dots_scaling_factor=}")                  
        Q_had_dots_and_proton_scaling_factor = EventsTreeReader.calculateMeanValInPeak(R_cal_charge_h_dots_and_proton, R_c_h_dots_p_bh, bin_width)                    
        # Electron exclusion
        R_c_h_dots_bh, _ = np.histogram(R_cal_charge_h_dots[R_cal_charge_h_dots!=0], bins=bin_edges) # Over 1000 events with Q_had_dots=0
        R_c_e_dots_e_bh, _ = np.histogram(R_cal_charge_e_dots_and_electron, bins=bin_edges)
        Q_had_dots_scaling_factor = EventsTreeReader.calculateMeanValInPeak(R_cal_charge_h_dots, R_c_h_dots_bh, bin_width)                    
        Q_EM_dots_and_electron_scaling_factor = EventsTreeReader.calculateMeanValInPeak(R_cal_charge_e_dots_and_electron, R_c_e_dots_e_bh, bin_width)                    

        modified_Q3_res = np.zeros(len(self.event_energies.keys()))
        for i, E_nu in enumerate(self.event_energies.keys()): 
            info = self.event_energies[E_nu]
            E_avail = info['E_avail']
            Q3_E_rec = info['E_rec_Q3']
            Q2_E_rec = info['E_rec_Q2']

            #8000:9000 would be 900 MeV, 9000:10000 would be 1000 MeV. (0:1000 is 100 MeV)
            modified_Q3_E_rec = E_dep_tracks_wout_proton[1000*i:1000*(i+1)] + (Q_dep_EM_dots[1000*i:1000*(i+1)] / Q_EM_dots_scaling_factor) + (Q_dep_had_dots_and_proton[1000*i:1000*(i+1)] / Q_had_dots_and_proton_scaling_factor)
            modified_Q3_res[i] = EventsTreeReader.calculateRes(modified_Q3_E_rec)
            
            modified_Q3_E_rec2 = E_dep_tracks_wout_electron[1000*i:1000*(i+1)] + (Q_dep_had_dots[1000*i:1000*(i+1)] / Q_had_dots_scaling_factor) + (Q_dep_EM_dots_and_electron[1000*i:1000*(i+1)] / Q_EM_dots_and_electron_scaling_factor)

            # ___ To plot and compare E_rec distributions obtained by the modified Q3 ___
            # plt.title(f"nu: {info['nu_pdg'][0]}. E_nu = {E_nu} MeV. 1000 events. Bin width = {E_nu / 100} MeV")
            # bin_width = E_nu / 100
            # min_x, max_x = min( np.concatenate( (E_avail, Q3_E_rec, Q2_E_rec, modified_Q3_E_rec) ) ), max( np.concatenate( (E_avail, Q3_E_rec, Q2_E_rec, modified_Q3_E_rec) ) )
            # bin_edges = EventsTreeReader.calculateBinEdges(min_x, max_x, bin_width)
            
            # plt.hist(E_avail, bins=bin_edges, histtype="step", label="E_avail")
            # plt.hist(Q3_E_rec, bins=bin_edges, histtype="step", label="E_rec by Q3")
            # plt.hist(Q2_E_rec, bins=bin_edges, histtype="step", label="E_rec by Q2")
            # plt.hist(modified_Q3_E_rec, bins=bin_edges, histtype="step", label="E_rec by modified Q3 where primary proton tracks' E_dep are not reconstructable")
            # # plt.hist(modified_Q3_E_rec2, bins=bin_edges, histtype="step", label="E_rec by modified Q3 where primary electron tracks' E_dep are not reconstructable")
            # plt.xlabel("Energy [MeV]")
            # plt.ylabel("Number of events")
            # plt.legend(fontsize="small")
            # plt.savefig("./plots/" + self.out_filename[:-4] + f"_{int(E_nu)}MeV.pdf")
            # plt.close()
        
        plt.figure()
        # plt.title(f"nu: {info['nu_pdg'][0]}. 10000 events. Resolutions.")
        plt.plot(self.event_energies.keys(), np.array([self.resolutions['E_rec_L1_res'][E_nu] for E_nu in self.event_energies.keys()]), color="red", label="L1", alpha=0.5)
        plt.plot(self.event_energies.keys(), np.array([self.resolutions['E_rec_Q1_res'][E_nu] for E_nu in self.event_energies.keys()]), color="black", label="Q1", alpha=0.5)
        plt.plot(self.event_energies.keys(), np.array([self.resolutions['E_rec_(Q+L)1_res'][E_nu] for E_nu in self.event_energies.keys()]), color="magenta", label="Q+L", alpha=0.5)
        plt.plot(self.event_energies.keys(), np.array([self.resolutions['E_rec_Q2_res'][E_nu] for E_nu in self.event_energies.keys()]), color="green", label="Q2")
        plt.plot(self.event_energies.keys(), np.array([self.resolutions['E_rec_Q3_res'][E_nu] for E_nu in self.event_energies.keys()]), color="blue", label="Q3")
        plt.plot(self.event_energies.keys(), modified_Q3_res, color="purple", label="Q3 without proton track reconstruction")
        plt.legend(fontsize="large")
        plt.xlabel(r"$E_\nu$ [MeV]", fontsize="large")
        plt.ylabel(r"Resolution of $E_\text{rec}$ (RMS/Mean) [\%]", fontsize="large")
        plt.savefig("./plots/" + self.out_filename[:-4] + f"_res.pdf")
        plt.close()

        print("\nFraction of E_dep_tracks in electron+desc:", E_dep_tracks_list_allE[0].sum()/E_dep_tracks_list_allE.sum())
        print("Fraction of E_dep_tracks in protons+desc:", E_dep_tracks_list_allE[1].sum()/E_dep_tracks_list_allE.sum())
        print("Fraction of E_dep_tracks in neutrons+desc:", E_dep_tracks_list_allE[2].sum()/E_dep_tracks_list_allE.sum())
        print("Fraction of E_dep_tracks in pions(charged or neutral)+desc:", E_dep_tracks_list_allE[[3,4]].sum()/E_dep_tracks_list_allE.sum())

        print("\nFraction of Q_dots in electron+desc:", Q_dep_dots_list_allE[0].sum()/Q_dep_dots_list_allE.sum())
        print("Fraction of Q_dots in protons+desc:", Q_dep_dots_list_allE[1].sum()/Q_dep_dots_list_allE.sum())
        print("Fraction of Q_dots in neutrons+desc:", Q_dep_dots_list_allE[2].sum()/Q_dep_dots_list_allE.sum())
        print("Fraction of Q_dots in pions(charged or neutral)+desc:", Q_dep_dots_list_allE[[3,4]].sum()/Q_dep_dots_list_allE.sum())

    # ------ L1 analysis ------ #
    def particle_light_responses(self):
        E_avail_list = np.zeros((4, 10000))
        L_dep_list = np.zeros((4, 10000))

        for i, E_nu in enumerate(self.event_energies.keys()):
            info = self.event_energies[E_nu]
            E_avail_list[:, 1000*i:1000*(i+1)] = info['E_avail_list'][[0,1,2,3]]
            L_dep_list[:, 1000*i:1000*(i+1)] = info['L_dep_list_180PEpMeV'][[0,1,2,3]]

        fig, (ax1, ax2) = plt.subplots(2, figsize=(6.4, 9.6))
        # fig.suptitle(f"nu: {info['nu_pdg'][0]}. {L_dep_list.shape[1]} events.")
        fig.set_layout_engine("tight")

        R_cal_light_list = L_dep_list/E_avail_list
        ax1.plot(R_cal_light_list[0], E_avail_list[0], '.', markersize=1, color='g')
        ax1.plot(R_cal_light_list[1], E_avail_list[1], '.', markersize=2, color='k')
        ax1.plot(R_cal_light_list[2], E_avail_list[2], '.', markersize=2, color='r')
        ax1.plot(R_cal_light_list[3], E_avail_list[3], '.', markersize=2, color='b')

        ax1.plot([], [], '.', markersize=5, color='g', label=r'$e$')
        ax1.plot([], [], '.', markersize=5, color='k', label=r'$p$')
        ax1.plot([], [], '.', markersize=5, color='r', label=r'$n$')
        ax1.plot([], [], '.', markersize=5, color='b', label=r'$\pi^{\pm}$')

        ax1.set_ylabel(r"$E_\text{avail}$ of primary particle", fontsize="large")
        ax1.set_xlabel(r"Light calorimetric response = $L_\text{incl. all desc.} / E_\text{avail}$", fontsize="large")
        ax1.set_xlim(0, 1.2)
        ax1.legend(fontsize="xx-large")

        be = np.linspace(0, 1.2, 121)
        ax2.hist(R_cal_light_list.T, bins=be, histtype='step', density=True, color=['g', 'k', 'r', 'b'], label=['e', 'p', 'n', 'chpi'])
        ax2.set_ylim(0, 7)
        ax2.set_ylabel("Probability Density")
        ax2.set_xlabel("Light calorimetric response = L(+desc.) / E_avail")
        ax2.set_title("Bin width = 0.01", fontsize="x-small")
        ax2.legend()
        
        fig.savefig("./plots/" + self.out_filename)
        plt.close()

    # ------ Q1 analysis ------ #
    def particle_charge_responses(self):
        E_avail_list = np.zeros((4, 10000))
        Q_dep_list = np.zeros((4, 10000))

        for i, E_nu in enumerate(self.event_energies.keys()):
            info = self.event_energies[E_nu]
            E_avail_list[:, 1000*i:1000*(i+1)] = info['E_avail_list'][[0,1,2,3]]
            Q_dep_list[:, 1000*i:1000*(i+1)] = info['Q_dep_list_th75keV'][[0,1,2,3]]

        fig, (ax1, ax2) = plt.subplots(2, figsize=(6.4, 9.6))
        fig.suptitle(f"nu: {info['nu_pdg'][0]}. {Q_dep_list.shape[1]} events.")
        fig.set_layout_engine("tight")

        R_cal_charge_list = Q_dep_list/E_avail_list
        ax1.plot(R_cal_charge_list[0], E_avail_list[0], '.', markersize=1, color='g', alpha=0.3)
        ax1.plot(R_cal_charge_list[1], E_avail_list[1], '.', markersize=2, color='k', alpha=0.3)
        ax1.plot(R_cal_charge_list[2], E_avail_list[2], '.', markersize=2, color='r', alpha=0.3)
        ax1.plot(R_cal_charge_list[3], E_avail_list[3], '.', markersize=2, color='b', alpha=0.3)

        ax1.plot([], [], '.', markersize=5, color='g', label=r'$e$')
        ax1.plot([], [], '.', markersize=5, color='k', label=r'$p$')
        ax1.plot([], [], '.', markersize=5, color='r', label=r'$n$')
        ax1.plot([], [], '.', markersize=5, color='b', label=r'$\pi^{\pm}$')

        ax1.set_ylabel(r"$E_\text{avail}$ of primary particle", fontsize="large")
        ax1.set_xlabel(r"Charge calorimetric response = $Q_\text{incl. all desc.} / E_\text{avail}$", fontsize="large")
        ax1.set_xlim(0, 0.85)
        ax1.legend(fontsize="xx-large")

        be = np.linspace(0, 1.2, 121)
        ax2.hist(R_cal_charge_list.T, bins=be, histtype='step', density=True, color=['g', 'k', 'r', 'b'], label=['e', 'p', 'n', 'chpi'])
        ax2.set_ylim(0, 6)
        ax2.set_ylabel("Probability Density")
        ax2.set_xlabel("Charge calorimetric response = Q(+desc.) / E_avail")
        ax2.set_title("Bin width = 0.01", fontsize="x-small")
        ax2.legend()
        
        fig.savefig("./plots/" + self.out_filename)
        plt.close()
    
    # ------ Q+L analysis ------ #
    def particle_QL_responses(self):
        E_avail_list = np.zeros((4, 10000))
        QL_dep_list = np.zeros((4, 10000))

        for i, E_nu in enumerate(self.event_energies.keys()):
            info = self.event_energies[E_nu]
            E_avail_list[:, 1000*i:1000*(i+1)] = info['E_avail_list'][[0,1,2,3]]
            QL_dep_list[:, 1000*i:1000*(i+1)] = info['Q_dep_list_th75keV'][[0,1,2,3]] + info['L_dep_list_180PEpMeV'][[0,1,2,3]]

        fig, (ax1, ax2) = plt.subplots(2, figsize=(6.4, 9.6))
        fig.suptitle(f"nu: {info['nu_pdg'][0]}. {QL_dep_list.shape[1]} events.")
        fig.set_layout_engine("tight")

        R_cal_QL_list = QL_dep_list/E_avail_list
        ax1.plot(R_cal_QL_list[0], E_avail_list[0], '.', markersize=1, color='g', label='e')
        ax1.plot(R_cal_QL_list[1], E_avail_list[1], '.', markersize=2, color='k', label='p')
        ax1.plot(R_cal_QL_list[2], E_avail_list[2], '.', markersize=2, color='r', label='n')
        ax1.plot(R_cal_QL_list[3], E_avail_list[3], '.', markersize=2, color='b', label='chpi')
        ax1.set_ylabel("E_avail of primary particle")
        ax1.set_xlabel("Charge+Light calorimetric response = (Q+L)(+desc.) / E_avail")
        ax1.set_xlim(0, 1.2)
        ax1.legend()

        be = np.linspace(0, 1.2, 121)
        ax2.hist(R_cal_QL_list.T, bins=be, histtype='step', density=True, color=['g', 'k', 'r', 'b'], label=['e', 'p', 'n', 'chpi'])
        ax2.set_ylim(0, 10)
        ax2.set_ylabel("Probability Density")
        ax2.set_xlabel("Charge+Light calorimetric response = (Q+L)(+desc.) / E_avail")
        ax2.set_title("Bin width = 0.01", fontsize="x-small")
        ax2.legend()
        
        fig.savefig("./plots/" + self.out_filename)
        plt.close()

    # --- E_avail resolution --- #
    def E_avail_res(self):
        E_a_res = []
        E_a_pFS_res = []
        E_nu_list = []
        for i, E_nu in enumerate(self.event_energies.keys()):
            info = self.event_energies[E_nu]
            E_a_res.append(EventsTreeReader.calculateRes(info['E_avail']))
            E_a_pFS_res.append(EventsTreeReader.calculateRes(info['E_avail_pre_FSI_gst']))
            E_nu_list.append(E_nu)

        plt.figure()
        plt.plot(E_nu_list, E_a_res, label=r"Truth $E_\text{avail}$ values", color="C0")
        # fitCoeff     = curve_fit(lambda E, a: a / np.sqrt(E), E_nu_list, E_a_res)[0][0] 
        # plt.plot(E_nu_list, fitCoeff     / np.sqrt(np.array(E_nu_list)), ls="--", color="C1", label=f"Fit curve to $E_\\text{{avail}}$: $\\text{{Res}} = {fitCoeff:.1f}/\\sqrt{{E_\\nu}}$")
        plt.xlabel(r"$E_\nu$ [MeV]", fontsize=14)
        plt.ylabel(r"$E_\text{avail}$ RMS/Mean ('resolution')", fontsize="large")
        # plt.legend()
        plt.savefig(f"plots/{self.out_filename}")
        plt.clf()
        plt.close()

        plt.figure()
        plt.plot(E_nu_list, E_a_pFS_res, label=r"Truth $E_\text{avail, pre-FSI}$ values", color="cyan")
        from scipy.optimize import curve_fit
        fitCoeff_pFS = curve_fit(lambda E, a: a / np.sqrt(E), E_nu_list, E_a_pFS_res)[0][0] 
        plt.plot(E_nu_list, fitCoeff_pFS / np.sqrt(np.array(E_nu_list)), ls="--", color="C2", label=f"Fit curve to $E_\\text{{avail, pre-FSI}}$: $\\text{{Res}} = {fitCoeff_pFS:.1f}/\\sqrt{{E_\\nu}}$")
        plt.xlabel(r"$E_\nu$ [MeV]", fontsize="large")
        plt.ylabel(r"$E_\text{avail, pre-FSI}$ RMS/Mean ('resolution')", fontsize="large")
        plt.legend()
        plt.savefig(f"plots/{self.out_filename[:-4]}_pre_FSI.pdf")
        plt.clf()
        plt.close()

    # --- E_avail breakdown --- #
    def E_avail_breakdown(self):
        no_of_samples = len(self.event_energies.keys())
        all_E_al = np.zeros((8, 1000*no_of_samples))
        all_E_dl = np.zeros((8, 1000*no_of_samples))
        all_N_pl = np.zeros((8, 1000*no_of_samples))

        E_nu_list = []

        fracs = np.zeros((6,no_of_samples))
        # fracs = np.zeros((4,no_of_samples))
        cur_levels = np.zeros(no_of_samples)
        comps = [r"EM $E_\text{dep}$", r"Proton $E_\text{dep}$", r"Neutron $E_\text{dep}$", r"$\pi^{\pm}$ $E_\text{dep}$", r"Other $E_\text{dep}$", r"Missing Energy"]
        # comps = [r"Lepton $E_\text{dep}$", r"Proton $E_\text{dep}$", r"Neutron $E_\text{dep}$", r"$\pi^{\pm,0}$,Other $E_\text{dep}$"]
        # colors = ["C2", "C0", "C1", "C3"]
        for i, E_nu in enumerate(self.event_energies.keys()):
            info = self.event_energies[E_nu]
            all_E_al[:, 1000*i:1000*(i+1)] = info['E_avail_list']
            all_E_dl[:, 1000*i:1000*(i+1)] = info['E_dep_list']
            all_N_pl[:, 1000*i:1000*(i+1)] = info['N_par_list']
            fracs[0, i] = info['E_dep_list'][[0, 4, 5]].sum() / info['E_avail'].sum() 
            fracs[1, i] = info['E_dep_list'][1].sum()         / info['E_avail'].sum() 
            fracs[2, i] = info['E_dep_list'][2].sum()         / info['E_avail'].sum() 
            fracs[3, i] = info['E_dep_list'][3].sum()         / info['E_avail'].sum() 
            fracs[4, i] = info['E_dep_list'][[6,7]].sum()     / info['E_avail'].sum() 
            fracs[5, i] = 1 - (info['E_dep_list'].sum()       / info['E_avail'].sum())

            # fracs[0, i] = (info['E_avail_list'][0] / info['E_avail']).mean() 
            # fracs[1, i] = (info['E_avail_list'][1] / info['E_avail']).mean() 
            # fracs[2, i] = (info['E_avail_list'][2] / info['E_avail']).mean() 
            # # fracs[3, i] = (info['E_avail_list'][3]         / info['E_avail']).mean() 
            # fracs[4, i] = (info['E_avail_list'][3:] / info['E_avail']).mean() 
            # # fracs[5, i] = 1 - (info['E_avail_list'].sum()       / info['E_avail'].sum())
            
            # fracs[0, i] = info['E_avail_list'][0].sum()          
            # fracs[1, i] = info['E_avail_list'][1].sum()          
            # fracs[2, i] = info['E_avail_list'][2].sum()          
            # fracs[3, i] = (E_nu*1000) - info['E_avail_list'][[0,1,2,7]].sum()   
            # fracs[:, i] /= fracs[:, i].sum()

            E_nu_list.append(E_nu)

        fig, axs = plt.subplot_mosaic([['.', 'fig', '.']], layout="constrained", width_ratios=(0.05, 1,0.05))
        # plt.title(f"nu: {info['nu_pdg'][0]}. {all_E_al.shape[1]} events.")
        for i, label in enumerate(comps): # Here fracs[i] is already a length-10 array of ratios, one for each energy
            bars = axs['fig'].bar(E_nu_list, fracs[i], width=100, bottom=cur_levels, label=label)#, color=colors[i])
            axs['fig'].bar_label(bars, labels=[f"{ratio*100:.0f}$\\%$" if ratio >= 0.05 else "" for ratio in fracs[i]], label_type='center', fontsize='medium')
            cur_levels += fracs[i]
        axs['fig'].set_xlabel(r"$E_\nu$ [MeV]", fontsize="large")
        axs['fig'].set_ylabel(r"Fraction of $E_\text{avail}$", fontsize="large")
        # axs['fig'].set_ylabel(r"Fraction")
        axs['fig'].legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize="large")
        fig.savefig(f"plots/{self.out_filename}")
        plt.clf()
        plt.close()

        print(f"Percentage of available energy deposited by EM component over all events: {(all_E_dl[[0, 4, 5]].sum() / all_E_al.sum()) * 100:.2f}%")
        print(f"Percentage of available energy deposited by protons      over all events: {(all_E_dl[1].sum()         / all_E_al.sum()) * 100:.2f}%")
        print(f"Percentage of available energy deposited by neutrons     over all events: {(all_E_dl[2].sum()         / all_E_al.sum()) * 100:.2f}%")
        print(f"Percentage of available energy deposited by pi+-s        over all events: {(all_E_dl[3].sum()         / all_E_al.sum()) * 100:.2f}%")
        print(f"Percentage of available energy never deposited           over all events: {(1-(all_E_dl.sum() / all_E_al.sum())) * 100:.2f}%")
        #print(f"Percentage of available energy deposited by others       over all {E_nu} MeV events: {(E_dl[[6, 7]].sum()    / E_al.sum()) * 100:.2f}%")

    # --- missing energy by particle --- #
    def missing_energy_by_particle(self):
        all_E_al = np.zeros((8, 10000))
        all_E_dl = np.zeros((8, 10000))

        fracs = np.zeros((6,10))
        cur_levels = np.zeros(10)
        for i, E_nu in enumerate(self.event_energies.keys()):
            info = self.event_energies[E_nu]
            all_E_al[:, 1000*i:1000*(i+1)] = info['E_avail_list']
            all_E_dl[:, 1000*i:1000*(i+1)] = info['E_dep_list']

        em_ratio = (all_E_dl[[0, 4, 5]].sum() / all_E_al[[0, 4, 5]].sum())
        p_ratio  = (all_E_dl[[1]].sum()             / all_E_al[[1]].sum())
        n_ratio  = (all_E_dl[[2]].sum()             / all_E_al[[2]].sum())
        pi_ratio = (all_E_dl[[3]].sum()             / all_E_al[[3]].sum())

        fig, ax = plt.subplots()
        ax.bar(["$p$", "$n$", r"$\pi^{\pm}$", "EM"], [p_ratio, n_ratio, pi_ratio, em_ratio], facecolor='0.3', edgecolor="black", hatch=r"\\", width=0.9)
        ax.set_xlabel("Primary Particle")
        ax.set_ylabel(r"Ratio of $E_\text{avail}$ deposited")
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        fig.savefig(f"plots/{self.out_filename}")
        plt.clf()
        plt.close()

        print(f"Percentage of EM component available energy deposited: {(all_E_dl[[0, 4, 5]].sum() / all_E_al[[0, 4, 5]].sum()) * 100:.2f}%")
        print(f"Percentage of protons' available energy deposited:     {(all_E_dl[[1]].sum()             / all_E_al[[1]].sum()) * 100:.2f}%")
        print(f"Percentage of neutrons' available energy deposited:    {(all_E_dl[[2]].sum()             / all_E_al[[2]].sum()) * 100:.2f}%")
        print(f"Percentage of pi+-s' available energy deposited:       {(all_E_dl[[3]].sum()             / all_E_al[[3]].sum()) * 100:.2f}%")

    #---------------------------------------------------
    def plotE_recHistogram(self, E_nu, ep=False, eaview=False):
        # Genie_v3_antinue_100MeV_to_1GeV_mean_scaling_factors = {"L1":        0.437,
        #                                                         "Q1":        0.431,
        #                                                         "Q2_e":      0.550,
        #                                                         "Q2_h":      0.293,
        #                                                         "Q3_e_dots": 0.468,
        #                                                         "Q3_h_dots": 0.148}

        info = self.event_energies[E_nu]
        mask_1p = np.array( [ (col ==  [1, 1, 0, 0, 0, 0, 0, 0]).all() for col in info['N_par_list'].T ] )    # Not separating pi+- for now

        # bin_width = E_nu / 100
        bin_width = E_nu / 50

        # What you are doing is trying to reconstruct the available energy distribution from the Q and L distributions.
        # Trying to correct from the available energy to the actual neutrino energy will be a different matter. Since we know the available energy distribution
        # for the same E_nu does have a decent amount of event-by-event fluctuation as it depends on the FSI effects on top of the interaction type.
        if not ep:
            L1, Q1, Q2, Q3 = info['E_rec_L1'], info['E_rec_Q1'], info['E_rec_Q2'], info['E_rec_Q3']                            # Has ~30 MeV pre-added
            # QL1, QL2 = info['E_rec_(Q+L)1'], info['E_rec_(Q+L)2']
            QL1 = info['E_rec_(Q+L)1']
            E_a = info['E_avail']
            min_x, max_x = np.concatenate((L1, Q1, Q2, Q3, QL1, E_a)).min(), np.concatenate((L1, Q1, Q2, Q3, QL1, E_a)).max()
        else:
            L1_1p, Q1_1p, Q2_1p, Q3_1p = info['E_rec_L1_1p'], info['E_rec_Q1_1p'], info['E_rec_Q2_1p'], info['E_rec_Q3_1p']    # Has ~30 MeV pre-added
            QL1_1p, QL2_1p = info['E_rec_(Q+L)1_1p'], info['E_rec_(Q+L)2_1p']
            E_a_1p = info['E_avail'][mask_1p]
            min_x, max_x = np.concatenate((L1_1p, Q1_1p, Q2_1p, Q3_1p, QL1_1p, QL2_1p, E_a_1p)).min(), np.concatenate((L1_1p, Q1_1p, Q2_1p, Q3_1p, QL1_1p, QL2_1p, E_a_1p)).max()

        # L1_1p, Q1_1p, Q2_1p, Q3_1p = [E_rec + E_a_to_E_nu_shift_1p for E_rec in [L1_1p, Q1_1p, Q2_1p, Q3_1p]]    # Adding back the ~30 MeV.
        # L1, Q1, Q2, Q3 = [E_rec + E_a_to_E_nu_shift for E_rec in [L1, Q1, Q2, Q3]]    # Adding back the ~30 MeV so that the peak (which is mostly 1p events) aligns.

        midrange_x, halfrange_x = (min_x + max_x) / 2, (max_x - min_x) / 2
        xlim_min, xlim_max = midrange_x - (1.1 * halfrange_x), midrange_x + (1.1 * halfrange_x)
        nearest_greater_multiple = xlim_max + bin_width - (xlim_max % bin_width)
        bin_edges = np.linspace(0, nearest_greater_multiple, int(nearest_greater_multiple / bin_width) + 1)

        if eaview:
            fig, (ax1, ax2) = plt.subplots(2, figsize=(6.4, 4.8*2))
        else:
            fig, ax1 = plt.subplots()
        # ax1.hist(L1_1p, bins=bin_edges, histtype="step", color="C3",    label=f"L1. Resolution:{(EventsTreeReader.calculateRes(L1_1p)*100):.1f}%")
        fig.set_layout_engine('tight')

        label_l = ["L1", "Q1",    "Q2", "Q3",   "Q+L",  "(Q+L)2"]
        color_l = ["C3", "black", "C2", "blue", "magenta", "C8"]
        if ep:
        #    rec_l = [L1_1p, Q1_1p, Q2_1p, Q3_1p, QL1_1p, QL2_1p]
            rec_l = [L1_1p, Q1_1p, Q2_1p, Q3_1p]
            fig.suptitle(f"nu:{info['nu_pdg'][0]}, E_nu = {E_nu} MeV, bin width = {bin_width} MeV, 1p0n0pi evts only ({len(E_a_1p)} / {info['nEvents']})", fontsize="medium")
            ax1.set_title(f"Energies shifted by +{self.E_a_to_E_nu_shift_1p} MeV", fontsize="medium")
        else:
            # rec_l = [L1, Q1, Q2, Q3, QL1, QL2]
            rec_l = [L1, Q1, Q2, Q3, QL1]
            # rec_l = [L1, Q1, Q2, Q3]
            # fig.suptitle(f"nu:{info['nu_pdg'][0]}, E_nu = {E_nu} MeV, bin width = {bin_width} MeV, {info['nEvents']} events")
            # ax1.set_title(f"Energies shifted by +{self.E_a_to_E_nu_shift} MeV", fontsize="medium")

        for i, E_rec in enumerate(rec_l):
            if not ep:
                # ax1.hist(E_rec, bins=bin_edges, histtype="step", color=color_l[i], label=f"{label_l[i]:<6s}. Resolution:{(self.resolutions['E_rec_' + label_l[i] + '_res'][E_nu]*100):.1f}%. Rel.Bias = {(self.biases['E_rec_' + label_l[i] + '_b'][E_nu]*100):.1f}%")
                ax1.hist(E_rec, bins=bin_edges, histtype="step", color=color_l[i], lw=2, label=f"{label_l[i]:<6s}")
            else:
                ax1.hist(E_rec, bins=bin_edges, histtype="step", color=color_l[i], label=f"{label_l[i]:<6s}. Resolution:{(self.resolutions[label_l[i] + '_res_1p'][E_nu]*100):.1f}%. Rel.Bias = {(self.biases[label_l[i] + '_b_1p'][E_nu]*100):.1f}%")
            if eaview:
                ax1.axvline(np.mean(E_rec), color=color_l[i], ls=":", alpha=0.4, label=f"{label_l[i]} mean")

        ax1.axvline(E_nu, color="indigo", ls="--", alpha=0.4, label=r"Actual $E_\nu$")

        if eaview:
            ax2.hist(E_a if not ep else E_a_1p, bins=bin_edges, histtype="step", label="Actual E_avail")
            ax2.axvline(E_nu,                               color="indigo", ls="--", alpha=0.4, label="Actual E_nu")
            ax2.axvline(np.mean(E_a if not ep else E_a_1p), color="blue",   ls=":",  alpha=0.4, label="Mean E_a")
            ax2.set_xlabel("E_avail post-FSI [MeV]")

            for ax in (ax1, ax2):
                ax.set_xlim(xlim_min, xlim_max)
                ax.set_ylabel("Number of events")
                ax.legend(fontsize="x-small")   
        else:
            ax1.set_xlim(xlim_min, xlim_max)
            ax1.set_xlabel("Reconstructed Energy [MeV]", fontsize="large")
            ax1.set_ylabel("Number of events", fontsize="large")
            ax1.legend(fontsize="large")

        self.SaveAndClose(E_nu)

    #---------------------------------------------------
    def calculateMeanValInPeak(data=np.array([]), bin_heights=np.array([]), bin_width=0, bin_left_limit=0):
        peak_left_edge = bin_left_limit + (np.argmax(bin_heights) * bin_width)
        peak_mean = np.mean( data[ (data >= peak_left_edge) & (data < peak_left_edge + bin_width) ] )
        return peak_mean

    #---------------------------------------------------
    #def plotR_calHistogram(self, E_nu=-1):
    def plotR_calHistogram(self, tplgy=None, peaks=True):
        R_cal_light_allE = np.array([])
        R_cal_light_e_allE = np.array([])
        R_cal_light_h_allE = np.array([])
        R_cal_charge_allE = np.array([])
        R_cal_charge_e_allE = np.array([])
        R_cal_charge_h_allE = np.array([])
        R_cal_charge_e_dots_allE = np.array([])
        R_cal_charge_h_dots_allE = np.array([])
        Q_e_dots_is_0_mask_allE = np.array([], dtype=np.bool_)
        Q_h_dots_is_0_mask_allE = np.array([], dtype=np.bool_)
        tplgy_mask_allE = np.array([], dtype=np.bool_)
        R_cal_QL_allE = np.array([])
        R_cal_QL_e_allE = np.array([])
        R_cal_QL_h_allE = np.array([])

        E_nu_minus_E_a_allE = np.array([])             # Will have as many elements as the total number of entries in the tree
        bws_str = ['E_nu / 10', 'E_nu / 20', 'E_nu / 25', 'E_nu / 50', 'E_nu / 100', 'E_nu / 200']
        E_nu_minus_E_a_peak_allE = np.empty((len(bws_str), 0))    # Will have as many columns as the total number of E_nu values in the tree.
        E_nu_minus_E_a_mean_allE = np.array([])                   # Will have as many elements as the total number of E_nu values in the tree.
        E_a_over_E_nu_mean_allE = np.array([])                    # Will have as many elements as the total number of E_nu values in the tree.
        E_a_allE = np.array([])
        E_d_allE = np.array([])

        # For proton / neutron individual graphs
        R_cal_charge_p_allE = np.array([])
        R_cal_charge_p_dots_allE = np.array([])
        R_cal_charge_n_allE = np.array([])
        R_cal_charge_n_dots_allE = np.array([])
        Q_p_dots_is_0_mask_allE = np.array([], dtype=np.bool_)
        Q_n_dots_is_0_mask_allE = np.array([], dtype=np.bool_)
        R_cal_light_p_allE = np.array([])
        R_cal_light_n_allE = np.array([])
        R_cal_QL_p_allE = np.array([])
        R_cal_QL_n_allE = np.array([])
        #---------------------------------------
        bin_width = 0.01
        bin_edges = np.linspace(0, 1.1, 111)                             # (1.1/0.01) + 1

        for E_nu in self.event_energies.keys():
            info = self.event_energies[E_nu]
            nu_pdg = info['nu_pdg'][0]
            E_a, E_al, E_al_dots = info['E_avail'], info['E_avail_list'], info['E_avail_list_dots']
#            E_a, E_al, E_al_dots = info['E_avail'], info['E_avail_list'], info['E_avail_list'] - info['E_dep_tracks_list']
            E_d = info['E_dep']

            # Calorimeter response = Energy visible to calorimeter from event / Total available energy in event
            R_cal_light         = info['L']        / E_a                                    # Assuming an avg light yield of 180 PE/MeV deposited by a MIP
            R_cal_light_e       = info['L_e']      / E_al[[0, 4, 5]].sum(axis=0)            # EM component is rows 0 (e), 4 (pi0) and 5 (gamma)
            R_cal_light_h       = info['L_h']      / E_al[[1, 2, 3, 6, 7]].sum(axis=0)      # Hadronic component is rows 1 (p), 2 (n), 3 (pi+-), 6 (alpha), 7 (other)
            R_cal_charge        = info['Q_thre']   / E_a                                    # Assuming 75 keV charge detection threshold
#            R_cal_charge        = info['Q']   / E_a
            R_cal_charge_e      = info['Q_e']      / E_al[[0, 4, 5]].sum(axis=0)            # Using 75 keV threshold. EM component is rows 0 (e), 4 (pi0) and 5 (gamma)
            R_cal_charge_h      = info['Q_h']      / E_al[[1, 2, 3, 6, 7]].sum(axis=0)      # Using 75 keV threshold. Hadronic component is rows 1 (p), 2 (n), 3 (pi+-), 6 (alpha), 7 (other)
            R_cal_charge_e_dots = info['Q_e_dots'] / E_al_dots[[0, 4, 5]].sum(axis=0)       # 'Dots' refer to tracks less than 2 cm in length, only show up as blips in detector
            R_cal_charge_h_dots = info['Q_h_dots'] / E_al_dots[[1, 2, 3, 6, 7]].sum(axis=0) # <^Using 75 keV threshold.
            Q_e_dots_is_0_mask = info['Q_e_dots'] == 0
            Q_h_dots_is_0_mask = info['Q_h_dots'] == 0
            R_cal_QL   = (info['L'] + info['Q_thre']) / E_a
            R_cal_QL_e = (info['L_e'] + info['Q_e'])  / E_al[[0, 4, 5]].sum(axis=0)
            R_cal_QL_h = (info['L_h'] + info['Q_h'])  / E_al[[1, 2, 3, 6, 7]].sum(axis=0)

            # For proton / neutron individual graphs
            R_cal_light_p            = info['L_dep_list_180PEpMeV'][1]    / E_al[1]
            R_cal_charge_p           = info['Q_dep_list_th75keV'][1]      / E_al[1]
            R_cal_charge_p_dots      = info['Q_dep_list_dots_th75keV'][1] / E_al_dots[1]
            R_cal_light_n            = info['L_dep_list_180PEpMeV'][2]    / E_al[2]
            R_cal_charge_n           = info['Q_dep_list_th75keV'][2]      / E_al[2]
            R_cal_charge_n_dots      = info['Q_dep_list_dots_th75keV'][2] / E_al_dots[2]
            Q_p_dots_is_0_mask = info['Q_dep_list_dots_th75keV'][1] == 0
            Q_n_dots_is_0_mask = info['Q_dep_list_dots_th75keV'][2] == 0
            R_cal_QL_p = (info['L_dep_list_180PEpMeV'][1] + info['Q_dep_list_th75keV'][1]) / E_al[1]
            R_cal_QL_n = (info['L_dep_list_180PEpMeV'][2] + info['Q_dep_list_th75keV'][2]) / E_al[2]
            #---------------------------------------
            R_cal_light_allE         = np.append(R_cal_light_allE,         R_cal_light)
            R_cal_light_e_allE       = np.append(R_cal_light_e_allE,       R_cal_light_e)
            R_cal_light_h_allE       = np.append(R_cal_light_h_allE,       R_cal_light_h)
            R_cal_charge_allE        = np.append(R_cal_charge_allE,        R_cal_charge)
            R_cal_charge_e_allE      = np.append(R_cal_charge_e_allE,      R_cal_charge_e)
            R_cal_charge_h_allE      = np.append(R_cal_charge_h_allE,      R_cal_charge_h)
            R_cal_charge_e_dots_allE = np.append(R_cal_charge_e_dots_allE, R_cal_charge_e_dots)
            R_cal_charge_h_dots_allE = np.append(R_cal_charge_h_dots_allE, R_cal_charge_h_dots)
            Q_e_dots_is_0_mask_allE = np.append(Q_e_dots_is_0_mask_allE, Q_e_dots_is_0_mask)
            Q_h_dots_is_0_mask_allE = np.append(Q_h_dots_is_0_mask_allE, Q_h_dots_is_0_mask)
            R_cal_QL_allE            = np.append(R_cal_QL_allE,            R_cal_QL)
            R_cal_QL_e_allE          = np.append(R_cal_QL_e_allE,          R_cal_QL_e)
            R_cal_QL_h_allE          = np.append(R_cal_QL_h_allE,          R_cal_QL_h)

            # For proton / neutron individual graphs
            R_cal_light_p_allE       = np.append(R_cal_light_p_allE,       R_cal_light_p)
            R_cal_charge_p_allE      = np.append(R_cal_charge_p_allE,      R_cal_charge_p)
            R_cal_charge_p_dots_allE = np.append(R_cal_charge_p_dots_allE, R_cal_charge_p_dots)
            R_cal_light_n_allE       = np.append(R_cal_light_n_allE,       R_cal_light_n)
            R_cal_charge_n_allE      = np.append(R_cal_charge_n_allE,      R_cal_charge_n)
            R_cal_charge_n_dots_allE = np.append(R_cal_charge_n_dots_allE, R_cal_charge_n_dots)
            Q_p_dots_is_0_mask_allE = np.append(Q_p_dots_is_0_mask_allE, Q_p_dots_is_0_mask)
            Q_n_dots_is_0_mask_allE = np.append(Q_n_dots_is_0_mask_allE, Q_n_dots_is_0_mask)
            R_cal_QL_p_allE          = np.append(R_cal_QL_p_allE,          R_cal_QL_p)
            R_cal_QL_n_allE          = np.append(R_cal_QL_n_allE,          R_cal_QL_n)
            #---------------------------------------

            E_nu_minus_E_a_allE = np.append(E_nu_minus_E_a_allE, E_nu - E_a)
            E_a_allE = np.append(E_a_allE, E_a)
            E_d_allE = np.append(E_d_allE, E_d)

            mask_tplgy = np.ones((info['nEvents'],), dtype=np.bool_)
            if tplgy:
                N_pl = info["N_par_list"]
                N_pil = info["N_pi+-_list_gst"]
                N_pl = np.concatenate( (N_pl[0:3], N_pil, N_pl[4:]), axis=0)

                # Right now this is 'nue-centered'
                tplgies = {                           # e|μ, p,                n,            π+,  π-,  π0,  γ,   α,   etc
                    "1p0n0π":  np.array( [ (col ==    [1,   1,                0,            0,   0,   0,   0,   0,   0  ]).all() for col in N_pl.T ] ),
                    "1p1n0π":  np.array( [ (col ==    [1,   1,                1,            0,   0,   0,   0,   0,   0  ]).all() for col in N_pl.T ] ),
                    "1pXn0π":  np.array( [ (col[0:2]==[1,   1]).all()&(col[2]>0)&(col[3:]==[0,   0,   0,   0,   0,   0  ]).all() for col in N_pl.T ] ),
                    "2p0n0π":  np.array( [ (col ==    [1,   2,                0,            0,   0,   0,   0,   0,   0  ]).all() for col in N_pl.T ] ),
                    "1p0n1π+": np.array( [ (col ==    [1,   1,                0,            1,   0,   0,   0,   0,   0  ]).all() for col in N_pl.T ] )
                }
                mask_tplgy = tplgies[tplgy]
            #    tplgy_mask_allE = np.append(tplgy_mask_allE, tplgies[tplgy])
            #else:
            #    tplgy_mask_allE = np.append(tplgy_mask_allE, np.ones((info['nEvents'],), dtype=np.bool_))
            tplgy_mask_allE = np.append(tplgy_mask_allE, mask_tplgy)

            bws = [eval(bw_s, {'E_nu':E_nu}) for bw_s in bws_str]
            E_nu_minus_E_a_peak = []
            for bw in bws:
                be = np.linspace(0, 2*E_nu, int(2*E_nu / bw) + 1)
                E_a_bh, _ = np.histogram(E_a[mask_tplgy], bins=be)
                E_a_peak_mean = EventsTreeReader.calculateMeanValInPeak(data=E_a[mask_tplgy], bin_heights=E_a_bh, bin_width=bw, bin_left_limit=0)
                E_nu_minus_E_a_peak.append(E_nu - E_a_peak_mean)
            print(E_nu, "E_nu - peak E_a (all evts, usual binning) =", E_nu_minus_E_a_peak[4])
            E_nu_minus_E_a_peak_allE = np.append(E_nu_minus_E_a_peak_allE, np.array([E_nu_minus_E_a_peak]).T, axis=1)

            E_nu_minus_E_a_mean_allE = np.append(E_nu_minus_E_a_mean_allE, np.mean(E_nu - E_a))
            E_a_over_E_nu_mean_allE = np.append(E_a_over_E_nu_mean_allE, np.mean(E_a / E_nu))

        if tplgy == "1p0n0π":
            print("Average difference between E_nu and E_avail for 1p0n0pi evts:", np.mean(E_nu_minus_E_a_allE[tplgy_mask_allE]))
            print("No. of 1p evts where E_dep is more than 20 MeV less than E_avail:", np.sum(tplgy_mask_allE & (E_a_allE - E_d_allE > 20)))
            print("No. of 1p evts where E_dep is greater than E_avail:", np.sum(tplgy_mask_allE & (E_a_allE < E_d_allE)))
        #print("No. of events where Q_e_dots is 0:", Q_e_dots_is_0_mask_allE.sum())
        #print("No. of events where Q_h_dots is 0:", Q_h_dots_is_0_mask_allE.sum())
        for r, bw_s in enumerate(bws_str):
            print(f"Average difference between E_nu and the peak of E_avail over all E_nu when bin width is {bw_s}:", np.mean(E_nu_minus_E_a_peak_allE[r]))
        print("Average difference between E_nu and the mean of E_avail over all E_nu:", np.mean(E_nu_minus_E_a_mean_allE), np.mean(E_nu_minus_E_a_allE))
        print("Average (mean of E_avail / E_nu) over all E_nu:", np.mean(E_a_over_E_nu_mean_allE))

        #fig = plt.figure(layout='tight', figsize=(8, 8))
        fig = plt.figure(layout='tight', figsize=(8, 12))
        #gs = fig.add_gridspec(2, hspace=0)
        gs = fig.add_gridspec(3, hspace=0)
        #Qax, Lax = gs.subplots(sharex=True)
        Qax, Lax, QLax = gs.subplots(sharex=True)
        # if not tplgy:
        #     fig.suptitle(f"nu: {nu_pdg}, E_nu from {list(self.event_energies.keys())[0]} to {list(self.event_energies.keys())[-1]} MeV, bin width = {bin_width}, {len(R_cal_light_allE)} events total")
        # else:
        #     fig.suptitle(f"nu: {nu_pdg}, E_nu from {list(self.event_energies.keys())[0]} to {list(self.event_energies.keys())[-1]} MeV, bin width = {bin_width}\n{tplgy} evts, {len(R_cal_light_allE[tplgy_mask_allE])} / {len(R_cal_light_allE)} evts total", fontsize="small")

        # I am using the peak value of the R_cal distribution rather than the mean as the scaling factor, so that the peak of the scaled up L or Q distribution
        # matches the peak in E_avail.

        # Lax.hist(R_cal_light_allE,   bins=bin_edges, histtype="step", label=f"Overall R_cal_light. Mean={R_cal_light_allE.mean():.3f}")
        R_cal_L_bh, _, R_cal_L_artist     = Lax.hist(R_cal_light_allE[tplgy_mask_allE],   bins=bin_edges, histtype="step", lw=1.3, label=f"Overall")
        R_cal_L_e_bh, _, R_cal_L_e_artist = Lax.hist(R_cal_light_e_allE[tplgy_mask_allE], bins=bin_edges, histtype="step", lw=1.3, label=f"EM")
        R_cal_L_h_bh, _, R_cal_L_h_artist = Lax.hist(R_cal_light_h_allE[tplgy_mask_allE], bins=bin_edges, histtype="step", lw=1.3, label=f"Hadronic")

        # For proton / neutron individual graphs
#        R_cal_L_p_bh, _, R_cal_L_p_artist = Lax.hist(R_cal_light_p_allE[tplgy_mask_allE], bins=bin_edges, histtype="step", label=f"Proton(+desc.) R_cal_light.")
#        R_cal_L_n_bh, _, R_cal_L_n_artist = Lax.hist(R_cal_light_n_allE[tplgy_mask_allE], bins=bin_edges, histtype="step", label=f"Neutron(+desc.) R_cal_light.")
        #---------------------------------------
        if peaks:
            R_cal_light_peak_mean = EventsTreeReader.calculateMeanValInPeak(  data=R_cal_light_allE[tplgy_mask_allE],   bin_heights=R_cal_L_bh,   bin_width=bin_width, bin_left_limit=0)
            # R_cal_L_artist[0].set_label(  R_cal_L_artist[0].get_label()   + f" Peak value = {R_cal_light_peak_mean:.3f}")
            R_cal_light_e_peak_mean = EventsTreeReader.calculateMeanValInPeak(data=R_cal_light_e_allE[tplgy_mask_allE], bin_heights=R_cal_L_e_bh, bin_width=bin_width, bin_left_limit=0)
            # R_cal_L_e_artist[0].set_label(R_cal_L_e_artist[0].get_label() + f" Peak value = {R_cal_light_e_peak_mean:.3f}")
            R_cal_light_h_peak_mean = EventsTreeReader.calculateMeanValInPeak(data=R_cal_light_h_allE[tplgy_mask_allE], bin_heights=R_cal_L_h_bh, bin_width=bin_width, bin_left_limit=0)
            # R_cal_L_h_artist[0].set_label(R_cal_L_h_artist[0].get_label() + f" Peak value = {R_cal_light_h_peak_mean:.3f}")

            # For proton / neutron individual graphs
#            R_cal_light_p_peak_mean = EventsTreeReader.calculateMeanValInPeak(data=R_cal_light_p_allE[tplgy_mask_allE], bin_heights=R_cal_L_p_bh, bin_width=bin_width, bin_left_limit=0)
#            R_cal_L_p_artist[0].set_label(R_cal_L_p_artist[0].get_label() + f" Peak value = {R_cal_light_p_peak_mean:.3f}")
#            R_cal_light_n_peak_mean = EventsTreeReader.calculateMeanValInPeak(data=R_cal_light_n_allE[tplgy_mask_allE], bin_heights=R_cal_L_n_bh, bin_width=bin_width, bin_left_limit=0)
#            R_cal_L_n_artist[0].set_label(R_cal_L_n_artist[0].get_label() + f" Peak value = {R_cal_light_n_peak_mean:.3f}")
            #---------------------------------------
        else:
            R_cal_L_artist[0].set_label(  R_cal_L_artist[0].get_label()   + f" Mean value = {np.mean(R_cal_light_allE[tplgy_mask_allE]):.3f}")
            R_cal_L_e_artist[0].set_label(R_cal_L_e_artist[0].get_label() + f" Mean value = {np.mean(R_cal_light_e_allE[tplgy_mask_allE]):.3f}")
            R_cal_L_h_artist[0].set_label(R_cal_L_h_artist[0].get_label() + f" Mean value = {np.mean(R_cal_light_h_allE[tplgy_mask_allE]):.3f}")
            # For proton / neutron individual graphs
            #R_cal_L_p_artist[0].set_label(  R_cal_L_p_artist[0].get_label() + f" Mean value = {np.mean(R_cal_light_p_allE[tplgy_mask_allE]):.3f}")
            #R_cal_L_n_artist[0].set_label(  R_cal_L_n_artist[0].get_label() + f" Mean value = {np.mean(R_cal_light_n_allE[tplgy_mask_allE]):.3f}")
            #---------------------------------------
        #Lax.set_xlabel("Calorimeter response ratio")
        Lax.plot([],[], lw=0, label="(180PE/MeV avg LY)")
        Lax.tick_params('x', direction='in')
        Lax.set_ylabel("Number of events", fontsize="x-large")
        Lax.set_title(r"$\qquad R_{\text{cal, light}}$", loc="left", y=0.8, fontsize=25)
        Lax.legend(fontsize="large")

        R_cal_charge_e_dots_allE_nonzero = R_cal_charge_e_dots_allE[(~Q_e_dots_is_0_mask_allE) & tplgy_mask_allE]
        #R_cal_charge_h_dots_allE_nonzero = R_cal_charge_h_dots_allE[(~Q_h_dots_is_0_mask_allE) & tplgy_mask_allE & np.isfinite(R_cal_charge_h_dots_allE) & (R_cal_charge_h_dots_allE>0)]
        R_cal_charge_h_dots_allE_nonzero = R_cal_charge_h_dots_allE[(~Q_h_dots_is_0_mask_allE) & tplgy_mask_allE]
        R_cal_Q_bh, _, R_cal_Q_artist       = Qax.hist(R_cal_charge_allE[tplgy_mask_allE],      bins=bin_edges, histtype="step", lw=1.3, label=f"Overall")
        R_cal_Q_e_bh, _, R_cal_Q_e_artist   = Qax.hist(R_cal_charge_e_allE[tplgy_mask_allE],    bins=bin_edges, histtype="step", lw=1.3, label=f"EM")
        R_cal_Q_h_bh, _, R_cal_Q_h_artist   = Qax.hist(R_cal_charge_h_allE[tplgy_mask_allE],    bins=bin_edges, histtype="step", lw=1.3, label=f"Hadronic")
        # R_cal_Q_ed_bh, _, R_cal_Q_ed_artist = Qax.hist(R_cal_charge_e_dots_allE_nonzero, bins=bin_edges, histtype="step",
        #          label=f"EM R_cal_charge_dots.\nConsidering {len(R_cal_charge_e_dots_allE_nonzero)} evts where Q_e_dots != 0.\n")
        # R_cal_Q_hd_bh, _, R_cal_Q_hd_artist = Qax.hist(R_cal_charge_h_dots_allE_nonzero, bins=bin_edges, histtype="step",
        #          label=f"Hadronic R_cal_charge_dots.\nConsidering {len(R_cal_charge_h_dots_allE_nonzero)} evts where Q_h_dots != 0.\n")
        R_cal_Q_ed_bh, _, R_cal_Q_ed_artist = Qax.hist(R_cal_charge_e_dots_allE_nonzero, bins=bin_edges, histtype="step",
                 lw=1.3, label=f"EM dots")
        R_cal_Q_hd_bh, _, R_cal_Q_hd_artist = Qax.hist(R_cal_charge_h_dots_allE_nonzero, bins=bin_edges, histtype="step",
                 lw=1.3, label=f"Hadronic dots")

        # For proton / neutron individual graphs
#        R_cal_charge_p_dots_allE_nonzero = R_cal_charge_p_dots_allE[(~Q_p_dots_is_0_mask_allE) & tplgy_mask_allE]
#        R_cal_charge_n_dots_allE_nonzero = R_cal_charge_n_dots_allE[(~Q_n_dots_is_0_mask_allE) & tplgy_mask_allE]
#        R_cal_Q_p_bh, _, R_cal_Q_p_artist   = Qax.hist(R_cal_charge_p_allE[tplgy_mask_allE],    bins=bin_edges, histtype="step", label=f"Proton(+desc.) R_cal_charge.")
#        R_cal_Q_n_bh, _, R_cal_Q_n_artist   = Qax.hist(R_cal_charge_n_allE[tplgy_mask_allE],    bins=bin_edges, histtype="step", label=f"Neutron(+desc.) R_cal_charge.")
#        R_cal_Q_pd_bh, _, R_cal_Q_pd_artist = Qax.hist(R_cal_charge_p_dots_allE_nonzero, bins=bin_edges, histtype="step",
#                 label=f"Proton(+desc.) R_cal_charge_dots.\nConsidering {len(R_cal_charge_p_dots_allE_nonzero)} evts where Q_p_dots != 0.\n")
#        R_cal_Q_nd_bh, _, R_cal_Q_nd_artist = Qax.hist(R_cal_charge_n_dots_allE_nonzero, bins=bin_edges, histtype="step",
#                 label=f"Neutron(+desc.) R_cal_charge_dots.\nConsidering {len(R_cal_charge_n_dots_allE_nonzero)} evts where Q_n_dots != 0.\n")
        #---------------------------------------
        if peaks:
            R_cal_charge_peak_mean        = EventsTreeReader.calculateMeanValInPeak(data=R_cal_charge_allE[tplgy_mask_allE],   bin_heights=R_cal_Q_bh,    bin_width=bin_width, bin_left_limit=0)
            # R_cal_Q_artist[0].set_label(   R_cal_Q_artist[0].get_label()    + f" Peak value = {R_cal_charge_peak_mean:.3f}")
            R_cal_charge_e_peak_mean      = EventsTreeReader.calculateMeanValInPeak(data=R_cal_charge_e_allE[tplgy_mask_allE], bin_heights=R_cal_Q_e_bh,  bin_width=bin_width, bin_left_limit=0)
            # R_cal_Q_e_artist[0].set_label( R_cal_Q_e_artist[0].get_label()  + f" Peak value = {R_cal_charge_e_peak_mean:.3f}")
            R_cal_charge_h_peak_mean      = EventsTreeReader.calculateMeanValInPeak(data=R_cal_charge_h_allE[tplgy_mask_allE], bin_heights=R_cal_Q_h_bh,  bin_width=bin_width, bin_left_limit=0)
            # R_cal_Q_h_artist[0].set_label( R_cal_Q_h_artist[0].get_label()  + f" Peak value = {R_cal_charge_h_peak_mean:.3f}")
            R_cal_charge_e_dots_peak_mean = EventsTreeReader.calculateMeanValInPeak(data=R_cal_charge_e_dots_allE_nonzero, bin_heights=R_cal_Q_ed_bh, bin_width=bin_width, bin_left_limit=0)
            # R_cal_Q_ed_artist[0].set_label(R_cal_Q_ed_artist[0].get_label() + f" Peak value = {R_cal_charge_e_dots_peak_mean:.3f}")
            R_cal_charge_h_dots_peak_mean = EventsTreeReader.calculateMeanValInPeak(data=R_cal_charge_h_dots_allE_nonzero, bin_heights=R_cal_Q_hd_bh, bin_width=bin_width, bin_left_limit=0)
            # R_cal_Q_hd_artist[0].set_label(R_cal_Q_hd_artist[0].get_label() + f" Peak value = {R_cal_charge_h_dots_peak_mean:.3f}")

            # For proton / neutron individual graphs
#            R_cal_charge_p_peak_mean      = EventsTreeReader.calculateMeanValInPeak(data=R_cal_charge_p_allE[tplgy_mask_allE], bin_heights=R_cal_Q_p_bh,  bin_width=bin_width, bin_left_limit=0)
#            R_cal_Q_p_artist[0].set_label( R_cal_Q_p_artist[0].get_label()  + f" Peak value = {R_cal_charge_p_peak_mean:.3f}")
#            R_cal_charge_p_dots_peak_mean = EventsTreeReader.calculateMeanValInPeak(data=R_cal_charge_p_dots_allE_nonzero, bin_heights=R_cal_Q_pd_bh, bin_width=bin_width, bin_left_limit=0)
#            R_cal_Q_pd_artist[0].set_label(R_cal_Q_pd_artist[0].get_label() + f" Peak value = {R_cal_charge_p_dots_peak_mean:.3f}")
#            R_cal_charge_n_peak_mean      = EventsTreeReader.calculateMeanValInPeak(data=R_cal_charge_n_allE[tplgy_mask_allE], bin_heights=R_cal_Q_n_bh,  bin_width=bin_width, bin_left_limit=0)
#            R_cal_Q_n_artist[0].set_label( R_cal_Q_n_artist[0].get_label()  + f" Peak value = {R_cal_charge_n_peak_mean:.3f}")
#            R_cal_charge_n_dots_peak_mean = EventsTreeReader.calculateMeanValInPeak(data=R_cal_charge_n_dots_allE_nonzero, bin_heights=R_cal_Q_nd_bh, bin_width=bin_width, bin_left_limit=0)
#            R_cal_Q_nd_artist[0].set_label(R_cal_Q_nd_artist[0].get_label() + f" Peak value = {R_cal_charge_n_dots_peak_mean:.3f}")
            #---------------------------------------
        else:
            R_cal_Q_artist[0].set_label(   R_cal_Q_artist[0].get_label()    + f" Mean value = {np.mean(R_cal_charge_allE[tplgy_mask_allE]):.3f}")
            R_cal_Q_e_artist[0].set_label( R_cal_Q_e_artist[0].get_label()  + f" Mean value = {np.mean(R_cal_charge_e_allE[tplgy_mask_allE]):.3f}")
            R_cal_Q_h_artist[0].set_label( R_cal_Q_h_artist[0].get_label()  + f" Mean value = {np.mean(R_cal_charge_h_allE[tplgy_mask_allE]):.3f}")
            R_cal_Q_ed_artist[0].set_label(R_cal_Q_ed_artist[0].get_label() + f" Mean value = {np.mean(R_cal_charge_e_dots_allE_nonzero):.3f}")
            R_cal_Q_hd_artist[0].set_label(R_cal_Q_hd_artist[0].get_label() + f" Mean value = {np.mean(R_cal_charge_h_dots_allE_nonzero):.3f}")
            # For proton / neutron individual graphs
            #R_cal_Q_p_artist[0].set_label(   R_cal_Q_p_artist[0].get_label()    + f" Mean value = {np.mean(R_cal_charge_p_allE[tplgy_mask_allE]):.3f}")
            #R_cal_Q_pd_artist[0].set_label(   R_cal_Q_pd_artist[0].get_label()    + f" Mean value = {np.mean(R_cal_charge_p_dots_allE_nonzero):.3f}")
            #R_cal_Q_n_artist[0].set_label(   R_cal_Q_n_artist[0].get_label()    + f" Mean value = {np.mean(R_cal_charge_n_allE[tplgy_mask_allE]):.3f}")
            #R_cal_Q_nd_artist[0].set_label(   R_cal_Q_nd_artist[0].get_label()    + f" Mean value = {np.mean(R_cal_charge_n_dots_allE_nonzero):.3f}")
            #---------------------------------------
        Qax.plot([], [], lw=0, label=r"(dQ $>$ 75keV)")
        Qax.tick_params('x', direction='in')
        Qax.set_ylabel("Number of events", fontsize="x-large")
        Qax.set_title(r"$\qquad R_{\text{cal, charge}}$", loc="left", y=0.8, fontsize=25)
        Qax.legend(fontsize="large")

        R_cal_QL_bh, _, R_cal_QL_artist     = QLax.hist(R_cal_QL_allE[tplgy_mask_allE],   bins=bin_edges, histtype="step", lw=1.3, label=f"Overall")
        R_cal_QL_e_bh, _, R_cal_QL_e_artist = QLax.hist(R_cal_QL_e_allE[tplgy_mask_allE], bins=bin_edges, histtype="step", lw=1.3, label=f"EM")
        R_cal_QL_h_bh, _, R_cal_QL_h_artist = QLax.hist(R_cal_QL_h_allE[tplgy_mask_allE], bins=bin_edges, histtype="step", lw=1.3, label=f"Hadronic")

        # For proton / neutron individual graphs
#        R_cal_QL_p_bh, _, R_cal_QL_p_artist = QLax.hist(R_cal_QL_p_allE[tplgy_mask_allE], bins=bin_edges, histtype="step", label=f"Proton(+desc.) R_cal_Q+L.")
#        R_cal_QL_n_bh, _, R_cal_QL_n_artist = QLax.hist(R_cal_QL_n_allE[tplgy_mask_allE], bins=bin_edges, histtype="step", label=f"Neutron(+desc.) R_cal_Q+L.")
        #---------------------------------------
        if peaks:
            R_cal_QL_peak_mean = EventsTreeReader.calculateMeanValInPeak(  data=R_cal_QL_allE[tplgy_mask_allE],   bin_heights=R_cal_QL_bh,   bin_width=bin_width, bin_left_limit=0)
            # R_cal_QL_artist[0].set_label(  R_cal_QL_artist[0].get_label()   + f" Peak value = {R_cal_QL_peak_mean:.3f}")
            R_cal_QL_e_peak_mean = EventsTreeReader.calculateMeanValInPeak(data=R_cal_QL_e_allE[tplgy_mask_allE], bin_heights=R_cal_QL_e_bh, bin_width=bin_width, bin_left_limit=0)
            # R_cal_QL_e_artist[0].set_label(R_cal_QL_e_artist[0].get_label() + f" Peak value = {R_cal_QL_e_peak_mean:.3f}")
            R_cal_QL_h_peak_mean = EventsTreeReader.calculateMeanValInPeak(data=R_cal_QL_h_allE[tplgy_mask_allE], bin_heights=R_cal_QL_h_bh, bin_width=bin_width, bin_left_limit=0)
            # R_cal_QL_h_artist[0].set_label(R_cal_QL_h_artist[0].get_label() + f" Peak value = {R_cal_QL_h_peak_mean:.3f}")

            # For proton / neutron individual graphs
#            R_cal_QL_p_peak_mean = EventsTreeReader.calculateMeanValInPeak(data=R_cal_QL_p_allE[tplgy_mask_allE], bin_heights=R_cal_QL_p_bh, bin_width=bin_width, bin_left_limit=0)
#            R_cal_QL_p_artist[0].set_label(R_cal_QL_p_artist[0].get_label() + f" Peak value = {R_cal_QL_p_peak_mean:.3f}")
#            R_cal_QL_n_peak_mean = EventsTreeReader.calculateMeanValInPeak(data=R_cal_QL_n_allE[tplgy_mask_allE], bin_heights=R_cal_QL_n_bh, bin_width=bin_width, bin_left_limit=0)
#            R_cal_QL_n_artist[0].set_label(R_cal_QL_n_artist[0].get_label() + f" Peak value = {R_cal_QL_n_peak_mean:.3f}")
            #---------------------------------------
        else:
            R_cal_QL_artist[0].set_label(  R_cal_QL_artist[0].get_label()   + f" Mean value = {np.mean(R_cal_QL_allE[tplgy_mask_allE]):.3f}")
            R_cal_QL_e_artist[0].set_label(  R_cal_QL_e_artist[0].get_label()   + f" Mean value = {np.mean(R_cal_QL_e_allE[tplgy_mask_allE]):.3f}")
            R_cal_QL_h_artist[0].set_label(  R_cal_QL_h_artist[0].get_label()   + f" Mean value = {np.mean(R_cal_QL_h_allE[tplgy_mask_allE]):.3f}")
            # For proton / neutron individual graphs
            #R_cal_QL_p_artist[0].set_label(  R_cal_QL_p_artist[0].get_label()   + f" Mean value = {np.mean(R_cal_QL_p_allE[tplgy_mask_allE]):.3f}")
            #R_cal_QL_n_artist[0].set_label(  R_cal_QL_n_artist[0].get_label()   + f" Mean value = {np.mean(R_cal_QL_n_allE[tplgy_mask_allE]):.3f}")
            #---------------------------------------
        QLax.set_xlabel("Calorimeter response ratio", fontsize="x-large")
        QLax.set_ylabel("Number of events", fontsize="x-large")
        QLax.set_title(r"$\qquad R_{\text{cal, total}}$", loc="left", y=0.8, fontsize=25)
        QLax.legend(fontsize="large", loc='upper center')

        #if len(self.event_energies.keys()) > 1:
        #    plt.savefig("./plots/" + self.out_filename[:-4] + "_" + str(int(E_nu)) + "MeV.pdf")
        #else:
        #    plt.savefig("./plots/" + self.out_filename)
        plt.savefig("./plots/" + self.out_filename)
        plt.clf()
        plt.close()
        print("Plot created.")

    #---------------------------------------------------
    def plotE_recResolutions(self, ep=False, iqr=False):
        res = self.resolutions
        bias = self.biases
        info = self.event_energies
        sorter = np.argsort(list(self.event_energies.keys()))

        L1res = np.array(list(res['E_rec_L1_res'].items())) # res['...'].items() returns view object of list of tuples [(E_nu1, res1), (E_nu2, res2)...]
        Q1res = np.array(list(res['E_rec_Q1_res'].items())) # list(res['...'].items()) returns list of tuples [(E_nu1, res1), (E_nu2, res2)...]
        Q2res = np.array(list(res['E_rec_Q2_res'].items())) # np.array(list(res['..'].items())) returns 2D array where each tuple is a row.
        Q3res = np.array(list(res['E_rec_Q3_res'].items()))
        QL1res = np.array(list(res['E_rec_(Q+L)1_res'].items()))
        # QL2res = np.array(list(res['E_rec_(Q+L)2_res'].items()))

        L1b = np.array(list(bias['E_rec_L1_b'].items()))
        Q1b = np.array(list(bias['E_rec_Q1_b'].items()))
        Q2b = np.array(list(bias['E_rec_Q2_b'].items()))
        Q3b = np.array(list(bias['E_rec_Q3_b'].items()))
        QL1b = np.array(list(bias['E_rec_(Q+L)1_b'].items()))
        # QL2b = np.array(list(bias['E_rec_(Q+L)2_b'].items()))

        # L1iqr, Q1iqr, Q2iqr, Q3iqr, QL1iqr, QL2iqr = [np.array(list(self.iqr['E_rec_'+rec+'_iqr'].items())) for rec in ["L1", "Q1", "Q2", "Q3", "(Q+L)1", "(Q+L)2"]]
        L1iqr, Q1iqr, Q2iqr, Q3iqr, QL1iqr = [np.array(list(self.iqr['E_rec_'+rec+'_iqr'].items())) for rec in ["L1", "Q1", "Q2", "Q3", "(Q+L)1"]]

        if ep:
            L1res_1p = np.array(list(res['L1_res_1p'].items()))
            Q1res_1p = np.array(list(res['Q1_res_1p'].items()))
            Q2res_1p = np.array(list(res['Q2_res_1p'].items()))
            Q3res_1p = np.array(list(res['Q3_res_1p'].items()))
            QL1res_1p = np.array(list(res['(Q+L)1_res_1p'].items()))
            QL2res_1p = np.array(list(res['(Q+L)2_res_1p'].items()))

            L1b_1p = np.array(list(bias['L1_b_1p'].items()))
            Q1b_1p = np.array(list(bias['Q1_b_1p'].items()))
            Q2b_1p = np.array(list(bias['Q2_b_1p'].items()))
            Q3b_1p = np.array(list(bias['Q3_b_1p'].items()))
            QL1b_1p = np.array(list(bias['(Q+L)1_b_1p'].items()))
            QL2b_1p = np.array(list(bias['(Q+L)2_b_1p'].items()))

        if self.option == "E_dep_more":
            L1res, Q1res, Q2res, Q3res = [np.array(list(self.ideal_res[rec_str].items())) for rec_str in ["L1", "Q1", "Q2", "Q3"]]
            L1b, Q1b, Q2b, Q3b         = [np.array(list(self.ideal_b[b_str].items()))     for b_str   in ["L1", "Q1", "Q2", "Q3"]]
            Lres, Qres = [np.array(list(self.QL_res[dep_str].items())) for dep_str in ["L", "Q"]]

        print(list(res['L1_res_1p'].items()))

        #plt.figure()
        fig, (Rax, Bax) = plt.subplots(2, figsize=(6.4, 9.6))
        fig.set_layout_engine('tight')
        # Rax.set_title("Resolutions", loc="left", fontsize="small")
        if iqr:
            Rax.set_ylabel(r"Resolution (IQR / Mean) [\%]", fontsize="large")
        else:
            Rax.set_ylabel(r"Resolution (RMS / Mean) [\%]", fontsize="large")
        # Bax.set_title("Relative Biases (Mean (reco - true)/true)", loc="left", fontsize="small")
        Bax.set_ylabel(r"Relative Bias [\%]", fontsize="large")
        # cl_list = [("C3", "L1"), ("black", "Q1"), ("C2", "Q2"), ("blue", "Q3"), ("magenta", "(Q+L)1"), ("C8", "(Q+L)2")]
        cl_list = [("C3", "L1"), ("black", "Q1"), ("C2", "Q2"), ("blue", "Q3"), ("magenta", "Q+L")]
        if not ep:
            if iqr:
                # res_list =  [L1iqr, Q1iqr, Q2iqr, Q3iqr, QL1iqr, QL2iqr]
                res_list =  [L1iqr, Q1iqr, Q2iqr, Q3iqr, QL1iqr]
            else:
                # res_list =  [L1res, Q1res, Q2res, Q3res, QL1res, QL2res]
                res_list =  [L1res, Q1res, Q2res, Q3res, QL1res]
            # bias_list = [L1b,   Q1b,   Q2b,   Q3b,   QL1b,   QL2b]
            bias_list = [L1b,   Q1b,   Q2b,   Q3b,   QL1b]
            # fig.suptitle(f"{sum([len(info[E_nu]['E_rec_L1']) for E_nu in L1res[:,0]])} events, nu:{info[L1res[0, 0]]['nu_pdg'][0]}, shift: +{self.E_a_to_E_nu_shift} MeV")
        else:
            res_list =  [L1res_1p, Q1res_1p, Q2res_1p, Q3res_1p, QL1res_1p, QL2res_1p]
            bias_list = [L1b_1p,   Q1b_1p,   Q2b_1p,   Q3b_1p,   QL1b_1p,   QL2b_1p]
            fig.suptitle(f"{sum([len(info[E_nu]['E_rec_L1_1p']) for E_nu in L1res_1p[:,0]])} / {sum([len(info[E_nu]['E_rec_L1']) for E_nu in L1res[:,0]])} 1p events, nu:{info[L1res[0, 0]]['nu_pdg'][0]}, shift: +{self.E_a_to_E_nu_shift_1p} MeV")

        if self.option == "E_dep_more":
            res_list = res_list[:-2]
            bias_list = bias_list[:-2]

        # To exclude Q+L
        # res_list = res_list[:-2]
        # bias_list = bias_list[:-2]

        for i, res_data in enumerate(res_list):
            Rax.plot(res_data[sorter, 0], res_data[sorter, 1] * 100, color=cl_list[i][0], label=cl_list[i][1])
        for i, bias_data in enumerate(bias_list):
            Bax.plot(bias_data[sorter, 0], bias_data[sorter, 1] * 100, color=cl_list[i][0], label=cl_list[i][1])

        for ax in (Rax, Bax):
            ax.set_xlabel(r"Incident neutrino energy $E_\nu$ [MeV]", fontsize="large")
            ax.legend(fontsize="large")

        plt.savefig("./plots/" + self.out_filename)
        plt.clf()
        plt.close()
        print("Plot created.")

    #---------------------------------------------------
    #def plotMultiplicityHistogram(self, E_nu):
    def plotMultiplicityHistogram(self):
        N_parl_allE = np.empty((8, 0))
        nu_proc_allE = np.array([])

        for E_nu in self.event_energies.keys():
            info = self.event_energies[E_nu]
            nu_pdg = info['nu_pdg'][0]
            N_parl  = info['N_par_list']
            nu_proc = info['interaction']
            N_parl_allE = np.append(N_parl_allE, N_parl, axis=1)
            nu_proc_allE = np.append(nu_proc_allE, nu_proc)

        #max_nucl = int(N_parl[[1,2]].max())
        max_nucl = int(N_parl_allE[[1,2]].max())
        bin_edges = np.linspace(-0.5, max_nucl+0.5, max_nucl+2)

        fig, axs = plt.subplots(1, 2)
        fig.set_layout_engine('tight')
        fig.set_size_inches(12.8, 4.8)
        #fig.suptitle(f"E_nu = {E_nu} MeV, {info['nEvents']} events")
        fig.suptitle(f"nu: {nu_pdg}, E_nu from {list(self.event_energies.keys())[0]} to {list(self.event_energies.keys())[-1]} MeV, {N_parl_allE.shape[1]} events total")

        #proc_masks = [
        #    ('QES', (info['interaction'] == 11)),
        #    ('RES', (info['interaction'] == 12)),
        #    ('DIS', (info['interaction'] == 13)),
        #    ('COH', (info['interaction'] == 14)),
        #    ('MEC', (info['interaction'] == 15)),
        #]
        proc_masks = [
            ('QES', (nu_proc_allE == 11)),
            ('RES', (nu_proc_allE == 12)),
            ('DIS', (nu_proc_allE == 13)),
            ('COH', (nu_proc_allE == 14)),
            ('MEC', (nu_proc_allE == 15)),
        ]

        #N_nucleon_breakdown = [N_parl[[1,2]][:, mask] for proc, mask in proc_masks]
        N_nucleon_breakdown = [N_parl_allE[[1,2]][:, mask] for proc, mask in proc_masks]
        N_nucleon_labels = [f"{proc}, {len(N_nucleon_breakdown[i][0])} events total" for i, (proc, mask) in enumerate(proc_masks)]

        # Creates a stacked bar chart of no. of events with each proton multiplicity broken down by the interaction type of the event.
        cur_level = np.zeros(max_nucl+1)
        for i, N_nucleon_dist in enumerate(N_nucleon_breakdown):
            proton_hist, _ = np.histogram(N_nucleon_dist[0], bins=bin_edges)
            axs[0].bar([f'{a}' for a in range(0, max_nucl+1)], proton_hist, bottom=cur_level, width=1, edgecolor="black", hatch="//", label=N_nucleon_labels[i])
            cur_level += proton_hist
        # axs[0].hist(N_parl[1], bins=bin_edges, histtype="step")
        axs[0].set_xlabel("Proton multiplicity")
        axs[0].set_ylabel("No. of events")
        axs[0].legend(fontsize='small')

        cur_level = np.zeros(max_nucl+1)
        for i, N_nucleon_dist in enumerate(N_nucleon_breakdown):
            neutron_hist, _ = np.histogram(N_nucleon_dist[1], bins=bin_edges)
            axs[1].bar([f'{a}' for a in range(0, max_nucl+1)], neutron_hist, bottom=cur_level, width=1, edgecolor="black", hatch="//", label=N_nucleon_labels[i])
            cur_level += neutron_hist
        # axs[1].hist(N_parl[2], bins=bin_edges, histtype="step")
        axs[1].set_xlabel("Neutron multiplicity")
        axs[1].set_ylabel("No. of events")
        axs[1].legend(fontsize='small')

        #if len(self.event_energies.keys()) > 1:
        #    plt.savefig("./plots/" + self.out_filename[:-4] + "_" + str(int(E_nu)) + "MeV.pdf")
        #else:
        #    plt.savefig("./plots/" + self.out_filename)
        plt.savefig("./plots/" + self.out_filename)
        plt.clf()
        plt.close()
        print("Plot created.")

    #---------------------------------------------------
    def plotMultiplicityE_availScatter(self, E_nu, fs_breakdown=False):
        info = self.event_energies[E_nu]
        E_a, N_pl = info['E_avail'], info['N_par_list']
        #E_a, N_pl = info['E_avail_gst'], info['N_par_list_gst']

        if fs_breakdown:
            fig, axs = plt.subplots(3, 2) # First column broken down by process, second broken down by particles present in final state.
        else:
            fig, axs = plt.subplots(3) # 3 - nucleons, protons, neutrons
        fig.suptitle(f"E_nu = {E_nu} MeV, {info['nEvents']} events.")
        if fs_breakdown:
            fig.set_size_inches(13, 14.4)
        else:
            fig.set_size_inches(6.5, 14.4)
        fig.set_layout_engine("tight")

        proc_masks = [
            ('QES', (info['interaction'] == 11)),
            ('RES', (info['interaction'] == 12)),
            ('DIS', (info['interaction'] == 13)),
            ('COH', (info['interaction'] == 14)),
            ('MEC', (info['interaction'] == 15)),
        ]

        E_a_breakdown = [E_a[mask] for proc, mask in proc_masks]
        N_nucleon_breakdown = [N_pl[[1,2]][:, mask] for proc, mask in proc_masks] # N_nucleon_breakdown is a list of np.arrays, each of
                                                                                  # shape (2, x). x = no. of events of certain interaction type.
                                                                                  # row 0 gives no. of protons in each event, row 1 the neutrons.
        labels = [f"{proc}, {len(E_a_breakdown[i])} events total" for i, (proc, mask) in enumerate(proc_masks)]

        col1 = (axs[:, 0] if fs_breakdown else axs)
        for i in range(len(proc_masks)):     # For the first column
            nucl, = col1[0].plot(E_a_breakdown[i], N_nucleon_breakdown[i].sum(axis=0), '.', label=labels[i])
            p,    = col1[1].plot(E_a_breakdown[i], N_nucleon_breakdown[i][0], '.', label=labels[i])
            n,    = col1[2].plot(E_a_breakdown[i], N_nucleon_breakdown[i][1], '.', label=labels[i])
            [l2d.set_markersize((2 if i == 0 else 2)) for l2d in [nucl, p, n]]

        if fs_breakdown:
                                # (electron,       proton,         neutron,        pi+-,           pi0          )
            FS_masks = [
                ('e|μ only',         ((N_pl[0] > 0) & (N_pl[1]== 0) & (N_pl[2]== 0) & (N_pl[3]== 0) & (N_pl[4]== 0))),
                ('e|μ,p',            ((N_pl[0] > 0) & (N_pl[1] > 0) & (N_pl[2]== 0) & (N_pl[3]== 0) & (N_pl[4]== 0))),
                ('e|μ,p,pi0',        ((N_pl[0] > 0) & (N_pl[1] > 0) & (N_pl[2]== 0) & (N_pl[3]== 0) & (N_pl[4] > 0))),
                ('e|μ,p,pi+-',       ((N_pl[0] > 0) & (N_pl[1] > 0) & (N_pl[2]== 0) & (N_pl[3] > 0) & (N_pl[4]== 0))),
                ('e|μ,p,pi0,pi+-',   ((N_pl[0] > 0) & (N_pl[1] > 0) & (N_pl[2]== 0) & (N_pl[3] > 0) & (N_pl[4] > 0))),
                ('e|μ,n',            ((N_pl[0] > 0) & (N_pl[1]== 0) & (N_pl[2] > 0) & (N_pl[3]== 0) & (N_pl[4]== 0))),
                ('e|μ,n,pi0',        ((N_pl[0] > 0) & (N_pl[1]== 0) & (N_pl[2] > 0) & (N_pl[3]== 0) & (N_pl[4] > 0))),
                ('e|μ,n,pi+-',       ((N_pl[0] > 0) & (N_pl[1]== 0) & (N_pl[2] > 0) & (N_pl[3] > 0) & (N_pl[4]== 0))),
                ('e|μ,n,pi0,pi+-',   ((N_pl[0] > 0) & (N_pl[1]== 0) & (N_pl[2] > 0) & (N_pl[3] > 0) & (N_pl[4] > 0))),
                ('e|μ,p,n',          ((N_pl[0] > 0) & (N_pl[1] > 0) & (N_pl[2] > 0) & (N_pl[3]== 0) & (N_pl[4]== 0))),
                ('e|μ,p,n,pi0',      ((N_pl[0] > 0) & (N_pl[1] > 0) & (N_pl[2] > 0) & (N_pl[3]== 0) & (N_pl[4] > 0))),
                ('e|μ,p,n,pi+-',     ((N_pl[0] > 0) & (N_pl[1] > 0) & (N_pl[2] > 0) & (N_pl[3] > 0) & (N_pl[4]== 0))),
                ('e|μ,p,n,pi0,pi+-', ((N_pl[0] > 0) & (N_pl[1] > 0) & (N_pl[2] > 0) & (N_pl[3] > 0) & (N_pl[4] > 0))),
                ('e|μ,pi0',          ((N_pl[0] > 0) & (N_pl[1]== 0) & (N_pl[2]== 0) & (N_pl[3]== 0) & (N_pl[4] > 0))),
                ('e|μ,pi+-',         ((N_pl[0] > 0) & (N_pl[1]== 0) & (N_pl[2]== 0) & (N_pl[3] > 0) & (N_pl[4]== 0))),
            ]
            FS_colors = ('gray', 'lightcoral', 'coral', 'crimson', 'maroon', 'palegreen', 'yellowgreen', 'olive', 'darkolivegreen', 'lightskyblue', 'turquoise', 'dodgerblue', 'darkblue', 'violet', 'darkviolet')
            E_a_breakdown = [E_a[mask] for FS, mask in FS_masks]
            N_nucleon_breakdown = [N_pl[[1,2]][:, mask] for FS, mask in FS_masks]
            labels = [f"{FS} post-FSI, {len(E_a_breakdown[i])} events total" if len(E_a_breakdown[i]) > 0 else "" for i, (FS, mask) in enumerate(FS_masks)]

            for i in range(len(FS_masks)):       # For the second column
                axs[0, 1].plot(E_a_breakdown[i], N_nucleon_breakdown[i].sum(axis=0), '.', markersize=4, color=FS_colors[i], label=labels[i])
                axs[1, 1].plot(E_a_breakdown[i], N_nucleon_breakdown[i][0], '.', markersize=4, color=FS_colors[i], label=labels[i])
                axs[2, 1].plot(E_a_breakdown[i], N_nucleon_breakdown[i][1], '.', markersize=4, color=FS_colors[i], label=labels[i])

            axs[0, 0].set_ylabel("Number of nucleons in final state")
            #axs[0, 0].set_ylabel("Number of 'primary' nucleons pre-FSI")

            axs[1, 0].set_ylabel("Number of protons in final state")
            #axs[1, 0].set_ylabel("Number of 'primary' protons pre-FSI")

            axs[2, 0].set_ylabel("Number of neutrons in final state")
            #axs[2, 0].set_ylabel("Number of 'primary' neutrons pre-FSI")


        for i, ax in enumerate(axs.flat):    # Applies to all subplots
            ax.set_xlabel("Available energy of event post-FSI [MeV]", fontsize="small")
            # ax.set_xlim(230.9, 231.7) # For analysis
            ax.set_yticks(np.arange(0, int(np.ceil(ax.get_ylim()[1])), int(np.ceil(ax.get_ylim()[1])) // 10)) # Ensures ~10 ticks at integer values
            ax.legend(fontsize="small")
            if (fs_breakdown and (i < axs.shape[1])) or (i == 0):       # Exploiting lazy evaluation
                ax.set_title("(All nucleons)", x=0.2, y=0.8)
            elif (fs_breakdown and (i < 2 * axs.shape[1])) or (i == 1):
                ax.set_title("(Protons)", x=0.2, y=0.8)
            else:
                ax.set_title("(Neutrons)", x=0.2, y=0.8)

        if len(self.event_energies.keys()) > 1:
            plt.savefig("./plots/" + self.out_filename[:-4] + "_" + str(int(E_nu)) + "MeV.pdf")
        else:
            plt.savefig("./plots/" + self.out_filename)

        plt.clf()
        plt.close()
        print("Plot created.")

    #---------------------------------------------------
    def plotFSIStats(self):
        E_nu_vals = np.array(list(self.event_energies.keys()))
        E_nu_ct = len(E_nu_vals)
        fsi_stats = {"E_a_increased_ct" : np.zeros(E_nu_ct), "E_a_decreased_ct" : np.zeros(E_nu_ct), "E_a_unchanged_ct" : np.zeros(E_nu_ct), 
                     "Avg_E_a_increase" : np.zeros(E_nu_ct), "Avg_E_a_decrease" : np.zeros(E_nu_ct), "Avg_E_a_change" : np.zeros(E_nu_ct),
                     "Std_E_a_increase" : np.zeros(E_nu_ct), "Std_E_a_decrease" : np.zeros(E_nu_ct), "Std_E_a_change" : np.zeros(E_nu_ct),
                     "E_a_changes" : np.array([]), "Nucl_ct_changes" : np.array([])}

        for i, E_nu in enumerate(E_nu_vals):
            info = self.event_energies[E_nu]
            E_a, E_a_pFS = info["E_avail_gst"], info["E_avail_pre_FSI_gst"] # E_avail from gst and edepsim files differ by on average order 0.01 MeV, I think
            N_pl = info['N_par_list']
            N_pl_pFS = info['N_par_list_pre_FSI_gst'] # New version with 11 places (indices 8, 9, 10 are for strange baryons)

            N_pl_pFS_tmp = N_pl_pFS.copy()    
            N_pl_pFS_tmp[7] = np.sum(N_pl_pFS_tmp[7:], axis=0)
            N_pl_pFS = N_pl_pFS_tmp[:8]   # Recollect indices 8,9,10 into index 7 (other)
            
            N_nucl     = np.sum(N_pl[[1,2]],     axis=0)
            N_nucl_pFS = np.sum(N_pl_pFS[[1,2]], axis=0)

            fsi_stats['E_a_increased_ct'][i] = np.sum( (E_a > E_a_pFS ) )
            fsi_stats['E_a_decreased_ct'][i] = np.sum( (E_a < E_a_pFS ) )
            fsi_stats['E_a_unchanged_ct'][i] = np.sum( (E_a == E_a_pFS ) )
            fsi_stats['Avg_E_a_increase'][i] = np.mean( (E_a - E_a_pFS)[E_a > E_a_pFS] )
            fsi_stats['Avg_E_a_decrease'][i] = np.mean( (E_a_pFS - E_a)[E_a < E_a_pFS] )
            fsi_stats['Avg_E_a_change'][i]   = np.mean( (E_a - E_a_pFS) )
            fsi_stats['Std_E_a_increase'][i] = np.std( (E_a - E_a_pFS)[E_a > E_a_pFS] )
            fsi_stats['Std_E_a_decrease'][i] = np.std( (E_a_pFS - E_a)[E_a < E_a_pFS] )
            fsi_stats['Std_E_a_change'][i]   = np.std( (E_a - E_a_pFS) )

            fsi_stats['E_a_changes']     = np.append(fsi_stats['E_a_changes'],     (E_a - E_a_pFS))
            fsi_stats['Nucl_ct_changes'] = np.append(fsi_stats['Nucl_ct_changes'], (N_nucl - N_nucl_pFS))

        fig, axs = plt.subplots(3, figsize=(6.5, 4.81*3))
        fig.set_layout_engine('tight')
        fig.suptitle(f"nu:{info['nu_pdg'][0]}, E_nu from {E_nu_vals[0]} to {E_nu_vals[-1]} MeV, {len(fsi_stats['E_a_changes'])} events. Energy info from GST.", y=0.99, fontsize='medium')

        axs[0].set_title("# of events where FSI increases/decreases/leaves unchanged E_avail as a function of E_nu", fontsize="small", loc="left")
        axs[0].set_xlabel("E_nu [MeV]")
        axs[0].set_ylabel("No. of events")
        axs[0].plot(E_nu_vals, fsi_stats['E_a_increased_ct'], 'b-', label="FSI increases E_avail")
        axs[0].plot(E_nu_vals, fsi_stats['E_a_decreased_ct'], 'C1-', label="FSI decreases E_avail")
        axs[0].plot(E_nu_vals, fsi_stats['E_a_unchanged_ct'], 'k-', label="FSI doesn't change E_avail")
        axs[0].legend(fontsize="medium")

        axs[1].set_title("Average changes in E_avail due to FSI as a function of E_nu. Errorbar is $\pm 1\sigma$", fontsize="small", loc="left")
        axs[1].set_xlabel("E_nu [MeV]")
        axs[1].set_ylabel("Change in E_avail [MeV]")
        axs[1].plot(E_nu_vals, np.zeros(E_nu_ct), "--", color='grey', alpha=0.3)
        axs[1].errorbar(E_nu_vals, fsi_stats['Avg_E_a_increase'],  yerr=fsi_stats['Std_E_a_increase'], fmt='bo-',  elinewidth=0.3, capsize=1, markersize=2.5, label=f"Avg increase in E_avail due to FSI among {fsi_stats['E_a_increased_ct'].sum()} evts where increase occurs")
        axs[1].errorbar(E_nu_vals, -fsi_stats['Avg_E_a_decrease'], yerr=fsi_stats['Std_E_a_decrease'], fmt='C1o-', elinewidth=0.8, capsize=1, markersize=2.5, label=f"Avg decrease in E_avail due to FSI among {fsi_stats['E_a_decreased_ct'].sum()} evts where decrease occurs")
        axs[1].errorbar(E_nu_vals, fsi_stats['Avg_E_a_change'],    yerr=fsi_stats['Std_E_a_change'],   fmt='ko-',  elinewidth=0.8, capsize=1, markersize=2.5, label=f"Avg change in E_avail due to FSI")
        axs[1].legend(fontsize="small")

        axs[2].set_title(f"Change in E_avail vs change in nucleon multiplicity, over all {len(fsi_stats['E_a_changes'])} evts", fontsize="small", loc="left")
        axs[2].set_xlabel("Change in nucleon multiplicity due to FSI")
        axs[2].set_ylabel("Change in E_avail due to FSI [MeV]")
        axs[2].plot(fsi_stats['Nucl_ct_changes'], np.zeros(len(fsi_stats['Nucl_ct_changes'])), "--", color='grey', alpha=0.3)
        axs[2].plot(fsi_stats['Nucl_ct_changes'], fsi_stats['E_a_changes'],  '.', markersize=2, alpha=0.5)

        fig.savefig("./plots/" + self.out_filename)

        fig.clf()
        plt.close(fig)
        print("Plot created.")

    #---------------------------------------------------
    def plotPreFSIParticles(self, E_nu):
        info = self.event_energies[E_nu]
        E_a_pFS, N_pl_pFS = info['E_avail_pre_FSI_gst'], info['N_par_list_pre_FSI_gst']

        proc_masks = [
            ('QES', (info['interaction_gst'] == 11)),
            ('RES', (info['interaction_gst'] == 12)),
            ('DIS', (info['interaction_gst'] == 13)),
            ('COH', (info['interaction_gst'] == 14)),
            ('MEC', (info['interaction_gst'] == 15)),
        ]

        E_a_pFS_breakdown  = [E_a_pFS[mask] for (proc, mask) in proc_masks]
        N_pl_pFS_breakdown = [N_pl_pFS[:, mask] for (proc, mask) in proc_masks]
        labels = [f"{proc}, {len(E_a_pFS_breakdown[i])} events" for i, (proc, mask) in enumerate(proc_masks)]
        titles  = ["protons", "neutrons", "pi+-s", "pi0s"]

        fig, axs = plt.subplots(4, figsize=(8, 2*4))
        fig.suptitle(f"nu:{info['nu_pdg'][0]}, E_nu = {E_nu} MeV, {info['nEvents']} events. Info from GST")
        fig.set_layout_engine("constrained")
        for i, ax in enumerate(axs):            # i is over the particle type
            for j in range(len(proc_masks)):    # j is over the interaction type
                ax.plot(E_a_pFS_breakdown[j], N_pl_pFS_breakdown[j][i+1], ".", markersize=2, alpha=0.8, label=labels[j])
            ax.set_xlabel("E_avail pre-FSI", fontsize="small")
            ax.set_ylabel(f"No. of {titles[i]} pre-FSI", fontsize="small")
            ax.set_title(titles[i], loc="left", fontsize="small")

        axs[0].legend(fontsize="xx-small", ncols=3)

        self.SaveAndClose(E_nu)

    #------------------------------------------------------------------------------------------------------------------------------------------------
    # This was not a great idea, preserved for records' sake. It was fun to make though
    # 1000 events are too many to plot along one x axis and make inferences
    # Instead only every <event_step> entries are considered. For each considered event (entry),
    #     The distribution of primary particles is given in a stacked bar (the total bar height is the total no. of primary particles produced)
    #     E_avail and E_dep are given in two bars next to it.
    def plotN_par(self, E_nu):
        info = self.event_energies[E_nu]
        bin_width = E_nu / 100
        bin_edges = np.linspace(0, 1.25*E_nu, 126)

        event_step = 25                                      # We are looking at only every <event_step> events
        N_parl = info['N_par_list']                          # Should be an 8 rows x nEvents columns array
        N_parl_sample = N_parl[:, ::event_step]              # All 8 rows, every <event_step>th columm (column 0, event_step, 2*event_step...)
        entryNos = np.arange(0, info['nEvents'], event_step) # 1D array of length nEvents / event_step

        E_a, E_d = info['E_avail'], info['E_dep']

        fig = plt.figure(figsize=[14.5, 6.5], layout='tight')
        gs = fig.add_gridspec(1, 2, wspace=0, width_ratios=[0.875,0.125])
        Nax, Hax = gs.subplots()                             # Hax = Histogram Axes
        Eax = Nax.twinx()                                    # Creates duplicate Axes with invisible x axis and y axis positioned opposite to Nax
        fig.suptitle(f"E_nu = {E_nu} MeV, bin width = {bin_width} MeV, {info['nEvents']} events")

        w = event_step / 4                                   # 2 bars, four spaces for legibility
                                                             # align='edge' means left edge of bar is at x
        par_names = ('electron', 'proton', 'neutron', 'pi+-', 'pi0')
        par_colors = ('yellow', 'magenta', 'green', 'pink', 'indigo')
        cur_level = np.zeros(len(N_parl_sample[0]))          # len(N_parl_sample) = no. of rows = 8, we want no. of columns
        for i in range(5):
            Nax.bar(entryNos, N_parl_sample[i], bottom=cur_level, align='edge', width=w, color=par_colors[i], label=f"{par_names[i]} count")
            cur_level += N_parl_sample[i]

        Nax.set_xlabel("Entry no.")
        Nax.set_ylabel("Primary particle count")
        Nax.set_title(f"Every {event_step}th event")
        Nax.legend(loc='upper left', fontsize="x-small")

        # info['E_avail'] and info['E_dep'] are 1D arrays
        Eax.bar(entryNos+w, E_a[::event_step], align='edge', width=w/2, color='C0', label="E_avail")
        Eax.bar(entryNos+w*1.5, E_d[::event_step], align='edge', width=w/2, color='C1', label="E_dep")
        Eax.set_ylim(2*(E_nu-40)/3, 1.05*max(E_a.max(), E_d.max())) # Sort of arbitrary, to help distinguish the two main peaks
        Eax.legend(fontsize="x-small")

        Hax.hist(E_a, bins=bin_edges, histtype="step", orientation='horizontal', label="E_avail")
        Hax.hist(E_d, bins=bin_edges, histtype="step", orientation='horizontal', label="E_dep")
        Hax.sharey(Eax)                                      # The Histogram and Energy bar chart share a y axis.
        Hax.tick_params('y', labelleft=False)                # To prevent double tick labels
        Hax.set_xlabel("No. of events")
        Hax.set_ylabel("Energy [MeV]", y=0.8)                # Only create the y label once, for use by both Eax and Hax
        Hax.set_title("All events")
        Hax.legend(fontsize="x-small")

        if len(self.event_energies.keys()) > 1:
            plt.savefig("./plots/" + self.out_filename[:-4] + "_" + str(int(E_nu)) + "MeV.pdf")
        else:
            plt.savefig("./plots/" + self.out_filename)

        plt.clf()
        plt.close()
        print("Plot created.")


# checks if we are running from command line
if __name__ == "__main__":

    # Options
    # E_dep_hist generates a histogram of E_avail, E_dep, Q, Q_thre and L for an input file of given E_nu.
    # E_rec_hist generates a histogram of E_rec via L1, Q1, Q2 and Q3 for an input file of given E_nu.
    # E_rec_resolutions generates a plot of the resolution in E_rec via L1, Q1, Q2, Q3 vs E_nu. NOT YET ACCURATE
    # E_dep_particles generates histograms of E_avail and E_dep for each individual particle
    # E_dep_processes generates histogram of E_avail and E_dep for each type of nu-Ar CC interaction
    # E_dep_fs generates histogram of E_avail and E_dep for each combination of primary particles formed in the nu-Ar CC interaction
    # N_par
    # R_cal_hist generates a histogram of the calorimeter responses of the different events of a given E_nu in one .root file.
    # multiplicity_hist gives the distribution of the proton and neutron multiplicities of the events of given E_nu in a written .root file.
    # multiplicity_scatter plots nucleon multiplicity vs E_avail for events of given E_nu in a written .root file
    # E_dep_scatter plots E_dep vs E_avail
    # E_avail_pre_FSI

    if len(sys.argv) < 2:
        print("Must specify a .root file to read from!")
        sys.exit()

    elif len(sys.argv) < 3:
        e = EventsTreeReader(sys.argv[1])

    elif len(sys.argv) < 4:
        e = EventsTreeReader(sys.argv[1], sys.argv[2])

    else:
        # Arguments: input filename, output filename (must use .pdf file extension), option
        e = EventsTreeReader(sys.argv[1], sys.argv[2], sys.argv[3])

