# from event_first import Event
# from event import Event
#from event_c import Event
# from event_v3 import Event
from event_v3_light_thres import Event
import sys
import numpy as np
from array import array
from ROOT import TFile, TTree

class Writer:

    def __init__(self, event, outfile='output.root'):
        self.event = event

        self.f_out = TFile(outfile, 'RECREATE')
        self.initOutputTree()

    def initOutputTree(self):
        self.T_out = TTree('Sim', 'Sim') # output tree, with name Sim

        self.Event_ID = array('i', [0]) #
        self.T_out.Branch('Event_ID', self.Event_ID, 'Event_ID/I')

        self.nu_pdg = array('i', [0]) #
        self.T_out.Branch('nu_pdg', self.nu_pdg, 'nu_pdg/I')

        self.nu_xs = array('f', [0])
        self.T_out.Branch('nu_xs', self.nu_xs, 'nu_xs/F')

        self.nu_proc = array('i', [0]) # genie process
        self.T_out.Branch('nu_proc', self.nu_proc, 'nu_proc/I')

        self.nu_nucl = array('i', [0]) # which nucleon
        self.T_out.Branch('nu_nucl', self.nu_nucl, 'nu_nucl/I')

        self.E_nu = array('f', [0]) # true neutrino energy
        self.T_out.Branch('E_nu', self.E_nu, 'E_nu/F')

        self.E_avail = array('f', [0]) # energy availabe from vertex interaction (excluding energy lost inside nuclei)
        self.T_out.Branch('E_avail', self.E_avail, 'E_avail/F')

        self.E_availList = np.zeros((8,), dtype=np.float32) # E avail for: lepton, proton, neutron, pi+-, pi0, gamma, alpha, others.
        self.T_out.Branch('E_availList', self.E_availList, 'E_availList[8]/F')

        self.E_depoTotal = array('f', [0]) # total energy deposit from all (charged) tracks
        self.T_out.Branch('E_depoTotal', self.E_depoTotal, 'E_depoTotal/F')

        self.E_depoList = np.zeros((8,), dtype=np.float32) # depo for: lepton, proton, neutron, pi+-, pi0, gamma, alpha, others.
        self.T_out.Branch('E_depoList', self.E_depoList, 'E_depoList[8]/F')

        self.E_depoTotal_track = array('f', [0]) # total energy deposit from long tracks
        self.T_out.Branch('E_depoTotal_track', self.E_depoTotal_track, 'E_depoTotal_track/F')

        ## FOR DEBUGGING PURPOSES, DELETE LATER ##
        # self.E_depoTotal_dots = array('f', [0]) # total energy deposit from dots
        # self.T_out.Branch('E_depoTotal_dots', self.E_depoTotal_dots, 'E_depoTotal_dots/F')
        ##################################################################################

        self.E_depoList_track = np.zeros((8,), dtype=np.float32) # depo for long tracks: lepton, proton, neutron, pi+-, pi0, gamma, alpha, others.
        self.T_out.Branch('E_depoList_track', self.E_depoList_track, 'E_depoList_track[8]/F')

        self.Q_depoTotal = array('f', [0]) # total charge deposit from all (charged) tracks
        self.T_out.Branch('Q_depoTotal', self.Q_depoTotal, 'Q_depoTotal/F')

        self.Q_depoList = np.zeros((8,), dtype=np.float32) # charge depo for: lepton, proton, neutron, pi+-, pi0, gamma, alpha, others.
        self.T_out.Branch('Q_depoList', self.Q_depoList, 'Q_depoList[8]/F')

        self.Q_depoTotal_th_75keV = array('f', [0]) # total charge deposit from all (charged) tracks with dQ cut of 75keV
        self.T_out.Branch('Q_depoTotal_th_75keV', self.Q_depoTotal_th_75keV, 'Q_depoTotal_th_75keV/F')

        self.Q_depoList_th_75keV = np.zeros((8,), dtype=np.float32) # charge depo with 75keV cut for: lepton, proton, neutron, pi+-, pi0, gamma, alpha, others.
        self.T_out.Branch('Q_depoList_th_75keV', self.Q_depoList_th_75keV, 'Q_depoList_th_75keV[8]/F')

        self.Q_depoTotal_th_500keV = array('f', [0]) # total charge deposit from all (charged) tracks with dQ cut of 500keV
        self.T_out.Branch('Q_depoTotal_th_500keV', self.Q_depoTotal_th_500keV, 'Q_depoTotal_th_500keV/F')

        self.Q_depoList_th_500keV = np.zeros((8,), dtype=np.float32) # charge depo with 500keV cut for: lepton, proton, neutron, pi+-, pi0, gamma, alpha, others.
        self.T_out.Branch('Q_depoList_th_500keV', self.Q_depoList_th_500keV, 'Q_depoList_th_500keV[8]/F')

        self.Q_depoTotal_MBox = array('f', [0]) # total charge deposit from all (charged) tracks Modified Box Model
        self.T_out.Branch('Q_depoTotal_MBox', self.Q_depoTotal_MBox, 'Q_depoTotal_MBox/F')

        self.Q_depoList_MBox = np.zeros((8,), dtype=np.float32) # charge depo for: lepton, proton, neutron, pi+-, pi0, gamma, alpha, others.
        self.T_out.Branch('Q_depoList_MBox', self.Q_depoList_MBox, 'Q_depoList_MBox[8]/F')

        self.Q_depoTotal_MBox_th_75keV = array('f', [0]) # total charge deposit from all (charged) tracks with dQ cut of 75keV
        self.T_out.Branch('Q_depoTotal_MBox_th_75keV', self.Q_depoTotal_MBox_th_75keV, 'Q_depoTotal_MBox_th_75keV/F')

        self.Q_depoList_MBox_th_75keV = np.zeros((8,), dtype=np.float32) # charge depo with 75keV cut for: lepton, proton, neutron, pi+-, pi0, gamma, alpha, others.
        self.T_out.Branch('Q_depoList_MBox_th_75keV', self.Q_depoList_MBox_th_75keV, 'Q_depoList_MBox_th_75keV[8]/F')

        self.Q_depoTotal_MBox_th_500keV = array('f', [0]) # total charge deposit from all (charged) tracks with dQ cut of 500keV
        self.T_out.Branch('Q_depoTotal_MBox_th_500keV', self.Q_depoTotal_MBox_th_500keV, 'Q_depoTotal_MBox_th_500keV/F')

        self.Q_depoList_MBox_th_500keV = np.zeros((8,), dtype=np.float32) # charge depo with 500keV cut for: lepton, proton, neutron, pi+-, pi0, gamma, alpha, others.
        self.T_out.Branch('Q_depoList_MBox_th_500keV', self.Q_depoList_MBox_th_500keV, 'Q_depoList_MBox_th_500keV[8]/F')

        self.Q_depoTotal_dots_th_75keV = array('f', [0]) # total charge deposit from all short dots/blips with dQ cut of 75keV
        self.T_out.Branch('Q_depoTotal_dots_th_75keV', self.Q_depoTotal_dots_th_75keV, 'Q_depoTotal_dots_th_75keV/F')

        self.Q_depoList_dots_th_75keV = np.zeros((8,), dtype=np.float32) # charge depo with 75keV cut for short dots/blips from: lepton, proton, neutron, pi+-, pi0, gamma, alpha, others.
        self.T_out.Branch('Q_depoList_dots_th_75keV', self.Q_depoList_dots_th_75keV, 'Q_depoList_dots_th_75keV[8]/F')

        self.L_depoTotal_avg_220PEpMeV = array('f', [0]) # total light deposit from all (charged) tracks with mean light yield 220PE/MeV
        self.T_out.Branch('L_depoTotal_avg_220PEpMeV', self.L_depoTotal_avg_220PEpMeV, 'L_depoTotal_avg_220PEpMeV/F')

        self.L_depoList_avg_220PEpMeV = np.zeros((8,), dtype=np.float32) # light depo with mean LY 220PE/MeV for: lepton, proton, neutron, pi+-, pi0, gamma, alpha, others.
        self.T_out.Branch('L_depoList_avg_220PEpMeV', self.L_depoList_avg_220PEpMeV, 'L_depoList_avg_220PEpMeV[8]/F')

        self.L_depoTotal_avg_180PEpMeV = array('f', [0]) # total light deposit from all (charged) tracks with mean light yield 180PE/MeV
        self.T_out.Branch('L_depoTotal_avg_180PEpMeV', self.L_depoTotal_avg_180PEpMeV, 'L_depoTotal_avg_180PEpMeV/F')

        self.L_depoList_avg_180PEpMeV = np.zeros((8,), dtype=np.float32) # light depo with mean LY 180PE/MeV for: lepton, proton, neutron, pi+-, pi0, gamma, alpha, others.
        self.T_out.Branch('L_depoList_avg_180PEpMeV', self.L_depoList_avg_180PEpMeV, 'L_depoList_avg_180PEpMeV[8]/F')

        self.L_depoTotal_avg_140PEpMeV = array('f', [0]) # total light deposit from all (charged) tracks with mean light yield 140PE/MeV
        self.T_out.Branch('L_depoTotal_avg_140PEpMeV', self.L_depoTotal_avg_140PEpMeV, 'L_depoTotal_avg_140PEpMeV/F')

        self.L_depoList_avg_140PEpMeV = np.zeros((8,), dtype=np.float32) # light depo with mean LY 140PE/MeV for: lepton, proton, neutron, pi+-, pi0, gamma, alpha, others.
        self.T_out.Branch('L_depoList_avg_140PEpMeV', self.L_depoList_avg_140PEpMeV, 'L_depoList_avg_140PEpMeV[8]/F')

        self.L_depoTotal_avg_100PEpMeV = array('f', [0]) # total light deposit from all (charged) tracks with mean light yield 100PE/MeV
        self.T_out.Branch('L_depoTotal_avg_100PEpMeV', self.L_depoTotal_avg_100PEpMeV, 'L_depoTotal_avg_100PEpMeV/F')

        self.L_depoList_avg_100PEpMeV = np.zeros((8,), dtype=np.float32) # light depo with mean LY 100PE/MeV for: lepton, proton, neutron, pi+-, pi0, gamma, alpha, others.
        self.T_out.Branch('L_depoList_avg_100PEpMeV', self.L_depoList_avg_100PEpMeV, 'L_depoList_avg_100PEpMeV[8]/F')

        self.L_depoTotal_avg_35PEpMeV = array('f', [0]) # total light deposit from all (charged) tracks with mean light yield 100PE/MeV
        self.T_out.Branch('L_depoTotal_avg_35PEpMeV', self.L_depoTotal_avg_35PEpMeV, 'L_depoTotal_avg_35PEpMeV/F')

        self.L_depoList_avg_35PEpMeV = np.zeros((8,), dtype=np.float32) # light depo with mean LY 35PE/MeV for: lepton, proton, neutron, pi+-, pi0, gamma, alpha, others.
        self.T_out.Branch('L_depoList_avg_35PEpMeV', self.L_depoList_avg_35PEpMeV, 'L_depoList_avg_35PEpMeV[8]/F')

        self.L_depoTotal_MBox_avg_220PEpMeV = array('f', [0]) # total light deposit from all (charged) tracks with mean light yield 220PE/MeV
        self.T_out.Branch('L_depoTotal_MBox_avg_220PEpMeV', self.L_depoTotal_MBox_avg_220PEpMeV, 'L_depoTotal_MBox_avg_220PEpMeV/F')

        self.L_depoList_MBox_avg_220PEpMeV = np.zeros((8,), dtype=np.float32) # light depo with mean LY 220PE/MeV for: lepton, proton, neutron, pi+-, pi0, gamma, alpha, others.
        self.T_out.Branch('L_depoList_MBox_avg_220PEpMeV', self.L_depoList_MBox_avg_220PEpMeV, 'L_depoList_MBox_avg_220PEpMeV[8]/F')

        self.L_depoTotal_MBox_avg_180PEpMeV = array('f', [0]) # total light deposit from all (charged) tracks with mean light yield 180PE/MeV
        self.T_out.Branch('L_depoTotal_MBox_avg_180PEpMeV', self.L_depoTotal_MBox_avg_180PEpMeV, 'L_depoTotal_MBox_avg_180PEpMeV/F')

        self.L_depoList_MBox_avg_180PEpMeV = np.zeros((8,), dtype=np.float32) # light depo with mean LY 180PE/MeV for: lepton, proton, neutron, pi+-, pi0, gamma, alpha, others.
        self.T_out.Branch('L_depoList_MBox_avg_180PEpMeV', self.L_depoList_MBox_avg_180PEpMeV, 'L_depoList_MBox_avg_180PEpMeV[8]/F')

        self.L_depoTotal_MBox_avg_140PEpMeV = array('f', [0]) # total light deposit from all (charged) tracks with mean light yield 140PE/MeV
        self.T_out.Branch('L_depoTotal_MBox_avg_140PEpMeV', self.L_depoTotal_MBox_avg_140PEpMeV, 'L_depoTotal_MBox_avg_140PEpMeV/F')

        self.L_depoList_MBox_avg_140PEpMeV = np.zeros((8,), dtype=np.float32) # light depo with mean LY 140PE/MeV for: lepton, proton, neutron, pi+-, pi0, gamma, alpha, others.
        self.T_out.Branch('L_depoList_MBox_avg_140PEpMeV', self.L_depoList_MBox_avg_140PEpMeV, 'L_depoList_MBox_avg_140PEpMeV[8]/F')

        self.L_depoTotal_MBox_avg_100PEpMeV = array('f', [0]) # total light deposit from all (charged) tracks with mean light yield 100PE/MeV
        self.T_out.Branch('L_depoTotal_MBox_avg_100PEpMeV', self.L_depoTotal_MBox_avg_100PEpMeV, 'L_depoTotal_MBox_avg_100PEpMeV/F')

        self.L_depoList_MBox_avg_100PEpMeV = np.zeros((8,), dtype=np.float32) # light depo with mean LY 100PE/MeV for: lepton, proton, neutron, pi+-, pi0, gamma, alpha, others.
        self.T_out.Branch('L_depoList_MBox_avg_100PEpMeV', self.L_depoList_MBox_avg_100PEpMeV, 'L_depoList_MBox_avg_100PEpMeV[8]/F')

        self.L_depoTotal_MBox_avg_35PEpMeV = array('f', [0]) # total light deposit from all (charged) tracks with mean light yield 100PE/MeV
        self.T_out.Branch('L_depoTotal_MBox_avg_35PEpMeV', self.L_depoTotal_MBox_avg_35PEpMeV, 'L_depoTotal_MBox_avg_35PEpMeV/F')

        self.L_depoList_MBox_avg_35PEpMeV = np.zeros((8,), dtype=np.float32) # light depo with mean LY 35PE/MeV for: lepton, proton, neutron, pi+-, pi0, gamma, alpha, others.
        self.T_out.Branch('L_depoList_MBox_avg_35PEpMeV', self.L_depoList_MBox_avg_35PEpMeV, 'L_depoList_MBox_avg_35PEpMeV[8]/F')

        self.N_parList = np.zeros((8,), dtype=np.int32) # number of particles for: lepton, proton, neutron, pi+-, pi0, gamma, alpha, others.
        self.T_out.Branch('N_parList', self.N_parList, 'N_parList[8]/I')

        ### TO TEST THE GST TREE FROM GENIE V3 ###
        self.E_avail_gst = array('d', [0])
        self.T_out.Branch('E_avail_gst', self.E_avail_gst, 'E_avail_gst/D')

        self.E_avail_pre_FSI_gst = array('d', [0])
        self.T_out.Branch('E_avail_pre_FSI_gst', self.E_avail_pre_FSI_gst, 'E_avail_pre_FSI_gst/D')

        self.N_parList_gst = np.zeros((8,), dtype=np.int32) # no. of (primary) particles for: lepton, p, n, pi+-, pi0, gamma, alpha, others post-FSI
        self.T_out.Branch('N_parList_gst', self.N_parList_gst, 'N_parList_gst[8]/I')

        # self.N_parList_pre_FSI_gst = np.zeros((8,), dtype=np.int32) # no. of particles for: lepton, p, n, pi+-, pi0, gamma, alpha, others pre-FSI
        self.N_parList_pre_FSI_gst = np.zeros((11,), dtype=np.int32) # no. of particles for: lepton, p, n, pi+-, pi0, gamma, alpha, others pre-FSI
        self.T_out.Branch('N_parList_pre_FSI_gst', self.N_parList_pre_FSI_gst, 'N_parList_pre_FSI_gst[11]/I')

        self.nu_proc_gst = array('i', [0]) # genie process
        self.T_out.Branch('nu_proc_gst', self.nu_proc_gst, 'nu_proc_gst/I')

        ### More testing ###
        self.E_availList_dots = np.zeros((8,), dtype=np.float32)
        self.T_out.Branch('E_availList_dots', self.E_availList_dots, 'E_availList_dots[8]/F')

        self.N_parList_extra = np.zeros((3,), dtype=np.int32)
        self.T_out.Branch('N_parList_extra', self.N_parList_extra, 'N_parList_extra[3]/I')

        self.N_pipmList_gst = np.zeros((2,), dtype=np.int32)
        self.T_out.Branch('N_pipmList_gst', self.N_pipmList_gst, 'N_pipmList_gst[2]/I')

        ### L THRESHOLD ###
        self.L_depoTotal_avg_180PEpMeV_th_75keV = array('f', [0]) # total light deposit from all (charged) tracks with mean light yield 180PE/MeV with a dL > 75 keV threshold applied to each hit
        self.T_out.Branch('L_depoTotal_avg_180PEpMeV_th_75keV', self.L_depoTotal_avg_180PEpMeV_th_75keV, 'L_depoTotal_avg_180PEpMeV_th_75keV/F')

        self.L_depoList_avg_180PEpMeV_th_75keV = np.zeros((8,), dtype=np.float32) # light depo with mean LY 180PE/MeV and 75keV dL cut for: lepton, proton, neutron, pi+-, pi0, gamma, alpha, others.
        self.T_out.Branch('L_depoList_avg_180PEpMeV_th_75keV', self.L_depoList_avg_180PEpMeV_th_75keV, 'L_depoList_avg_180PEpMeV_th_75keV[8]/F')

        self.L_depoTotal_avg_180PEpMeV_th_500keV = array('f', [0]) # total light deposit from all (charged) tracks with mean light yield 180PE/MeV with a dL > 500 keV threshold applied to each hit
        self.T_out.Branch('L_depoTotal_avg_180PEpMeV_th_500keV', self.L_depoTotal_avg_180PEpMeV_th_500keV, 'L_depoTotal_avg_180PEpMeV_th_500keV/F')

        self.L_depoList_avg_180PEpMeV_th_500keV = np.zeros((8,), dtype=np.float32) # light depo with mean LY 180PE/MeV and 500keV dL cut for: lepton, proton, neutron, pi+-, pi0, gamma, alpha, others.
        self.T_out.Branch('L_depoList_avg_180PEpMeV_th_500keV', self.L_depoList_avg_180PEpMeV_th_500keV, 'L_depoList_avg_180PEpMeV_th_500keV[8]/F')


    def Write(self):
        # self.stat = {
        # }
        self.f_out.cd()

        for i in range(self.event.nEntry):
        # for i in range(100):
        # for i in range(4):
            self.event.Jump(i)

            # proc = str(self.event.info['nu_proc']) + '-' + str(self.event.info['nu_nucl'])
            # v = self.stat.setdefault(proc, 0)
            # self.stat[proc] = v + 1
            self.Event_ID[0] = self.event.info['Event_ID']
            self.nu_pdg[0] = self.event.info['nu_pdg']
            self.nu_xs[0] = self.event.info['nu_xs']
            self.nu_proc[0] = self.event.info['nu_proc']
            self.nu_nucl[0] = self.event.info['nu_nucl']
            self.E_nu[0] = self.event.info['E_nu']

            self.E_avail[0] = self.event.info['E_avail']
            self.E_depoTotal[0] = self.event.info['E_depoTotal']
            self.E_depoTotal_track[0] = self.event.info['E_depoTotal_track']
            # self.E_depoTotal_dots[0] = self.event.info['E_depoTotal_dots'] ## DEBUGGING PURPOSES
            self.Q_depoTotal[0] = self.event.info['Q_depoTotal']
            self.Q_depoTotal_th_75keV[0] = self.event.info['Q_depoTotal_th_75keV']
            self.Q_depoTotal_th_500keV[0] = self.event.info['Q_depoTotal_th_500keV']
            self.Q_depoTotal_dots_th_75keV[0] = self.event.info['Q_depoTotal_dots_th_75keV']
            self.L_depoTotal_avg_220PEpMeV[0] = self.event.info['L_depoTotal_avg_220PEpMeV']
            self.L_depoTotal_avg_180PEpMeV[0] = self.event.info['L_depoTotal_avg_180PEpMeV']
            self.L_depoTotal_avg_140PEpMeV[0] = self.event.info['L_depoTotal_avg_140PEpMeV']
            self.L_depoTotal_avg_100PEpMeV[0] = self.event.info['L_depoTotal_avg_100PEpMeV']
            self.L_depoTotal_avg_35PEpMeV[0] = self.event.info['L_depoTotal_avg_35PEpMeV']

            self.Q_depoTotal_MBox[0] = self.event.info['Q_depoTotal_MBox']
            self.Q_depoTotal_MBox_th_75keV[0]  = self.event.info['Q_depoTotal_MBox_th_75keV']
            self.Q_depoTotal_MBox_th_500keV[0] = self.event.info['Q_depoTotal_MBox_th_500keV']
            self.L_depoTotal_MBox_avg_220PEpMeV[0] = self.event.info['L_depoTotal_MBox_avg_220PEpMeV']
            self.L_depoTotal_MBox_avg_180PEpMeV[0] = self.event.info['L_depoTotal_MBox_avg_180PEpMeV']
            self.L_depoTotal_MBox_avg_140PEpMeV[0] = self.event.info['L_depoTotal_MBox_avg_140PEpMeV']
            self.L_depoTotal_MBox_avg_100PEpMeV[0] = self.event.info['L_depoTotal_MBox_avg_100PEpMeV']
            self.L_depoTotal_MBox_avg_35PEpMeV[0]  = self.event.info['L_depoTotal_MBox_avg_35PEpMeV']

            self.E_availList[:] = self.event.info['E_availList']
            self.E_depoList[:] = self.event.info['E_depoList']
            self.E_depoList_track[:] = self.event.info['E_depoList_track']
            self.Q_depoList[:] = self.event.info['Q_depoList']
            self.Q_depoList_th_75keV[:] = self.event.info['Q_depoList_th_75keV']
            self.Q_depoList_th_500keV[:] = self.event.info['Q_depoList_th_500keV']
            self.Q_depoList_dots_th_75keV[:] = self.event.info['Q_depoList_dots_th_75keV']
            self.L_depoList_avg_220PEpMeV[:] = self.event.info['L_depoList_avg_220PEpMeV']
            self.L_depoList_avg_180PEpMeV[:] = self.event.info['L_depoList_avg_180PEpMeV']
            self.L_depoList_avg_140PEpMeV[:] = self.event.info['L_depoList_avg_140PEpMeV']
            self.L_depoList_avg_100PEpMeV[:] = self.event.info['L_depoList_avg_100PEpMeV']
            self.L_depoList_avg_35PEpMeV[:] = self.event.info['L_depoList_avg_35PEpMeV']

            self.Q_depoList_MBox[:] = self.event.info['Q_depoList_MBox']
            self.Q_depoList_MBox_th_75keV[:] = self.event.info['Q_depoList_MBox_th_75keV']
            self.Q_depoList_MBox_th_500keV[:] = self.event.info['Q_depoList_MBox_th_500keV']
            self.L_depoList_MBox_avg_220PEpMeV[:] = self.event.info['L_depoList_MBox_avg_220PEpMeV']
            self.L_depoList_MBox_avg_180PEpMeV[:] = self.event.info['L_depoList_MBox_avg_180PEpMeV']
            self.L_depoList_MBox_avg_140PEpMeV[:] = self.event.info['L_depoList_MBox_avg_140PEpMeV']
            self.L_depoList_MBox_avg_100PEpMeV[:] = self.event.info['L_depoList_MBox_avg_100PEpMeV']
            self.L_depoList_MBox_avg_35PEpMeV[:] = self.event.info['L_depoList_MBox_avg_35PEpMeV']

            self.N_parList[:] = self.event.info['N_parList']

            self.N_parList[2] = self.event.info['N_parList'][2]

            ### TO TEST THE GST TREE FROM GENIE V3 ###
            if 'E_avail_gst' in self.event.info.keys():
                self.E_avail_gst[0] = self.event.info['E_avail_gst']
                self.E_avail_pre_FSI_gst[0] = self.event.info['E_avail_pre_FSI_gst']
                self.N_parList_gst[:] = self.event.info['N_parList_gst']
                self.N_parList_pre_FSI_gst[:] = self.event.info['N_parList_pre_FSI_gst']
                self.nu_proc_gst[0] = self.event.info['nu_proc_gst']

            ### More testing ###
            if 'E_availList_dots' in self.event.info.keys():
                self.E_availList_dots[:] = self.event.info['E_availList_dots']
                self.N_parList_extra[:]  = self.event.info['N_parList_extra']
            if 'N_pi+-List_gst' in self.event.info.keys():
                self.N_pipmList_gst[:]  = self.event.info['N_pi+-List_gst']

            ### IF L WITH dL CUT INFORMATION IS AVAILABLE ###
            if 'L_depoTotal_avg_180PEpMeV_th_75keV' in self.event.info.keys():
                self.L_depoTotal_avg_180PEpMeV_th_75keV[0] = self.event.info['L_depoTotal_avg_180PEpMeV_th_75keV']
                self.L_depoList_avg_180PEpMeV_th_75keV[:] = self.event.info['L_depoList_avg_180PEpMeV_th_75keV']

                self.L_depoTotal_avg_180PEpMeV_th_500keV[0] = self.event.info['L_depoTotal_avg_180PEpMeV_th_500keV']
                self.L_depoList_avg_180PEpMeV_th_500keV[:] = self.event.info['L_depoList_avg_180PEpMeV_th_500keV']

            self.T_out.Fill()

        self.T_out.Write()
        # print(self.stat)

if __name__ == "__main__":
    #if (len(sys.argv)>3):
    #    outfile = sys.argv[3]
    #else:
    #    outfile = 'output.root'
    #event = Event(sys.argv[1], sys.argv[2])
    # For v3 testing with GST file
    if (len(sys.argv)>4):
        outfile = sys.argv[4]
    else:
        outfile = 'output.root'
    # Now, sys.argv[1] is the input .root file, sys.argv[2] is the input .gst file, and sys.argv[3] is the evgen.
    event = Event(sys.argv[1], sys.argv[2], sys.argv[3])
    w = Writer(event, outfile)
    w.Write()
