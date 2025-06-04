import sys, os

import numpy as np
import matplotlib.pyplot as plt

from ROOT import TFile, TTree

class EDepHistPlotter:

    def __init__(self, filename, out_filename="energy_depo_hist.pdf"):
        self.filename = filename
        self.out_filename = out_filename

        # In case somehow the plots directory does not exist in PyEdep directory (from event.py)
        if not os.path.exists("./plots"):
            os.makedirs("./plots")
            print("./plots directory did not exist. It was created.")


    def plotHistogram(self):
        f_in = TFile.Open(self.filename)
        edepTree = f_in.Sim

        # For debugging purposes
        print("Number of entries in edep-sim tree from writer: ", edepTree.GetEntries())

        E_avail = np.array([])
        E_dep = np.array([])
        Q = np.array([])
        Q_thre = np.array([])
        L = np.array([])


        for entry in edepTree:
            E_avail = np.append(E_avail, entry.E_avail)
            E_dep = np.append(E_dep, entry.E_depoTotal)
            Q = np.append(Q, entry.Q_depoTotal)
            Q_thre = np.append(Q_thre, entry.Q_depoTotal_th_75keV)
            L = np.append(L, entry.L_depoTotal_avg_180PEpMeV) # Deposited energy in scintillation light assuming a light yield of 180 photoelectrons per MeV (assuming MIPs, 0.5 kV/cm E field, 0.8% APEX detector PCE, Birks Model)

        plt.figure()
        plt.hist(E_avail, bins=100, histtype="step", label="E_avail")
        plt.hist(E_dep, bins=100, histtype="step", label="E_dep")
        plt.hist(Q, bins=100, histtype="step", label="Q")
        plt.hist(Q_thre, bins=100, histtype="step", linestyle="--", label="Q_thre (dQ > 75keV)")
        plt.hist(L, bins=100, histtype="step", label="L")
        plt.xlabel("Energy [MeV]")
        plt.ylabel("Number of events")
        plt.legend()
        plt.savefig("./plots/" + self.out_filename)
        plt.clf()
        plt.close()

# ensures we are running from command line
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Must specify a .root file to make histogram from!")
        sys.exit()

    e = EDepHistPlotter(sys.argv[1], sys.argv[2]) # sys.argv[0] will be module name (energy_depo_hist), sys.argv[1] is first argument (input file), sys.argv[2] is second argument (output file)
    e.plotHistogram()


