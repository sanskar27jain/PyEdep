import sys, os

import numpy as np
import matplotlib.pyplot as plt

from ROOT import TFile, TTree

class EventsTreeReader:

    def __init__(self, filename, out_filename="energy_depo_hist.pdf"):
        self.filename = filename
        self.out_filename = out_filename

        # In case somehow the plots directory does not exist in PyEdep directory (from event.py)
        if not os.path.exists("./plots"):
            os.makedirs("./plots")
            print("./plots directory did not exist. It was created.")

        self.GetEnergies()

    def GetEnergies(self):
        f_in = TFile.Open(self.filename)
        edepTree = f_in.Sim

        # For debugging purposes
        print("Number of entries in edep-sim tree from writer: ", edepTree.GetEntries())

        self.event_energies = {}

        for entry in edepTree:
            E_nu = entry.E_nu
            if E_nu not in self.event_energies:
                self.event_energies[E_nu] = {}
                self.event_energies[E_nu]['E_avail'] = np.array([])
                self.event_energies[E_nu]['E_dep'] = np.array([])
                self.event_energies[E_nu]['Q'] = np.array([])
                self.event_energies[E_nu]['Q_thre'] = np.array([])
                self.event_energies[E_nu]['L'] = np.array([])

                self.event_energies[E_nu]['Q_e'] = np.array([])
                self.event_energies[E_nu]['Q_h'] = np.array([])

            self.event_energies[E_nu]['E_avail'] = np.append(self.event_energies[E_nu]['E_avail'], entry.E_avail)
            self.event_energies[E_nu]['E_dep'] = np.append(self.event_energies[E_nu]['E_dep'], entry.E_depoTotal)
            self.event_energies[E_nu]['Q'] = np.append(self.event_energies[E_nu]['Q'], entry.Q_depoTotal)
            self.event_energies[E_nu]['Q_thre'] = np.append(self.event_energies[E_nu]['Q_thre'], entry.Q_depoTotal_th_75keV)
            self.event_energies[E_nu]['L'] = np.append(self.event_energies[E_nu]['L'], entry.L_depoTotal_avg_180PEpMeV) # Deposited energy in scintillation light assuming a light yield of 180 photoelectrons per MeV (assuming MIPs, 0.5 kV/cm E field, 0.8% APEX detector PCE, Birks Model)

            # Assuming gamma (photons) are also in EM component
            self.event_energies[E_nu]['Q_e'] = np.append(self.event_energies[E_nu]['Q_e'], np.sum([entry.Q_depoList[i] for i in [0, 4, 5]]))
            # Assuming the 'other' particles and alpha particles are also in hadronic component
            self.event_energies[E_nu]['Q_h'] = np.append(self.event_energies[E_nu]['Q_h'], np.sum([entry.Q_depoList[i] for i in [1, 2, 3, 6, 7]]))

        for E_nu in self.event_energies.keys():
            self.event_energies[E_nu]['E_rec_L1'] = self.event_energies[E_nu]['L'] / 0.42
            self.event_energies[E_nu]['E_rec_Q1'] = self.event_energies[E_nu]['Q'] / 0.48
            self.event_energies[E_nu]['E_rec_Q2'] = (self.event_energies[E_nu]['Q_e'] / 0.55) + (self.event_energies[E_nu]['Q_h'] / 0.31)

            self.event_energies[E_nu]['E_rec_L1_res'] = EventsTreeReader.calculateRes(self.event_energies[E_nu]['E_rec_L1'])
            self.event_energies[E_nu]['E_rec_Q1_res'] = EventsTreeReader.calculateRes(self.event_energies[E_nu]['E_rec_Q1'])
            self.event_energies[E_nu]['E_rec_Q2_res'] = EventsTreeReader.calculateRes(self.event_energies[E_nu]['E_rec_Q2'])

            print(f"E_rec_L1 resolution: {self.event_energies[E_nu]['E_rec_L1_res']*100:.2f}%")
            print(f"E_rec_Q1 resolution: {self.event_energies[E_nu]['E_rec_Q1_res']*100:.2f}%")
            print(f"E_rec_Q2 resolution: {self.event_energies[E_nu]['E_rec_Q2_res']*100:.2f}%")

    def calculateRes(energyDist):
        return np.std(energyDist) / np.mean(energyDist)

    def plotE_depQLHistogram(self, E_nu):
        info = self.event_energies[E_nu]

        plt.figure()
        plt.hist(info['E_avail'], bins=100, histtype="step", label="E_avail")
        plt.hist(info['E_dep'], bins=100, histtype="step", label="E_dep")
        plt.hist(info['Q'], bins=100, histtype="step", label="Q")
        plt.hist(info['Q_thre'], bins=100, histtype="step", linestyle="--", label="Q_thre (dQ > 75keV)")
        plt.hist(info['L'], bins=100, histtype="step", label="L")
        plt.xlabel("Energy [MeV]")
        plt.ylabel("Number of events")
        plt.legend()
        plt.savefig("./plots/" + self.out_filename)
        plt.clf()
        plt.close()

# checks if we are running from command line
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Must specify a .root file to make histogram from!")
        sys.exit()

    e = EventsTreeReader(sys.argv[1], sys.argv[2]) # sys.argv[0] will be module name (energy_depo_hist), sys.argv[1] is first argument (input file), sys.argv[2] is second argument (output file)
    e.plotE_depQLHistogram(1000)


