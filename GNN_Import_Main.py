import numpy as np                                                                                                # Used below
import math

import uproot4                                                                                                    # Used below: 85, 89
import ROOT                                                                                                       # Used below: 178-179, 258-260
import pickle                                                                                                     # Used below: 408, 412, 416, 423, 427, 431

from tqdm import tqdm

import GNN_Import_DataClass as DataClass
import GNN_Import_Functions as Function

baf = True

if baf:
    FilePath     = "/cephfs/user/s6vaanan/FNN/nominal_variables_v4_bootstrap/"
else:
    FilePath     = "/mnt/c/Users/vakho/Desktop/ATLAS/Data/nominal_variables_v4_bootstrap/"

particleType_lep = 0
particleType_met = 1
particleType_jet = 2

AVG = {"g_counter": 0,   'g_sum': [0., 0., 0., 0., 0., 0., 0., 0.],       'g_sum_sq': [0., 0., 0., 0., 0., 0., 0., 0.],
       'n_counter': 0,   'n_sum': [0., 0., 0., 0., 0., 0., 0., 0., 0.],   'n_sum_sq': [0., 0., 0., 0., 0., 0., 0., 0., 0.],
       'e_counter': 0,   'e_sum': [0., 0., 0., 0., 0., 0.],               'e_sum_sq': [0., 0., 0., 0., 0., 0.]}


# Normalisation constants from subset of MC sample
mean_g = [18.05783655,      7.46320195,       5.96103814,       5.32315627,       4.9889818,        5.04893305,       1.73611771,       5.68339035]
mean_n = [1.89814222e+00,   2.35495168e-03,  -7.24897904e-05,   2.08233266e+00,   2.07498404e-01,   9.36408703e-02,   6.98860726e-01,  -1.94851562e-02,   2.30206834e-03]
mean_e = [0.,               0.,               2.18537344,       2.04720357,       2.29350206,       2.15881393]
rms_g  = [3.71775359,       1.41353566,       0.14194099,       0.21174093,       0.32483645,       0.23322738,       0.86765459,       0.16422097]
rms_n  = [0.30090143,       1.07130244,       1.81316388,       0.33736591,       0.40551549,       0.29132844,       0.45875311,       0.68772055,       0.45551411]
rms_e  = [1.43222332,       1.89346064,       0.92768516,       0.33192029,       0.31742706,       0.30754365]


# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #
#   V a r i a b l e s                                                                                             #
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #

variables_global = ['jet_sum_mv2c10_Continuous', 'nJets', 'HT_all', 'leading_jet_pT', 'met_met', 'lep_0_pt', 'deltaR_bb_min', 'HT_jets_noleadjet']

variables_jets   = ['jet_pt',   'jet_eta',   'jet_phi',   'jet_mv2c10']                                      
variables_el     = ['el_pt',    'el_eta',    'el_phi',    'el_charge']                                                      
variables_muon   = ['mu_pt',    'mu_eta',    'mu_phi',    'mu_charge']                                                       
variables_met    = ['met_met',  'met_phi']                                                                     

variables = variables_global + variables_jets + variables_el + variables_muon + variables_met

print("\nVariables: ", variables)

graphsPerFile      = 1000000
maxGraphsPerSample = -1



# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #
#   I m p o r t   D a t a                                                                                         #
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #

List_of_Samples=DataClass.Init(FilePath, variables, Cuts=True)                                                    # Get the list of DataClass instances defined in gnn_Dataclass.py

print("\nImporting Sample:") 
DataDict = {}                                                                                                     # Create a dictionary for storing pandas dataframes 
for Sample in List_of_Samples:
    DataFrame = Sample.GetGNNInput()                                                                              # For each sample open the ROOT files and import the dataframes
    DataDict[Sample.Name] = DataFrame                                                                             # Save the dataframes in the dictionary with corresponding key
  
Total = 0
for DF in DataDict.values():
    Total += DF.shape[0]
print("\nTotal Number of Events: ".ljust(35, " ") + str(Total).rjust(6," ") + "\n")


###################################################################################################################
###   G e n e r a t e   G r a p h s                                                                             ###
###################################################################################################################

print("\nGenerating Graphs:\n") 

for sample in DataDict.keys():
  
    data_dict_list = []
    data_dict_targ = []
    out_ID = 0
    ID = 0
    
    if baf:
        output_file_input =  open('/cephfs/user/s6vaanan/GNN/Data/GNN_Processed_Inputs_2/graphs_input_'  + str(sample) + '.dat', 'wb') 
        output_file_target = open('/cephfs/user/s6vaanan/GNN/Data/GNN_Processed_Inputs_2/graphs_target_' + str(sample) + '.dat', 'wb') 
    else:
        output_file_input =  open('/mnt/c/Users/vakho/Desktop/4tops_GNN/Data/GNN_ProcessedInputs_Test/graphs_input_'  + str(sample) + '.dat', 'wb') 
        output_file_target = open('/mnt/c/Users/vakho/Desktop/4tops_GNN/Data/GNN_ProcessedInputs_Test/graphs_target_' + str(sample) + '.dat', 'wb') 



    for index, row in tqdm(DataDict[sample].iterrows(), total=DataDict[sample].shape[0], desc=sample.ljust(15, " "), dynamic_ncols=True):
  
        #####################################   C r e a t e   G l o b a l s   #####################################
    
        globals_i = [row['jet_sum_mv2c10_Continuous'], row['nJets'], np.log10(row['HT_all']), np.log10(row['leading_jet_pT']), np.log10(row['met_met']), np.log10(row['lep_0_pt']), row['deltaR_bb_min'], np.log10(row['HT_jets_noleadjet'])]

        
        AVG['g_counter'] += 1
        AVG['g_sum']     += np.array(globals_i)
        AVG['g_sum_sq']  += np.array(globals_i)**2

        globals_t = [1 if "tttt" in sample else 0, row['weights']]
  
        #######################################   C r e a t e   N o d e s   #######################################
        
        nodes_i = []
        
        # Add Jet Nodes ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #
        for i in range(len(row['jet_pt'])):                                                                       # Loop over Jet vector
            pt   = row['jet_pt']    [i]
            eta  = row['jet_eta']   [i]
            phi  = row['jet_phi']   [i]
            bTag = row['jet_mv2c10'][i]                                                                                 
            
            Function.addToNodes(nodes_i, AVG, pt, eta, phi, particleType_jet, bTag)
                
        # Add Electron Nodes  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #
        for i in range(len(row['el_pt'])):                                                                        # Loop over Electron vector
            pt     = row['el_pt']    [i]
            eta    = row['el_eta']   [i]
            phi    = row['el_phi']   [i]
            charge = row['el_charge'][i]
            bTag   = 0

            Function.addToNodes(nodes_i, AVG, pt, eta, phi, particleType_lep, bTag, charge)

        # Add Muon Nodes  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #
        for i in range(len(row['mu_pt'])):                                                                        # Loop over Muon vector
            pt     = row['mu_pt']    [i]
            eta    = row['mu_eta']   [i]
            phi    = row['mu_phi']   [i]
            charge = row['mu_charge'][i]
            bTag   = 0
            
            Function.addToNodes(nodes_i, AVG, pt, eta, phi, particleType_lep, bTag, charge)
        
        # Add MET Node  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #
        Function.addToNodes(nodes_i, AVG, row.met_met, 0., row.met_phi, particleType_met)

        
        #######################################   C r e a t e   E d g e s   #######################################
        edges_i, senders_i, receivers_i = [], [], []
        
        Function.createEdges(nodes_i, edges_i, senders_i, receivers_i, AVG)                                                       
        

        ##########################################   N o r m a l i z e   ##########################################     

        # Reduction of pT and p
        for i in range(len(nodes_i)):
            nodes_i[i][0] = math.log10(nodes_i[i][0]/1e3)
            nodes_i[i][3] = math.log10(nodes_i[i][3]/1e3)
    
        Function.normElements(globals_i, mean_g, rms_g)                                                           # Defined in GNN_Import_Functions.py
        Function.normElements(nodes_i,   mean_n, rms_n)                                                            
        Function.normElements(edges_i,   mean_e, rms_e)



        # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #
        #   S a v e   D a t a                                                                                     #
        # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #

        data_dict_i = {"globals":   globals_i,
                       "nodes":     nodes_i,
                       "edges":     edges_i,
                       "senders":   senders_i,
                       "receivers": receivers_i}
        
        data_dict_t = {"globals":   globals_t,
                       "nodes":     [[]],
                       "edges":     [[]],
                       "senders":   [],
                       "receivers": []}
    
        data_dict_list.append(data_dict_i)
        data_dict_targ.append(data_dict_t)
    
        ID += 1
    
        # Reduction of event number for now...
        if maxGraphsPerSample > 0 and ID > maxGraphsPerSample:
            break
        
        pickle.dump(data_dict_list, output_file_input)
        data_dict_list = []
    
        pickle.dump(data_dict_targ, output_file_target)
        data_dict_targ = []

        if ID > graphsPerFile:
            break

    output_file_input.close()
    output_file_target.close()
    
    # if len(data_dict_list) > graphsPerFile:
      
    #   with open('/mnt/c/Users/vakho/Desktop/4tops_GNN/Data/GNN_ProcessedInputs/graphs_input_'  + str(sample) + "_" + str(out_ID) + '.dat', 'wb') as output_file:
    #     pickle.dump(data_dict_list, output_file)
    #     data_dict_list = []
      
    #   with open('/mnt/c/Users/vakho/Desktop/4tops_GNN/Data/GNN_ProcessedInputs/graphs_target_' + str(sample) + "_" + str(out_ID) + '.dat', 'wb') as output_file:
    #     pickle.dump(data_dict_targ, output_file)
    #     data_dict_targ = []
      
    #   out_ID += 1


  # with open('/mnt/c/Users/vakho/Desktop/4tops_GNN/Data/GNN_ProcessedInputs/graphs_input_'  + str(sample)  + "_" + str(out_ID) + '.dat', 'wb') as output_file:
  #   pickle.dump(data_dict_list, output_file)
  #   data_dict_list = []

  # with open('/mnt/c/Users/vakho/Desktop/4tops_GNN/Data/GNN_ProcessedInputs/graphs_target_' + str(sample)  + "_" + str(out_ID) + '.dat', 'wb') as output_file:
  #   pickle.dump(data_dict_targ, output_file)
  #   data_dict_targ = []

  # out_ID += 1

print("\n")

print("AVG Globals:".ljust(15) + str(AVG['g_sum']/AVG['g_counter']))
print("AVG Nodes:".ljust(15)   + str(AVG['n_sum']/AVG['n_counter']))
print("AVG Edges:".ljust(15)   + str(AVG['e_sum']/AVG['e_counter']))

print("\n")

print("RMS Globals:".ljust(15) + str(np.sqrt(AVG['g_sum_sq']/AVG['g_counter'] - (AVG['g_sum']/AVG['g_counter'])**2)))
print("RMS Nodes:".ljust(15)   + str(np.sqrt(AVG['n_sum_sq']/AVG['n_counter'] - (AVG['n_sum']/AVG['n_counter'])**2)))
print("RMS Edges:".ljust(15)   + str(np.sqrt(AVG['e_sum_sq']/AVG['e_counter'] - (AVG['e_sum']/AVG['e_counter'])**2)))
    


