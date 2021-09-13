import numpy as np
import math
import ROOT

particleType_lep = 0
particleType_met = 1
particleType_jet = 2


###################################################################################################################
###   F u n c t i o n s                                                                                         ###
###################################################################################################################

def addToNodes(nodes_i, AVG, pt, eta, phi, particleType, bTag = 0., charge = 0.):
  
  p = pt * np.cosh(eta)
  
  nodes_i.append([pt, eta, phi, p, 1.*(particleType == particleType_lep), 1.*(particleType == particleType_met), 1.*(particleType == particleType_jet), bTag, charge])
  
  AVG['n_counter'] += 1
  AVG['n_sum']     += np.array([math.log10(pt/1e3), eta, phi, math.log10(p/1e3), 1.*(particleType == particleType_lep), 1.*(particleType == particleType_met), 1.*(particleType == particleType_jet), bTag, charge])
  AVG['n_sum_sq']  += np.array([math.log10(pt/1e3), eta, phi, math.log10(p/1e3), 1.*(particleType == particleType_lep), 1.*(particleType == particleType_met), 1.*(particleType == particleType_jet), bTag, charge])**2


def createEdges(nodes_i, edges_i, senders_i, receivers_i, AVG):

  for i in range(len(nodes_i)):
    for j in range(i):
      
      #print(i,j,len(nodes_i))
      
      pt_i, eta_i, phi_i = getFromNodes(nodes_i, i)                                                               # Defined above
      pt_j, eta_j, phi_j = getFromNodes(nodes_i, j)
      
      part_i = ROOT.TLorentzVector()
      part_j = ROOT.TLorentzVector()
      
      part_i.SetPtEtaPhiM(pt_i, eta_i, phi_i, 0.)
      part_j.SetPtEtaPhiM(pt_j, eta_j, phi_j, 0.)
      
      dEta = eta_i-eta_j
      dPhi = part_i.DeltaPhi(part_j)
      dR   = part_i.DeltaR(part_j)
      
      part_comb = part_i + part_j
      
      # Reduction of pT, p and M
      combPt = np.log10(part_comb.Pt()/1e3)
      combP  = np.log10(part_comb.P()/1e3)
      combM  = np.log10(part_comb.M()/1e3)
      
      # First direction
      edges_i.append([ dEta,  dPhi, dR, combPt, combP, combM])
      senders_i  .append(i)
      receivers_i.append(j)
      
      AVG['e_counter'] += 1
      AVG['e_sum']     += np.array([dEta,  dPhi, dR, combPt, combP, combM])
      AVG['e_sum_sq']  += np.array([ dEta,  dPhi, dR, combPt, combP, combM])**2
      
      # Second direction
      edges_i.append([-dEta, -dPhi, dR, combPt, combP, combM])
      senders_i  .append(j)
      receivers_i.append(i)
      
      AVG['e_counter'] += 1
      AVG['e_sum']     += np.array([-dEta, -dPhi, dR, combPt, combP, combM])
      AVG['e_sum_sq']  += np.array([-dEta, -dPhi, dR, combPt, combP, combM])**2


def getFromNodes(nodes_i, j):
  return nodes_i[j][0], nodes_i[j][1], nodes_i[j][2]
  

def normElements(vec, mean, rms):

  if isinstance(vec[0], int):
    for i in range(len(vec)):
      vec[i] = np.float32((vec[i]-mean[i])/rms[i])
  
  elif isinstance(vec[0], list):
    for x in range(len(vec)):
      for i in range(len(vec[0])):
        vec[x][i] = np.float32((vec[x][i]-mean[i])/rms[i])

  else:
    print("Error: Normalization Failed!")


    
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #
#   U n u s e d   F u n c t i o n s                                                                               #
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #

# def get_DataFrame_variableFilter(tree, _filter):
#   return tree.arrays(filter_name=_filter, library="pd")


# def get_NumPyArray_variableFilter(tree, _filter):
#   return tree.arrays(filter_name=_filter, library="np")


# def get_NumPyArray_variableNames(tree, _names):
#   return tree.arrays(filter_name=_names, library="np")


# def convert_DataFrame_to_NumPyArray(df):
#   return df.to_numpy()


# def add_binaryFlag(df, isSignal = False):
#   df['isSignal'] = [1 if isSignal else 0] * df.shape[0]