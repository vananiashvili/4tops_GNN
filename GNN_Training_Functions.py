import tensorflow as tf
import numpy as np
import pickle
import glob                                                                                                       # Finds all the pathnames matching a specified pattern according to the rules used by the Unix shell.
import math

import gnn_common


# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #
#   G e t   t h e   D a t a                                                                                       #
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #

def getData(path, samples, frac_events = 1, frac_test = 0.25):

  """ Import the Input Graphs from the corrseponding .dat files """     

  ################################################# Find the Data #################################################
  print("\nLoading data from: ", path, "\n")
  print(" "*3 +"-"*67 + "\n" +"   | " + "Sample".ljust(19, " ") + "|" + "Total |".rjust(15, " ") + "Training |".rjust(15, " ") + "Testing |".rjust(15, " ") + "\n" + " "*3 +"-"*67)
  data_input_train, data_input_test, data_target_train, data_target_test = np.array([]), np.array([]), np.array([]), np.array([])
  
  for sample in samples:                                                                                     
    filepath_i  = path + 'graphs_input_'  + sample + '.dat'                                             
    filepath_t  = path + 'graphs_target_' + sample + '.dat'
    shuffle_ids = []

    ################################################ Load the Data ################################################
    with open(filepath_i, 'rb') as input_file:                                                                    # 'rb' mode opens the file in binary format for reading
      data_input = loadpickledfile(input_file)
      data_input = data_input[:math.floor(len(data_input)*frac_events)]                                           # Only keep a part of the data, if RAM can't handle it in whole                                                                                                            
      
      shuffle_ids = np.random.permutation(len(data_input))
      data_input  = np.array(data_input)[shuffle_ids]
      
      data_train, data_test = traintestsplit(sample, data_input, frac_test)
      data_input_train      = np.concatenate(( data_input_train, data_train )) 
      data_input_test       = np.concatenate(( data_input_test,  data_test  ))                        
      
      print("   | " + str(sample).ljust(18, " ") + str(len(data_input)).rjust(15, " ") + str(len(data_train)).rjust(15, " ") + str(len(data_test)).rjust(15, " ") + " |")
  
    with open(filepath_t, 'rb') as input_file:                                                             
      data_target = loadpickledfile(input_file)
      data_target = data_target[:math.floor(len(data_target)*frac_events)]

      data_target = np.array(data_target)[shuffle_ids]
      
      data_train, data_test = traintestsplit(sample, data_target, frac_test)
      data_target_train     = np.concatenate(( data_target_train, data_train )) 
      data_target_test      = np.concatenate(( data_target_test,  data_test  )) 
      
      assert len(data_input) == len(data_target), "\nUnequal number of events in input and target files!\n"

  print(" "*3 + "-"*67 + "\n   | " + "Total".ljust(19, " ") + "|"+ str(len(data_input_train)+len(data_input_test)).rjust(13, " ") + " |" + str(len(data_input_train)).rjust(13, " ") + " |"+ str(len(data_input_test)).rjust(13, " ") + " |" + "\n" + " "*3 + "-"*67 + "\n")

  return data_input_train, data_input_test, data_target_train, data_target_test


def loadpickledfile(input_file):
  
  """ Load the data from a pickled file """
  
  data_list = []
  while True:
    try:
      data_list += pickle.load(input_file)                                                                        # data_list is a list of dictionaries
    except EOFError:
      break       
  
  return data_list


def traintestsplit(sample, data, frac_test):
  
  """ Split the Data for training and testing """
  
  data_train, data_test = [], []

  if   sample is 'tttt_LO':
    data_train =  data
  elif sample is 'tttt_NLO':
    data_test  =  data
  else:
    data_train =  data[math.floor(len(data)*frac_test):]                                                          # math.floor(len()*0.25) rounds down the resulting number
    data_test  =  data[:math.floor(len(data)*frac_test)]

  return data_train, data_test



# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #
#   P r e p a r e   t h e   D a t a                                                                               #
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #

def remove_negative_weights(data_target_train):
  
  """ Scans through the training set of events and sets the negative weights to zero """
  
  counter = 0
  
  for i in range(len(data_target_train)):
    if data_target_train[i]['globals'][-1] < 0:
      data_target_train[i]['globals'][-1] = 0.0
      counter += 1

  print('Number of Events with Negative Weight used in Training: ' + str(counter))

  return data_target_train


def choose_global_variables(list_of_data_arrays, list_global_variables):
  
  """ Picks out the desired variables from the list of 19 variables savd in the graph tuples """

  all_variables = ['jet_sum_mv2c10_Continuous', 'nJets', 'HT_all',
                    'deltaR_ll_max', 'leading_jet_pT',    'met_met',  'lep_0_pt',      'leading_bjet_pT', 'deltaR_bb_min', 'jet_5_pt',      'deltaR_lb_max',
                    'deltaR_lj_min', 'HT_jets_noleadjet', 'lep_1_pt', 'deltaR_ll_sum', 'jet_1_pt',        'deltaR_ll_max', 'deltaR_lb_min', 'lep_0_phi']
  
  # all_variables = ['jet_sum_mv2c10_Continuous', 'nJets', 'HT_all', 'leading_jet_pT', 'met_met', 'lep_0_pt', 'deltaR_bb_min', 'HT_jets_noleadjet']
  
  indices = sorted([all_variables.index(variable) for variable in list_global_variables])
   
  for data_array in list_of_data_arrays:
    for i in range(len(data_array)):
      data_array[i]['globals'] = [data_array[i]['globals'][x] for x in indices]

  return list_of_data_arrays



# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #
#                                                                                                                 #
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #
def update_step(model, optimizer, loss_function, inputs_tr, targets_tr, batch_size):
  
  with tf.GradientTape() as tape:                                                                                 # tf.GradientTape(): Record operations for automatic differentiation.
    outputs_graphs, outputs_classes = model(inputs_tr)                                                            # Defined above: model = gnn_model.MLPGraphNetwork_withMerge(global_output_size = len(gnn_common.stxsBins) + 1)
    loss_tr = create_loss(loss_function, targets_tr, outputs_classes, batch_size)                                 # Defined above

  gradients = tape.gradient(loss_tr, model.trainable_variables)                                                   

  # optimizer.apply(gradients, model.trainable_variables)                                                           
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  return outputs_graphs, outputs_classes, loss_tr


def create_loss(loss_function, target, outputs, batch_size):

  """ Returns graphs containing the inputs and targets for classification. """                                    # Refer to create_data_dicts_tf and create_linked_list_target for more details.
  """
      target:  'graphs.GraphsTuple' which contains the target as a graph.
      outputs: 'list' of 'graphs.GraphsTuple's which contains the model outputs for each processing step as graphs.

      Returns: 'list' of ops which are the loss for each processing step.
  """

  targets = target.globals[:,0][:,tf.newaxis]
  weights = target.globals[:,1][:,tf.newaxis]/tf.reduce_sum(target.globals[:,1][:,tf.newaxis], 0)

  loss = loss_function(targets,  outputs, sample_weight=weights)*batch_size 
      
  return loss


def compute_AUC(target, output):

  targets = target.globals[:,0][:,tf.newaxis]
  weights = target.globals[:,1][:,tf.newaxis]/tf.reduce_sum(target.globals[:,1][:,tf.newaxis], 0)
  outputs = tf.keras.activations.sigmoid(output)

  AUC = tf.keras.metrics.AUC(num_thresholds=200, curve='ROC', summation_method='interpolation')
  AUC.update_state(targets, outputs, sample_weight=weights)

  return AUC.result().numpy()






# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #
#   U n u s e d   f u n c t i o n s                                                                               #
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #

# def getData_old(path, samples, frac_events = 1, frac_test = 0.25):                                                # Used in gnn_STXS_GraphNetworkBlock.py
#   ################################################# Find the Data #################################################
#   print("Loading data from", path)
#   file_list, file_list_targ = [], []
  
#   for sample in samples:                                                                                     
#     files = glob.glob(path + 'graphs_input_'  + sample + '.dat')                                                # glob.glob(pathname, *) - Return a list of path names that match pathname.
#     files.sort()                                                                                                  # Sort the list alphabetically
#     file_list += files
#     files = glob.glob(path + 'graphs_target_' + sample + '.dat')
#     files.sort()
#     file_list_targ += files
  
#   file_list, file_list_targ = np.array(file_list), np.array(file_list_targ)

#   ################################################# Load the Data #################################################
#   file_names, data_input_train, data_input_test, data_target_train, data_target_test = [], [], [], [], []
  
#   # Loop over Inputs  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #
#   for datafile in file_list:   
#     file_names += [datafile[59:]]
#     with open(datafile, 'rb') as input_file:                                                                      # 'rb' mode opens the file in binary format for reading
#       data_input        = pickle.load(input_file)                                                                 # Load the data from a pickled file. data_input is a list of dictionaries  
#       data_input        = data_input[:math.floor(len(data_input)*frac_events)]                                    # Only keep a part of the data, if RAM can't handle it in whole                                                                                                            
#       data_input_train += data_input[math.floor(len(data_input)*frac_test):]                                      # Split the Data for training and testing
#       data_input_test  += data_input[:math.floor(len(data_input)*frac_test)]                                      # math.floor(len()*0.25) rounds down the resulting number
      
#   # Loop over Targets ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #
#   for datafile in file_list_targ:
#     file_names += [datafile[59:]]
#     with open(datafile, 'rb') as input_file:                                                             
#       data_target        = pickle.load(input_file)
#       data_target        = data_target[:math.floor(len(data_target)*frac_events)]
#       data_target_train += data_target[math.floor(len(data_target)*frac_test):]
#       data_target_test  += data_target[:math.floor(len(data_target)*frac_test)]

#   data_input_train, data_target_train = np.array(data_input_train), np.array(data_target_train)
#   data_input_test,  data_target_test  = np.array(data_input_test),  np.array(data_target_test)

#   return data_input_train, data_input_test, data_target_train, data_target_test





# def predictedClass_fromPt(inTensor):
  
#   classTensor = inTensor
  
#   classTensor = tf.where(classTensor >= gnn_common.stxsBins[0], classTensor, 0)
#   for i in range(1,len(gnn_common.stxsBins)):
#     classTensor = tf.where(tf.math.logical_or(classTensor >= gnn_common.stxsBins[i], classTensor < gnn_common.stxsBins[i-1]), classTensor, i)
#   classTensor = tf.where(classTensor >= gnn_common.stxsBins[len(gnn_common.stxsBins)-1], len(gnn_common.stxsBins), classTensor)
  
#   return classTensor



# def getSampleWeights_HiggsPt(data_target_train, data_target_test):
#   return getSampleWeights_HiggsPt_single(data_target_train), getSampleWeights_HiggsPt_single(data_target_test)



# def getSampleWeights_HiggsPt_single(inArray, normBins = gnn_common.higgsPtBins, binSize = gnn_common.higgsPtSteps):
  
#   outArray = np.zeros(len(inArray))
  
#   for i in range(outArray.shape[0]):
    
#     ID = (int) (inArray[i]['globals'][0]/binSize)
#     #print(inArray[i]['globals'][0],ID,gnn_common.higgsPtStat[ID])
#     if ID > normBins-1:
#       ID = normBins-1
      
#     outArray[i] = gnn_common.higgsPtStatSum/gnn_common.higgsPtStat[ID]
  
#   return outArray



# def compute_accuracy(target, output):
  
#   _stxsStat = gnn_common.stxsStat_2lSS                                                                            # stxsStat_2lSS = [93993., 159151., 130297., 60550., 23310., 6330.]
  
#   tdds = target.globals
#   odds = output
  
#   _tdds = tf.math.argmax(tdds, axis=1)
#   _odds = tf.math.argmax(odds, axis=1)
  
#   print("\n", _odds, "\n")
#   _tot = 0.
  
#   equal = 0
#   equalW = 0.

#   for i in range(_tdds.shape[0]):
#     if _tdds[i] == _odds[i]:
#       equal  += 1
#       equalW += 1./_stxsStat[_tdds[i]]
#     _tot += 1./_stxsStat[_tdds[i]]
  
#   return equal/_tdds.shape[0], equalW/_tot



# def compute_accuracy_fromPt(targetPt, outputPt):
#   tdds = targetPt.globals
#   odds = outputPt
  
#   _tdds = predictedClass_fromPt(tdds)
#   _odds = predictedClass_fromPt(odds)
  
#   _tot = 0.
  
#   equal  = 0
#   equalW = 0.
  
#   for i in range(_tdds.shape[0]):
#     if int(_tdds[i]) == int(_odds[i]):
#       equal  += 1
#       equalW += 1./gnn_common.stxsStat[int(_tdds[i])]
#     _tot += 1./gnn_common.stxsStat[int(_tdds[i])]
  
#   simpleAcc = equal/_tdds.shape[0]
#   weightedAcc = equalW/_tot
  
#   return simpleAcc, weightedAcc