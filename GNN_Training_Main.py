import numpy as np
import math
import time
import sys

import tensorflow as tf
import sonnet as snt                                                                                              # Library built on top of TensorFlow designed to provide simple, composable abstractions for machine learning research.
import pickle                                                                                                     # "Pickling" is the process whereby a Python object hierarchy is converted into a byte stream.
from graph_nets import utils_tf
from tqdm import trange

import GNN_Training_Functions as Func
import GNN_Training_Model as Model


# np.random.seed(10)
# tf.random.set_seed(1234)


baf = True

list_global_variables = ['jet_sum_mv2c10_Continuous', 'nJets', 'HT_all', 'leading_jet_pT', 'met_met', 'lep_0_pt', 'deltaR_bb_min', 'HT_jets_noleadjet']

# ['jet_sum_mv2c10_Continuous', 'nJets', 'HT_all',
#  'deltaR_ll_max', 'leading_jet_pT',    'met_met',  'lep_0_pt',      'leading_bjet_pT', 'deltaR_bb_min', 'jet_5_pt',      'deltaR_lb_max',
#  'deltaR_lj_min', 'HT_jets_noleadjet', 'lep_1_pt', 'deltaR_ll_sum', 'jet_1_pt',        'deltaR_ll_max', 'deltaR_lb_min', 'lep_0_phi']

print('\nGlobal variables used in the training: ' + str(list_global_variables))

if baf:
  filepath = '/cephfs/user/s6vaanan/GNN/Data/GNN_Processed_Inputs_1/'
  
  samples  = ['tttt_LO',          'tttt_NLO',           'ttW',           'ttWW',          'ttZ',
              'ttH',              'vjets',              'vv',            'singletop',     'others',
              'ttbar_Qmis',       'ttbar_CO',           'ttbar_HF',      'ttbar_light',   'ttbar_others']

else:
  filepath = 'C:/Users/vakho/Desktop/4tops_GNN/Data/GNN_Processed_Inputs_Test/'
  
  samples  = ['tttt_LO',          'tttt_NLO',           'ttW',           'ttH']


# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #
#   N N   P a r a m e t e r s                                                                                     #
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #

# Training Parameters ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #
n_training_epochs  = 35
batch_size_tr      = 256
batch_size_ge      = 1024

learning_rate_init = 0.001                                                                                       # float(sys.argv[1])

# learning_rate_init  = 0.01 
# learning_rate       = tf.keras.optimizers.schedules.ExponentialDecay(learning_rate_init, decay_steps=100000, decay_rate=0.96, staircase=False)
# optimizer           = snt.optimizers.Adam(learning_rate_init)

lossfunc           = tf.keras.losses.BinaryCrossentropy(from_logits=True)                                        # Computes the crossentropy loss between the labels and predictions.

model              = Model.MLPGraphNetwork_withMerge(global_output_size = 1)                                     # Defined in: gnn_model.py

print("\nBatch size for Training: ".ljust(36, " ") + str(batch_size_tr))
print("Batch size for Testing: ".ljust(35, " ")    + str(batch_size_ge))
print("Learning Rate: ".ljust(35, " ")             + str(learning_rate_init))


# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #
#   I m p o r t   t h e   D a t a                                                                                 #
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #

data_input_train, data_input_test, data_target_train, data_target_test = Func.getData(filepath,
                                                                                      samples,
                                                                                      frac_events = 0.8,
                                                                                      frac_test   = 0.5)


################################################ Prepare the Data #################################################

data_target_train = Func.remove_negative_weights(data_target_train)                                               # Scans through the training set of events and sets the negative weights to zero

 
# data_input_train, data_input_test = Func.choose_global_variables([data_input_train, data_input_test], list_global_variables)


###################################################################################################################
###   I n s t a n t i a t e   t h e   M o d e l                                                                 ###
###################################################################################################################
tb                = math.floor(len(data_input_train)/batch_size_tr)
boundaries        = [       10*tb,  20*tb,   30*tb   ]
values            = [0.001, 0.0001, 0.00001, 0.000001]
learning_rate     = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
optimizer         = tf.keras.optimizers.Adam(learning_rate)                                                

log_every_seconds = -1
logged_iterations = []
lossVals          = []
minLossVal        = 1e99

print("\n" + "Starting training!" + "\n")

for iteration in range(n_training_epochs):                                                                        # Iterate over epochs
  start_time    = time.time()
  if iteration == 0:
    print("-"*179+"\n"+ "|".rjust(14," ")+"Time |".rjust(15," ")+"Loss_train |".rjust(30," ")+"Loss_test |".rjust(30," ")+"AUC_train |".rjust(30," ")+"AUC_test |".rjust(30," ")+"Overtraining |".rjust(30," ")+"\n"+"-"*179)
  
  ################################################# Shuffle Data ##################################################    
  shuffle_ids = np.random.permutation(len(data_input_train))                                                      # Genereates a random array of numbers from 0 to (len()-1)
  data_input_train  = data_input_train [shuffle_ids]
  data_target_train = data_target_train[shuffle_ids]
  
  shuffle_ids = np.random.permutation(len(data_input_test))
  data_input_test   = data_input_test [shuffle_ids]
  data_target_test  = data_target_test[shuffle_ids]

  # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #
  #   T r a i n   t h e   M o d e l                                                                               #
  # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #
  loss_train_epoch = []
  AUC_train_epoch  = [] 

  for idx in trange(0, len(data_input_train)-batch_size_tr, batch_size_tr, desc="Training ", leave=False):                                       # Iterate over batches

    inputs_tr  = utils_tf.data_dicts_to_graphs_tuple(data_input_train [idx:idx+batch_size_tr])                    # Creates a graphs.GraphsTuple containing tensors from data dicts.
    targets_tr = utils_tf.data_dicts_to_graphs_tuple(data_target_train[idx:idx+batch_size_tr])

    outputs_tr_graphs, outputs_tr_classes, loss_train_batch = Func.update_step(model, optimizer, lossfunc, inputs_tr, targets_tr, batch_size_tr)
    
    AUC_train_batch = Func.compute_AUC(targets_tr, outputs_tr_classes)                                       

    loss_train_epoch.append(loss_train_batch)
    AUC_train_epoch.append(AUC_train_batch)

    
  # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #
  #   T e s t   t h e   M o d e l                                                                                 #
  # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #
  loss_test_epoch = []
  AUC_test_epoch  = []  

  for _idx in trange(0, len(data_input_test)-batch_size_ge, batch_size_ge, desc="Testing  ", leave=False):

    inputs_ge  = utils_tf.data_dicts_to_graphs_tuple(data_input_test [_idx : _idx + batch_size_ge])
    targets_ge = utils_tf.data_dicts_to_graphs_tuple(data_target_test[_idx : _idx + batch_size_ge])
      
    outputs_ge_graphs, outputs_ge_classes = model(inputs_ge)
    
    loss_test_batch = Func.create_loss(lossfunc, targets_ge, outputs_ge_classes, batch_size_ge)
    
    AUC_test_batch  = Func.compute_AUC(targets_ge, outputs_ge_classes)
    
    loss_test_epoch.append(loss_test_batch)
    AUC_test_epoch.append(AUC_test_batch)


  ############################################### S u m m a r i z e ###############################################
  
  loss_train = np.mean(loss_train_epoch)
  loss_test  = np.mean(loss_test_epoch)
  AUC_train  = np.mean(AUC_train_epoch)
  AUC_test   = np.mean(AUC_test_epoch)
 
  RMS_loss_train = np.sqrt(np.mean((loss_train_epoch - loss_train)**2))
  RMS_loss_test  = np.sqrt(np.mean((loss_test_epoch  - loss_test )**2))
  RMS_AUC_train  = np.sqrt(np.mean((AUC_train_epoch  - AUC_train )**2))
  RMS_AUC_test   = np.sqrt(np.mean((AUC_test_epoch   - AUC_test  )**2))
  
  overtraining       = 1 - AUC_test/AUC_train
  overtraining_error = np.sqrt((RMS_AUC_train/AUC_train)**2 + (RMS_AUC_test/AUC_test)**2) * (AUC_test/AUC_train)

  lossVals.append(loss_test)

 
  # if loss_test < minLossVal:
  #   minLossVal = loss_test
  #   if baf:
  #     with open('/cephfs/user/s6vaanan/GNN/Submission_Output/models_GraphNetworkBlock.dat', 'wb') as output_file:
  #       pickle.dump(model, output_file)
  #   else:
  #     with open('C:/Users/vakho/Desktop/4tops_GNN/models_GraphNetworkBlock.dat', 'wb') as output_file:
  #       pickle.dump(model, output_file)
  
  # if len(lossVals) > 8:
  #   isDecreasing = False
    
  #   for i in range(8):
  #     if lossVals[-8-1] > lossVals[-i-1]:
  #       isDecreasing = True
    
  #   if not isDecreasing:
  #     print("Check for decreasing loss -->", isDecreasing, lossVals[-8-1:])
  #     if baf:
  #       with open('/cephfs/user/s6vaanan/GNN/Submission_Output/models_GraphNetworkBlock.dat', 'rb') as output_file:
  #         model = pickle.load(input_file)
  #     else:
  #       with open('C:/Users/vakho/Desktop/4tops_GNN/models_GraphNetworkBlock.dat', 'rb') as output_file:
  #         model = pickle.load(input_file)

  #     lossVals = []
  #     lossVals.append(minLossVal)
  #     optimizer.learning_rate *= (1./2.)
  
  elapsed = time.strftime("%M:%S", time.gmtime(time.time() - start_time))
  logged_iterations.append(iteration)
  

  print("   Epoch " + str(iteration + 1).ljust(3," "), end=" |")
  print(str(elapsed).rjust(13," "), end="")
  print("{:0.5f} \u00B1 {:0.5f}".format( loss_train,     RMS_loss_train     ).rjust(30, " "), end="")
  print("{:0.5f} \u00B1 {:0.5f}".format( loss_test,      RMS_loss_test      ).rjust(30, " "), end="")
  print("{:0.5f} \u00B1 {:0.5f}".format( AUC_train,      RMS_AUC_train      ).rjust(30, " "), end="")
  print("{:0.5f} \u00B1 {:0.5f}".format( AUC_test,       RMS_AUC_test       ).rjust(30, " "), end="")
  print("{:0.5f} \u00B1 {:0.5f}".format( overtraining,   overtraining_error ).rjust(30, " "), end="")
  print(str(optimizer._decayed_lr('float32').numpy()                        ).rjust(15, " "))
  
  # print(optimizer._decayed_lr('float32').numpy())
  
  # print(optimizer.__getattribute__('learning_rate'))
  
  # print(optimizer.iterations)

  # print(optimizer.learning_rate)
  
  # if ((iteration+1)%5) == 0:
  #   optimizer.learning_rate *= (1./10.)
  #   print('Learning rate reduced: ' + str(optimizer.learning_rate)) 

