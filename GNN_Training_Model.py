# Copyright 2018 The GraphNets Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Model architectures for the demos in TensorFlow 2."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from graph_nets import modules
from graph_nets import utils_tf
from six.moves import range
import sonnet as snt
import tensorflow as tf


NUM_LAYERS_e  =   2  # Hard-code number of layers in the edge/node/global models.
NUM_LAYERS_n  =   2  # Hard-code number of layers in the edge/node/global models.
NUM_LAYERS_g  =   4  # Hard-code number of layers in the edge/node/global models.
LATENT_SIZE_e = 256  # Hard-code latent layer sizes for demos.
LATENT_SIZE_n = 256  # Hard-code latent layer sizes for demos.
LATENT_SIZE_g = 256  # Hard-code latent layer sizes for demos.


def make_mlp_model_n():
  """Instantiates a new MLP, followed by LayerNorm.

  The parameters of each new MLP are not shared with others generated by
  this function.

  Returns:
    A Sonnet module which contains the MLP and LayerNorm.
  """
  return snt.Sequential([
      snt.nets.MLP([LATENT_SIZE_e] * NUM_LAYERS_e, activation=tf.nn.leaky_relu, activate_final=True),
      #snt.LayerNorm(axis=-1, create_offset=True, create_scale=True)
  ])

def make_mlp_model_e():
  """Instantiates a new MLP, followed by LayerNorm.

  The parameters of each new MLP are not shared with others generated by
  this function.

  Returns:
    A Sonnet module which contains the MLP and LayerNorm.
  """
  return snt.Sequential([
      snt.nets.MLP([LATENT_SIZE_n] * NUM_LAYERS_n, activation=tf.nn.leaky_relu, activate_final=True),
      #snt.LayerNorm(axis=-1, create_offset=True, create_scale=True)
  ])

def make_mlp_model_g():
  """Instantiates a new MLP, followed by LayerNorm.

  The parameters of each new MLP are not shared with others generated by
  this function.

  Returns:
    A Sonnet module which contains the MLP and LayerNorm.
  """
  return snt.Sequential([
      snt.nets.MLP([512, 256, 128, 64, 32, 16], activation=tf.nn.leaky_relu, activate_final=True),
      #snt.LayerNorm(axis=-1, create_offset=True, create_scale=True)
  ])



class MLPGraphIndependent(snt.Module):
  """GraphIndependent with MLP edge, node, and global models."""

  def __init__(self, name="MLPGraphIndependent"):
    super(MLPGraphIndependent, self).__init__(name=name)
    self._network = modules.GraphIndependent(
        edge_model_fn=make_mlp_model_e,
        node_model_fn=make_mlp_model_n,
        global_model_fn=make_mlp_model_g)

  def __call__(self, inputs):
    return self._network(inputs)

# Options for reducer: https://www.tensorflow.org/api_docs/python/tf/math/unsorted_segment_mean

class MLPGraphNetwork(snt.Module):
  """GraphNetwork with MLP edge, node, and global models."""

  def __init__(self, name="MLPGraphNetwork"):
    super(MLPGraphNetwork, self).__init__(name=name)
    self._network = modules.GraphNetwork(make_mlp_model_e, make_mlp_model_n,
                                         make_mlp_model_g, reducer=tf.math.unsorted_segment_sqrt_n)

  def __call__(self, inputs):
    return self._network(inputs)



class MLPGraphNetwork_withMerge(snt.Module):                                                                      # class child_class(parent_class)
  
  """GraphNetwork with MLP edge, node, and global models."""                                                      

  def __init__(self, global_output_size = None, name = "MLPGraphNetwork"):
    
    super(MLPGraphNetwork_withMerge, self).__init__(name=name)                                                    # Initialization of the parent class
    
    self._network = modules.GraphNetwork(make_mlp_model_e,
                                         make_mlp_model_n,
                                         make_mlp_model_g,
                                         reducer = tf.math.unsorted_segment_sqrt_n)                               # Aggregator function
    
    self._global_fn = snt.nets.MLP([global_output_size],
                                   activation     = tf.nn.leaky_relu,
                                   activate_final = False)
    
    # self._global_fn = snt.Linear(global_output_size, name="global_output")


  def __call__(self, inputs):
    
    transformed = self._network(inputs)
    
    return transformed, self._global_fn(transformed.globals)



class EncodeProcessDecode(snt.Module):
  """Full encode-process-decode model.

  The model we explore includes three components:
  
  - An "Encoder" graph net, which independently encodes the edge, node, and global attributes (does not compute relations etc.).
  
  - A "Core" graph net, which performs N rounds of processing (message-passing) steps. The input to the Core is the concatenation of the Encoder's output
      and the previous output of the Core (labeled "Hidden(t)" below, where "t" is the processing step).
  
  - A "Decoder" graph net, which independently decodes the edge, node, and global attributes (does not compute relations etc.), on each message-passing step.

                      Hidden(t)   Hidden(t+1)
                         |            ^
            *---------*  |  *------*  |  *---------*
            |         |  |  |      |  |  |         |
  Input --->| Encoder |  *->| Core |--*->| Decoder |---> Output(t)
            |         |---->|      |     |         |
            *---------*     *------*     *---------*
  """

  def __init__(self, edge_output_size   = None,
                     node_output_size   = None,
                     global_output_size = None,
                     name               = "EncodeProcessDecode"):
    
    super(EncodeProcessDecode, self).__init__(name=name)
    
    self._encoder = MLPGraphIndependent()
    self._core    = MLPGraphNetwork()
    self._decoder = MLPGraphIndependent()
    
    # Transforms the outputs into the appropriate shapes.
    if edge_output_size is None:
      edge_fn = None
    else:
      edge_fn = lambda: snt.Linear(edge_output_size, name="edge_output")
    
    if node_output_size is None:
      node_fn = None
    else:
      node_fn = lambda: snt.Linear(node_output_size, name="node_output")
    
    if global_output_size is None:
      global_fn = None
    else:
      global_fn = lambda: snt.Linear(global_output_size, name="global_output")
    
    self._output_transform = modules.GraphIndependent(edge_fn, node_fn, global_fn)


  def __call__(self, input_op, num_processing_steps):
    
    latent  = self._encoder(input_op)
    latent0 = latent
    
    output_ops = []
    
    for _ in range(num_processing_steps):
      core_input = utils_tf.concat([latent0, latent], axis=1)
      latent     = self._core(core_input)
      decoded_op = self._decoder(latent)
      output_ops.append(self._output_transform(decoded_op))
    
    return output_ops