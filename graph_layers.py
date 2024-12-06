# graph_layers.py

import keras
from keras import layers
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)

class GraphConvLayer(keras.layers.Layer):
    def __init__(self, output_dim, dropout_rate=0.2):
        super(GraphConvLayer, self).__init__()
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.dense = keras.layers.Dense(output_dim, activation='relu')
        self.dropout = keras.layers.Dropout(dropout_rate)
        self.batch_norm = keras.layers.BatchNormalization()

    def build(self, input_shape):
        super().build(input_shape)
    def create_ffn(self, output_dim, dropout_rate):
        ffn_layers = [
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            layers.Dense(output_dim, activation='relu')
        ]
        return keras.Sequential(ffn_layers)

    def call(self, inputs):
        node_features, edges, edge_weights = inputs
        
        # Ensure correct types
        edges = tf.cast(edges, tf.int32)
        node_indices, neighbor_indices = edges[0], edges[1]
        edge_weights = tf.cast(edge_weights, tf.float32)

        # Process features
        neighbor_features = tf.gather(node_features, neighbor_indices)
        weighted_features = neighbor_features * tf.expand_dims(edge_weights, -1)
        
        # Aggregate
        output = tf.math.unsorted_segment_mean(
            weighted_features,
            node_indices,
            tf.shape(node_features)[0]
        )
        
        # Apply transformations
        output = self.batch_norm(output)
        output = self.dropout(output)
        output = self.dense(output)
        
        return output