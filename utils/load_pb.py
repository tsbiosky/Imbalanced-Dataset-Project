# -*- coding: utf-8 -*-
# Copyright 2019 Inceptio Technology. All Rights Reserved.
# Author:
#   Jingyu Qian (jingyu.qian@inceptioglobal.ai)


import tensorflow as tf


def load_pb_model(model_path):
    """
    Load a trained and frozen pb model and return its graph.
    Args:
        model_path: Path to the frozen model (.pb file).

    Returns:
        A TensorFlow graph containing the model's structure and weights.
    """
    with tf.io.gfile.GFile(model_path, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
    return graph


def node_to_tensor(node, index=0):
    """
    Get tensor name from node name and index
    Args:
        node: Name of an operation in a a graph.
        index: index of the tensor among the outputs of the node. Defautls to 0.
            When some nodes return more than one tensors, index helps to
            identify which one should be returned.

    Returns:
        A string indicating name of the required tensor
    """
    return node + ':' + str(index)


def get_io_tensor_by_node_name(graph: tf.Graph, input_nodes, output_nodes,
                               input_indices=None, output_indices=None):
    """
    Get graph input and output tensors by their corresponding operation names
    Args:
        graph: A tf.Graph object.
        input_nodes: A list of the names of all the input nodes.
        output_nodes: A list of the names of all the output nodes.
        input_indices: A list of integers with the length same as input_nodes,
            specifying which tensor should be returned from the outputs of an
            operation. Defaults to None, which converts to a list of zeroes.
            e.g. If input_nodes = ['node_a', 'node_b']
                    input_indices = [0, 1]
                 Then ['node_a:0', 'node_b:1'] will be the tensors searched.
        output_indices: A list of integers with the length same as output_nodes.

    Returns:
        Two lists of tf.Tensor objects, corresponding to inputs and outputs.
    """
    input_tensors = []
    output_tensors = []
    if not input_indices:
        input_indices = [0] * len(input_nodes)
    if not output_indices:
        output_indices = [0] * len(output_nodes)

    assert len(input_nodes) == len(input_indices)
    assert len(output_nodes) == len(output_indices)

    for node, index in zip(input_nodes, input_indices):
        tensor_name = node_to_tensor(node, index)
        try:
            input_tensors.append(graph.get_tensor_by_name(tensor_name))
        except KeyError as e:
            print(e)
    for node, index in zip(output_nodes, output_indices):
        tensor_name = node_to_tensor(node, index)
        try:
            output_tensors.append(graph.get_tensor_by_name(tensor_name))
        except KeyError as e:
            print(e)
    return input_tensors, output_tensors
