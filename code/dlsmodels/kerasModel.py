"""
    Keras hyper-parameter tunning
    
    A way to create dinamically a Keras Model. It can be used with 
    params for hyper-parameter optimisation, or to generate dinamically 
    one combination
    
    @todo Rename
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

# Import Libs
from . import utils
import tensorflow_addons as tfa
import tensorflow
import sys

from tensorflow.keras.metrics import RootMeanSquaredError as RMSE
from tqdm import tqdm
from keras import backend as K


def rmse (y_pred, y_true):
    return K.sqrt (K.mean (K.square (y_pred - y_true)))


def create (parameters, dataset = None, inputs_size = None):
    
    """
    Create a parametrizable Keras Model
    
    @param parameters Dict
    @param dataset Dataset
    
    @return KerasModel
    """
    
    # @var output_bias Constant
    output_bias = tensorflow.keras.initializers.Constant (parameters['output_bias']) if 'output_bias' in parameters else None
    
    
    # @var dataset_options Dict
    dataset_options = dataset.get_task_options ()
        
    
    # @var task_type String
    task_type = dataset.get_task_type ()
    
    
    # @var is_imbalanced Boolean Determine if the dataset is imbalaced
    is_imbalanced = dataset.is_imbalanced ()
    
    
    # @var metrics List
    metrics = []
    
    
    # Determine configuration according to the task type
    if task_type == 'classification':
        
        # @var number_of_classes int
        number_of_classes = dataset.get_num_labels ()

    
        # Determine if the task is binary or multi-class
        is_binary = number_of_classes <= 2
        
        
        # @var last_activation_layer String Get the last activation layer based on the number of classes
        # @link https://developers.google.com/machine-learning/crash-course/multi-class-neural-networks/softmax
        last_activation_layer = 'sigmoid' if is_binary else 'softmax'
        
        
        # @var loss_function String Depends if multiclass or binary
        # @link https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
        loss_function = 'binary_crossentropy' if is_binary else 'categorical_crossentropy'
        
        
        # @var number_of_neurons_in_the_last_layer int
        number_of_classes_in_the_last_layer = 1 if is_binary else number_of_classes
        
        
        # Attach accuracy
        metrics = [tensorflow.keras.metrics.BinaryAccuracy (name = "accuracy") if is_binary else tensorflow.keras.metrics.CategoricalAccuracy (name = 'accuracy')]
        
        
        # @var link https://github.com/tensorflow/tensorflow/issues/45615
        # @var link https://www.biostat.wisc.edu/~page/rocpr.pdf
        
        if is_imbalanced or 'f1_score_scheme' in dataset_options:
        
            # @var f1_score_scheme String
            f1_score_scheme = dataset_options['f1_score_scheme'] \
                if 'f1_score_scheme' in dataset_options else 'micro'

            
            # Include new metrics for imbalanced datasets
            # https://github.com/tensorflow/addons/issues/746
            metrics += [
                tensorflow.keras.metrics.AUC (name = 'prc', curve = 'PR'),
                tfa.metrics.F1Score (
                    average = f1_score_scheme, 
                    name = 'f1_score', 
                    threshold = .5 if is_binary else None,
                    num_classes = number_of_classes if not is_binary else 1
                )
            ]
    
    # Regression problems
    elif task_type == 'regression':
        
        # Get the last activation layer
        # @link https://towardsdatascience.com/deep-learning-which-loss-and-activation-functions-should-i-use-ac02f1c56aa8
        last_activation_layer = 'linear'
        
        
        # Define loss function as MSE
        # loss_function = tensorflow.keras.losses.MeanSquaredError (name = 'mse')
        loss_function = rmse
    
    
        # @var metrics Select the metric according to the problem
        metrics = [
            tensorflow.keras.metrics.RootMeanSquaredError (name = 'rmse')
        ]
        
        
        # @var number_of_neurons_in_the_last_layer int 
        number_of_classes_in_the_last_layer = 1
        
        
    # Multi label problems
    elif task_type == 'multi_label':
    
        # @var number_of_classes int
        number_of_classes = dataset.get_num_labels ()

        
        # @var last_activation_layer String 
        # @link https://medium.com/deep-learning-with-keras/how-to-solve-multi-label-classification-problems-in-deep-learning-with-tensorflow-keras-7fb933243595
        last_activation_layer = 'sigmoid'
        
        
        # @var loss_function String Depends if multiclass or binary
        # @link https://stackoverflow.com/questions/55929401/how-to-specify-model-compile-for-binary-crossentropy-activation-sigmoid-and-act
        loss_function = tensorflow.keras.losses.BinaryCrossentropy (from_logits = False)
        
        
        # @var number_of_neurons_in_the_last_layer int
        number_of_classes_in_the_last_layer = number_of_classes
        
        
        # @var f1_score_scheme String
        f1_score_scheme = dataset_options['f1_score_scheme'] \
            if 'f1_score_scheme' in dataset_options else 'micro'
            
        
        # Attach accuracy
        # We need to use keras.metrics.BinaryAccuracy() for measuring the 
        # accuracy since it calculates how often predictions match binary labels.
        metrics = [
            tensorflow.keras.metrics.BinaryAccuracy (name = "accuracy"),
            tfa.metrics.F1Score (
                average = f1_score_scheme, 
                name = 'f1_score', 
                num_classes = number_of_classes
            )            
        ]

        
    
    # @var neurons_per_layer List Contains a list of the neurons per layer according to 
    #                             different shapes
    neurons_per_layer = utils.get_neurons_per_layer (parameters['shape'], parameters['num_layers'], parameters['first_neuron'])
    
    
    # @var layers_input Dict
    layers_input = {}
    
    
    # Define the input layers
    if 'we' in parameters['features']:
    
        # Get the full tokenizer size
        vocab_size = len (parameters['tokenizer'].word_index) + 1
        
        
        # @var language String
        language = dataset.get_dataset_language ()
        
        
        # @var custom_embeddings Dict
        if language == 'es':
            custom_embeddings = {
                'fasttext': utils.get_embedding_matrix (
                    key = 'fasttext', 
                    tokenizer = parameters['tokenizer'], 
                    dataset = dataset, 
                    lang = language
                ),
                
                'glove': utils.get_embedding_matrix (
                    key = 'glove', tokenizer = parameters['tokenizer'], 
                    dataset = dataset, 
                    lang = language
                ),
                
                'word2vec': utils.get_embedding_matrix (
                    key = 'word2vec', 
                    tokenizer = parameters['tokenizer'], 
                    dataset = dataset, lang = language
                )
            }
        else:
            custom_embeddings = {
                'fasttext': utils.get_embedding_matrix (
                    key = 'fasttext', 
                    tokenizer = parameters['tokenizer'], 
                    dataset = dataset, 
                    lang = language
                )
            }
        
        
        # Define main embedding layer
        layers_input['we'] = tensorflow.keras.layers.Input (shape = (parameters['maxlen'],), name = 'input_we')
        
        
        # @var embeddings_source String
        embeddings_source = parameters['pretrained_embeddings']
        
        
        # @var embedding_dim int Get the embedding dimension
        # @note We use 300 if no pretrained dimension was supplied because is the 
        #                  same number as the majority of embeddings we used. 
        # @todo Allow to parametrise this number
        embedding_dim = custom_embeddings[embeddings_source].shape[1] if embeddings_source != 'none' else 300 
        
        
        # @var weights multidimensional List Get the weights from the pretrained word embeddings used in this task
        #                                    If not set, use the default value
        weights = [custom_embeddings[parameters['pretrained_embeddings']]] if parameters['pretrained_embeddings'] != 'none' else None
        
        
        # Input for word embeddings require a embedding layer with the weights
        # In Keras, each layer has a parameter called “trainable”. For freezing the weights 
        # of a particular layer, we should set this parameter to False, indicating that this 
        # layer should not be trained. 
        # @link https://www.learnopencv.com/keras-tutorial-fine-tuning-using-pre-trained-models
        layer_we = tensorflow.keras.layers.Embedding (
            input_dim = vocab_size, 
            output_dim = embedding_dim, 
            weights = weights, 
            input_length = parameters['maxlen'],
            trainable = True
        )(layers_input['we'])

    
        # Generate word embedding architecture
        # Some notes about the pooling layers
        
        # GlobalMaxPool1D
        # @link https://stats.stackexchange.com/questions/257321/what-is-global-max-pooling-layer-and-what-is-its-advantage-over-maxpooling-layer
        
        # SpatialDropout1D
        # @link https://stackoverflow.com/questions/50393666/how-to-understand-spatialdropout1d-and-when-to-use-it
        
        # @todo. Param recurrent dropout different from dropout
        
        # Multilayer perceptron. This layer will be connected to the rest of the MLP
        if (parameters['architecture'] == 'dense'):
            layer_we = tensorflow.keras.layers.GlobalMaxPool1D ()(layer_we)
        
        # Convolutional neuronal network
        if (parameters['architecture'] == 'cnn'):
            layer_we = tensorflow.keras.layers.SpatialDropout1D (parameters['dropout'])(layer_we)
            layer_we = tensorflow.keras.layers.Conv1D (
                filters = neurons_per_layer[0], 
                kernel_size = parameters['kernel_size'], 
                activation = parameters['activation']
            )(layer_we)
            layer_we = tensorflow.keras.layers.GlobalMaxPool1D ()(layer_we)
            

        # LSTM
        if (parameters['architecture'] == 'lstm'):
            layer_we = tensorflow.keras.layers.SpatialDropout1D (parameters['dropout'])(layer_we)
            layer_we = tensorflow.keras.layers.LSTM (neurons_per_layer[0], 
                dropout = parameters['dropout'], 
                recurrent_dropout = parameters['dropout'],
                return_sequences = False
            )(layer_we)

        # GRU
        if (parameters['architecture'] == 'gru'):
            layer_we = tensorflow.keras.layers.SpatialDropout1D (parameters['dropout'])(layer_we)
            layer_we = tensorflow.keras.layers.GRU (neurons_per_layer[0], 
                dropout = parameters['dropout'], 
                return_sequences = False
            )(layer_we)
        
        
        # BiLSTM
        if (parameters['architecture'] == 'bilstm'):
            layer_we = tensorflow.keras.layers.SpatialDropout1D (parameters['dropout'])(layer_we)
            for i in range (parameters['num_layers']):
                layer_we = tensorflow.keras.layers.Bidirectional (tensorflow.keras.layers.LSTM (neurons_per_layer[0], 
                    dropout = parameters['dropout'], 
                    recurrent_dropout = parameters['dropout'],
                    return_sequences = i != parameters['num_layers'] - 1
                ))(layer_we)

        # BiGRU
        if (parameters['architecture'] == 'bigru'):
            layer_we = tensorflow.keras.layers.SpatialDropout1D (parameters['dropout'])(layer_we)
            for i in range (parameters['num_layers']):
                layer_we = tensorflow.keras.layers.Bidirectional (tensorflow.keras.layers.GRU (neurons_per_layer[0], 
                    dropout = parameters['dropout'], 
                    recurrent_dropout = parameters['dropout'],
                    return_sequences = i != parameters['num_layers'] - 1
                ))(layer_we)
    
    
    # Create the input layers for the rest of the feature sets
    for feature in ['lf', 'se', 'be', 'bf', 'ne', 'cf', 'pr', 'ng', 'it', 'bi', 'rf']:
        layers_input[feature] = tensorflow.keras.layers.Input (shape = (inputs_size[feature],), name = 'input_' + feature) if feature in parameters['features'] else None
    
    
    # @var inner_layers Dict
    inner_layers = {}
    
    
    # Create the hidden layers
    for key, layer in layers_input.items ():
        
        # Prevent None layers
        if layer == None:
            continue
            
        
        # Attach to the inner layers to keep track on it. 
        # Here we set the first hidden layer that in it related 
        # to their inputs layers. However, in case of 
        # word embeddings, get get the input layer as the
        # last embedding layer
        inner_layers[key] = layer_we if key == 'we' else layer
        
        
        # Next, we are going to add a MLP to every input
        for i in range (parameters['num_layers']):

            # @var activation_function String
            # We use i to only apply linear functions to odd layers
            activation_function = parameters['activation'] if i % 2 == 0 else 'linear'

        
            # @var layer_name String
            layer_name = "dense_" + key + "_" + str (i)
            
            
            # @var activation_name String
            activation_name = activation_function + "_" + key + "_" + str (i)
            
            
            # @var activation_name
            dropup_name = "dropout_" + key + "_" + str (i)
            
            
            # @var neurons int Depending of the architecture, we set different 
            #                  types of neurons
            # @todo. Parametrize this
            neurons = 10 if 'we' == key else neurons_per_layer[i]
        
            
            # Attach a dense layer
            inner_layers[key] = tensorflow.keras.layers.Dense (neurons, 
                use_bias = True,
                activity_regularizer = tensorflow.keras.regularizers.l2 (0.01) if i > 1 else None,
                kernel_initializer = 'glorot_uniform',
                name = layer_name
            )(inner_layers[key])
            
            
            # Attach the activation function as a separate layer
            inner_layers[key] = tensorflow.keras.layers.Activation (activation_function, name = activation_name)(inner_layers[key])
            
            
            # Attach dropout layer if needed
            if (parameters['dropout'] and i % 2 == 0):
                inner_layers[key] = tensorflow.keras.layers.Dropout (parameters['dropout'], name = dropup_name)(inner_layers[key])
    
    
    # @var layer_merged Layer This is necesary when we had architectures with multiple inputs
    # @todo. Maybe we need to use "-1" rather than "0"
    layer_merged = tensorflow.keras.layers.concatenate ([layer for key, layer in inner_layers.items ()]) \
        if len (inner_layers.items ()) > 1 else inner_layers[list (inner_layers.keys ())[0]]
    
    
    # @var Inputs List
    inputs = [layer for key, layer in layers_input.items () if layer != None]

    
    # @var Outputs Dense Configure out layer for final predictions
    outputs = tensorflow.keras.layers.Dense (number_of_classes_in_the_last_layer, 
        activation = last_activation_layer, 
        bias_initializer = output_bias,
        name = 'output_layer'
    )(layer_merged)
    
    
    # @var model Model Create model with their inputs, hidden, and outputs layers
    model = tensorflow.keras.models.Model (inputs = inputs, outputs = outputs)
    
    
    # @var Optimizer
    optimizer = parameters['optimizer'] (learning_rate = parameters['lr'], clipnorm = 1.0)
    

    # Compile model
    model.compile (optimizer = optimizer, loss = loss_function, metrics = metrics)
    
    
    # Return the model
    return model
