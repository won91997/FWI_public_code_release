# -*- coding: utf-8 -*-

####################################################
####             MAIN PARAMETERS                ####
####################################################

# Existing datasets: SEGSalt|SEGSimulation|CurveVelA|CurveVelB|CurveFaultA|CurveFaultB)
#dataset_name  = 'SEGSimulation'
dataset_name  = 'CurveVelB'
learning_rate = 0.001                               # Learning rate
classes = 1                                         # Number of output channels
display_step = 2                                    # Number of training sessions required to print a "loss"
iterSize = 5000
####################################################
####            DATASET PARAMETERS              ####
####################################################

if dataset_name  == 'SEGSimulation':
    data_dim = [400, 301]                           # Dimension of original one-shot seismic data
    model_dim = [201, 301]                          # Dimension of one velocity model
    inchannels = 29                                 # Number of input channels
    train_size = 1600
    test_size = 100
    epochs = 120

    train_batch_size = 10
    test_batch_size = 2
    n_classes = 1

elif dataset_name  == 'SEGSalt':
    data_dim = [400, 301]
    model_dim = [201, 301]
    inchannels = 29
    train_size = 130
    test_size = 10

    epochs = 120

    train_batch_size = 10
    test_batch_size = 2

elif dataset_name == 'FlatVelA':
    data_dim = [1000, 70]
    model_dim = [70, 70]
    inchannels = 5
    train_size = 24000
    test_size = 6000

    epochs = 150

    train_batch_size = 64
    test_batch_size = 10

elif dataset_name == 'CurveVelA':
    data_dim = [1000, 70]
    model_dim = [70, 70]
    inchannels = 5
    train_size = 24000
    test_size = 6000

    epochs = 150

    train_batch_size = 64
    test_batch_size = 10

elif dataset_name == 'CurveVelB':
    data_dim = [1000, 70]
    model_dim = [70, 70]
    inchannels = 5
    train_size = 24000
    test_size = 6000

    epochs = 150
    train_batch_size = 64
    test_batch_size = 10

elif dataset_name == 'FlatFaultA':
    data_dim = [1000, 70]
    model_dim = [70, 70]
    inchannels = 5
    train_size = 48000
    test_size = 6000

    epochs = 150

    train_batch_size = 64
    test_batch_size = 10

elif dataset_name == 'CurveFaultA':
    data_dim = [1000, 70]
    model_dim = [70, 70]
    inchannels = 5
    train_size = 48000
    test_size = 6000

    epochs = 150

    train_batch_size = 64
    test_batch_size = 10

elif dataset_name == 'CurveFaultB':
    data_dim = [1000, 70]
    model_dim = [70, 70]
    inchannels = 5
    train_size = 48000
    test_size = 5000

    epochs = 150
    train_batch_size = 64
    test_batch_size = 10

else:
    print('The selected dataset is invalid')
    exit(0)

