####################################################
####             MAIN PARAMETERS                ####
####################################################

# choose the dataset
SEGSimulateData = False # True
SEGSaltData = False
OpenFWI = True
DataSet = "CurveVelA/"  # CurveVelA  FlatFaultA  FlatVelA   CurveFaultA  SEGSaltData  SEGSimulation

# choose the network
NetworkName = 'TU_Net'  # DD_Net DD_Net70 FCNVMB InversionNet TU_Net TU_Net_SEG VelocityGAN

# whether to use and train the model
ReUse = False  # If False always re-train a network

if OpenFWI:
    DataDim = [1000, 70]  # Dimension of original one-shot seismic data
    ModelDim = [70, 70]  # Dimension of one velocity model
    InChannel = 5  # Source numbers
    OutChannel = 1  # Number of channels in the output velocity model

elif SEGSaltData or SEGSimulateData:
    DataDim = [400, 301]  # Dimension of original one-shot seismic data
    ModelDim = [201, 301]  # Dimension of one velocity model
    InChannel = 29  # Source numbers
    OutChannel = 1  # Number of channels in the output velocity model

dh = 10  # Space interval

####################################################
####             NETWORK PARAMETERS             ####
####################################################

if NetworkName == "InversionNet":
    LearnRate = 1e-4
elif NetworkName in ["TU_Net", "TU_Net_SEG"]:
    LearnRate = 3e-4
else:
    LearnRate = 1e-3


if DataSet == "FlatVelA/":
    Epochs = 140
    TrainSize = 24000  # 24000
    ValSize = 500
    TestSize = 500
    TestBatchSize = 10
    BatchSize = 20  # 64 # Number of batch size
    SaveEpoch = 5
    loss_weight = [1, 0.01]
elif DataSet == "CurveVelA/":
    Epochs = 140
    TrainSize = 24000  # 24000
    ValSize = 500
    TestSize = 6000
    TestBatchSize = 10
    BatchSize = 20  # 64 Number of batch size
    SaveEpoch = 5
    loss_weight = [1, 0.1]
elif DataSet == "FlatFaultA/":
    Epochs = 140
    TrainSize = 48000  # 24000
    ValSize = 500
    TestSize = 500
    TestBatchSize = 10
    BatchSize = 20  # 64 # Number of batch size
    SaveEpoch = 5
    loss_weight = [1, 0.01]
elif DataSet == "CurveFaultA/":
    Epochs = 140
    TrainSize = 48000  # 48000
    ValSize = 500
    TestSize = 500
    TestBatchSize = 10
    BatchSize = 20  # 64 # Number of batch size
    SaveEpoch = 5
    loss_weight = [1, 0.1]
elif DataSet == "SEGSaltData/":
    Epochs = 60
    TrainSize = 130  # 130
    ValSize = 2
    TestSize = 10
    TestBatchSize = 2
    BatchSize = 5  # 64 # Number of batch size
    SaveEpoch = 5
    loss_weight = [1, 1e6]
elif DataSet == "SEGSimulation/":
    Epochs = 160
    TrainSize = 1600  # 1600
    ValSize = 10
    TestSize = 100
    TestBatchSize = 2
    BatchSize = 10  # 64 # Number of batch size
    SaveEpoch = 1
    loss_weight = [1, 1e6]
