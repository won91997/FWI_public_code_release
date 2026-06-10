import time
from data.dataset import *
from data.loss import *
from utils import *
from model.TU_Net import *
from model.InversionNet import *
from model.FCNVMB import *
from model.DDNet import *
from model.DDNet70 import *
from model.TU_Net_SEG import *
import torch
from param_config import NetworkName

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

################################################
########             NETWORK            ########
################################################

# Here indicating the GPU you want to use. if you don't have GPU, just leave it.
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")

if NetworkName == "TU_Net":
    net = TU_Net(n_classes=OutChannel,
               in_channels=InChannel,
               is_deconv=True,
               is_batchnorm=True)

elif NetworkName == "TU_Net_SEG":
    net = TU_Net_SEG(n_classes=OutChannel,
                     in_channels=InChannel,
                     is_deconv=True,
                     is_batchnorm=True)

elif NetworkName == "InversionNet":
    net = InversionNet(n_classes=OutChannel,
                       in_channels=InChannel,
                       is_deconv=True,
                       is_batchnorm=True)

elif NetworkName == "DD_Net70":
    net = DDNet70Model(n_classes=OutChannel,
                       in_channels=InChannel,
                       is_deconv=True,
                       is_batchnorm=True)

elif NetworkName == "FCNVMB":
    net = FCNVMB(n_classes=OutChannel,
                 in_channels=InChannel,
                 is_deconv=True,
                 is_batchnorm=True)

elif NetworkName == "DD_Net":
    net = DDNetModel(n_classes=OutChannel,
                     in_channels=InChannel,
                     is_deconv=True,
                     is_batchnorm=True)

net = net.to(device)

# Optimizer we want to use
optimizer = torch.optim.Adam(net.parameters(), lr=LearnRate)

# If ReUse, it will load saved model from premodel_filepath and continue to train
if ReUse:
    print('***************** Loading pre-training model *****************')
    print('')
    premodel_file = train_result_dir + PreModelname
    net.load_state_dict(torch.load(premodel_file))
    net = net.to(device)
    print('Finish downloading:', str(premodel_file))

################################################
########    LOADING TRAINING DATA       ########
################################################
print('***************** Loading training dataset *****************')


trainSet = Dataset_train_edge(Data_path, TrainSize, 1, "seismic", "train")
train_loader = DataLoader(trainSet, batch_size=BatchSize, shuffle=True)
valSet = Dataset_test_edge(Data_path, ValSize, 1, "seismic", "test")
val_loader = DataLoader(valSet, batch_size=BatchSize, shuffle=True)

################################################
########            TRAINING            ########
################################################

print()
print('*******************************************')
print('*******************************************')
print('                Training ...               ')
print('*******************************************')
print('*******************************************')
print()

print('Seismic Data Dim:%s' % str(DataDim))
print('Velocity Model Dim:%s' % str(ModelDim))
print('Train size:%d' % int(TrainSize))
print('Batch size:%d' % int(BatchSize))
print('Epochs:%d' % int(Epochs))
print('Learning rate:%.5f' % float(LearnRate))

# Initialization
step = int(TrainSize / BatchSize)
start = time.time()


def train():
    net.train()
    total_loss = 0
    for i, (seismic_datas, velocity_models, edges) in enumerate(train_loader):
        net.train()
        seismic_datas = seismic_datas[0].to(device)  # Tensor:(20,5,1000,70)
        velocity_models = velocity_models[0].to(device).to(torch.float32)  # Tensor: (20,10,70,70)
        edges = edges[0].to(device).to(torch.float32)  # Tensor: (20,10,70,70)

        # Zero the gradient buffer
        optimizer.zero_grad()
        # Forward pass
        outputs = net(seismic_datas)

        if NetworkName in ["TU_Net", "TU_Net_SEG"]:
            outputs = outputs.to(torch.float32)
            # loss = criterion_12(outputs, velocity_models)
            loss = criterion_tu(outputs, velocity_models, edges)
        elif NetworkName in ["DD_Net", "DD_Net70"]:
            criterion_dd = LossDDNet(weights=loss_weight)
            outputs[0] = outputs[0].to(torch.float32)
            outputs[1] = outputs[1].to(torch.float32)
            loss = criterion_dd(outputs[0], outputs[1], velocity_models, edges)
        elif NetworkName == "InversionNet":
            outputs = outputs.to(torch.float32)
            loss = criterion_inv(outputs, velocity_models)
        elif NetworkName == "FCNVMB":
            outputs = outputs.to(torch.float32)
            loss = criterion_fcn(outputs, velocity_models)
        else:
            raise ValueError(f"Unknown NetworkName: {NetworkName}")

        if np.isnan(float(loss.item())):
            raise ValueError('loss is nan while training')

        total_loss += loss.item()
        loss = loss.to(torch.float32)  # Loss backward propagation
        loss.backward()
        optimizer.step()  # Optimize

    avg_loss = total_loss / len(train_loader)
    return avg_loss


def validate():
    total_loss = 0
    net.eval()
    with torch.no_grad():
        for i, (seismic_datas, velocity_models, edges, vmodel_max_min) in enumerate(val_loader):
            seismic_datas = seismic_datas[0].to(device)
            velocity_models = velocity_models[0].to(device).to(torch.float32)
            edges = edges[0].to(device).to(torch.float32)
            optimizer.zero_grad()  # Zero the gradient buffer
            outputs = net(seismic_datas)
            if NetworkName in ["TU_Net", "TU_Net_SEG"]:
                outputs = outputs.to(torch.float32)
                # loss = criterion_12(outputs, velocity_models)
                loss = criterion_tu(outputs, velocity_models, edges)
            elif NetworkName in ["DD_Net", "DD_Net70"]:
                criterion_dd = LossDDNet(weights=loss_weight)
                outputs[0] = outputs[0].to(torch.float32)
                outputs[1] = outputs[1].to(torch.float32)
                loss = criterion_dd(outputs[0], outputs[1], velocity_models, edges)
            elif NetworkName == "InversionNet":
                outputs = outputs.to(torch.float32)
                loss = criterion_inv(outputs, velocity_models)
            elif NetworkName == "FCNVMB":
                outputs = outputs.to(torch.float32)
                loss = criterion_fcn(outputs, velocity_models)
            else:
                raise ValueError(f"Unknown NetworkName: {NetworkName}")

            total_loss += loss.item()

        avg_loss = total_loss / len(val_loader)
        return avg_loss


train_loss_list = 0
val_loss_list = 0

for epoch in range(Epochs):
    epoch_loss = 0.0
    since = time.time()

    train_loss = train()
    val_loss = validate()

    # Show train and val loss every 1 epoch
    if (epoch % 1) == 0:
        print(f"Epoch: {epoch + 1},Train loss: {train_loss:.4f}, Val loss: {val_loss: .4f}")
        time_elapsed = time.time() - since
        print('Epoch consuming time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # Save net parameters every 10 epochs
    if (epoch + 1) % SaveEpoch == 0:
        torch.save(net.state_dict(), train_result_dir + ModelName + '_epoch' + str(epoch + 1) + '.pkl')
        print('Trained model saved: %d percent completed' % int((epoch + 1) * 100 / Epochs))

    train_loss_list = np.append(train_loss_list, train_loss)
    val_loss_list = np.append(val_loss_list, val_loss)

# Record the consuming time
time_elapsed = time.time() - start
print('Training complete in {:.0f}m  {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

# Save the loss
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 17,
         }
font3 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 21,
         }

SaveTrainResults(train_loss=train_loss_list,
                 val_loss=val_loss_list,
                 SavePath=train_result_dir,
                 ModelName=ModelName,
                 font2=font2,
                 font3=font3)
