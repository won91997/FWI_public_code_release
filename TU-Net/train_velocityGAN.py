################################################
########            导入库               ########
################################################
import os
import torch
import time
from torch.utils.data import DataLoader
from model.InversionNet import *
from path_config import *
from data.dataset import *
from data.loss import *
from utils import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

################################################
########             NETWORK            ########
################################################

# Here indicating the GPU you want to use. if you don't have GPU, just leave it.
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")

net = InversionNet()
net = net.to(device)
net_d = Discriminator()
net_d = net_d.to(device)

optimizer_g = torch.optim.Adam(net.parameters(), lr=LearnRate)
optimizer_d = torch.optim.Adam(net_d.parameters(), lr=LearnRate)

# If ReUse, it will load saved model from premodel filepath and continue to train
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
print('***************** Loading dataset *****************')

dataset_dir = Data_path

trainSet = Dataset_train_edge(dataset_dir, TrainSize, 1, "seismic", "train")
train_loader = DataLoader(trainSet, batch_size=BatchSize, shuffle=True)

valSet = Dataset_test_edge(dataset_dir, ValSize, 1, "seismic", "test")
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
loss1 = 0.0
step = int(TrainSize / BatchSize)
start = time.time()

n_critic = 5
max_itr = len(train_loader)


def train():
    total_loss_g = 0
    total_loss_d = 0

    net.train()
    net_d.train()

    iter_g = 0
    for i, (seismic_datas, vmodels, edges) in enumerate(train_loader):

        seismic_datas = seismic_datas[0].to(device)
        vmodels = vmodels[0].to(device)

        optimizer_d.zero_grad()

        with torch.no_grad():
            outputs = net(seismic_datas)

        criterion_d = Wasserstein_GP(device, 10)
        loss_d, loss_diff, loss_gp = criterion_d(vmodels, outputs, net_d)
        loss_d.backward()
        optimizer_d.step()
        total_loss_d += loss_d.item()

        if ((i + 1) % n_critic == 0) or (i == max_itr - 1):
            optimizer_g.zero_grad()
            outputs = net(seismic_datas)

            loss_g, loss_g1v, loss_g2v = criterion_g(outputs, vmodels, net_d)
            loss_g.backward()
            optimizer_g.step()
            total_loss_g += loss_g.item()
            iter_g = iter_g + 1

    avg_loss_g = total_loss_g / iter_g
    avg_loss_d = total_loss_d / len(train_loader)

    return avg_loss_g, avg_loss_d


def validate():
    total_loss = 0
    net.eval()

    with torch.no_grad():
        for i, (seismic_datas, vmodels, edge, vmodel_max_min) in enumerate(val_loader):
            seismic_datas = seismic_datas[0].to(device)
            vmodels = vmodels[0].to(device)

            outputs = net(seismic_datas)

            outputs = outputs.to(torch.float32)
            vmodels = vmodels.to(torch.float32)

            loss_g1v = l1loss(outputs, vmodels)
            loss_g2v = l2loss(outputs, vmodels)
            loss = loss_g1v + loss_g2v

            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    return avg_loss


train_loss_list_g = 0
train_loss_list_d = 0
val_loss_list = 0

for epoch in range(Epochs):
    epoch_loss = 0.0
    since = time.time()

    train_loss_g, train_loss_d = train()
    val_loss = validate()

    if (epoch % 1) == 0:
        print(f"Epoch: {epoch + 1}, Train loss:{train_loss_g:.4f},Val loss: {val_loss: .4f}")
        time_elapsed = time.time() - since
        print('Epoch consuming time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # Save net parameters every 10 epochs
    if (epoch + 1) % SaveEpoch == 0:
        torch.save(net.state_dict(), train_result_dir + ModelName + '_epoch' + str(epoch + 1) + '.pkl')
        torch.save(net_d.state_dict(), train_result_dir + ModelName + '_d_epoch' + str(epoch + 1) + '.pkl')
        print('Trained model saved: %d percent completed' % int((epoch + 1) * 100 / Epochs))

    train_loss_list_g = np.append(train_loss_list_g, train_loss_g)
    train_loss_list_d = np.append(train_loss_list_d, train_loss_d)
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

SaveTrainValidGANResults(train_loss_g=train_loss_list_g,
                         train_loss_d=train_loss_list_d,
                         val_loss=val_loss_list,
                         SavePath=train_result_dir,
                         ModelName=ModelName,
                         font2=font2,
                         font3=font3)
