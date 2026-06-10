# -*- coding: utf-8 -*-

from math import ceil

from func.datasets_reader import *
from func.comparison_net import InversionNet, FCNVMB
from DCNet import DCModel, LossDCNet
from SEGSimulate import  SEGSimulate

def determine_network(external_model_src = None, model_type = "PixelNet"):
    '''
    Request a network object and import an external network, or create an initialized network

    :param external_model_src:  External pkl file path
    :param model_type:          The main model used, this model is differentiated based on different papers.
                                The available key model keywords are [DDNet | DDNet70 | InversionNet | SwinTransNet]
    :return:                    A triplet: model object, GPU environment object and optimizer
    '''

    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    gpus = [0]

    # Network initialization
    if model_type == "DCNet":
        net_model = DCModel()
    elif model_type == "DDNet70":
        net_model = DDNet70Model(n_classes=classes,
                                 in_channels=inchannels,
                                 is_deconv=True,
                                 is_batchnorm=True)
    elif model_type == "InversionNet":
        net_model = InversionNet()
    elif model_type == "FCNVMB":
        net_model = FCNVMB(n_classes=classes,
                           in_channels=inchannels,
                           is_deconv=True,
                           is_batchnorm=True)
    else:
        print(
            'The "model_type" parameter selected in the determine_network(...)'
            ' is the undefined network model keyword! Please check!')
        exit(0)

    if external_model_src != None:
        net_model = model_reader(net=net_model, device=device, save_src=external_model_src)

    # Allocate GPUs and set optimizers
    if torch.cuda.is_available():
        net_model = torch.nn.DataParallel(net_model, device_ids=gpus).cuda()

    if model_type != "InversionNet":
        optimizer = torch.optim.Adam(net_model.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.Adam(net_model.parameters(), lr=0.0001)

    return net_model, device, optimizer

def load_dataset():
    '''
    Load the training data according to the parameters in "param_config"

    :return:
    '''

    print("---------------------------------")
    print("· Loading the datasets...")

    if dataset_name in ['SEGSalt', 'SEGSimulation']:
        data_set, label_sets = SEGSimulate(data_dir, 0, train_size, 29, "train", 5, 5, (70,70))
    else:
        data_set, label_sets = batch_read_npyfile(data_dir, 1, ceil(train_size / 500), "train")
        print("正在进行速度模型归一化")
        for i in range(data_set.shape[0]):
            vm = label_sets[0][i][0]
            label_sets[0][i][0] = (vm - np.min(vm)) / (np.max(vm) - np.min(vm))

    seis_and_vm = data_utils.TensorDataset(
        torch.from_numpy(data_set[:train_size, ...]).float(),
        torch.from_numpy(label_sets[0][:train_size, ...]).float(),
        torch.from_numpy(label_sets[1][:train_size, ...]).long())
    seis_and_vm_loader = data_utils.DataLoader(
        seis_and_vm,
        batch_size=train_batch_size,
        pin_memory=True,
        shuffle=True)

    print("· Number of seismic gathers included in the training set: {}".format(train_size))
    print("· Dimensions of seismic data: ({},{},{},{})".format(train_size, inchannels, data_dim[0], data_dim[1]))
    print("· Dimensions of velocity model: ({},{},{},{})".format(train_size, classes, model_dim[0], model_dim[1]))
    print("---------------------------------")

    return seis_and_vm_loader, data_set, label_sets

def preparation_stage_task(data_set, label_sets):
    '''
    Generate the corresponding stage1 difficulty set and dataset loader for
    all seismic gathers in advance, by this way, saving overhead

    :param data_set:    seismic gathers
    :param label_sets:  velocity models
    :return:            datasets loader
    '''

    if dataset_name in ['SEGSalt', 'SEGSimulation']:
        middle_shot_id = 15
        first_p = 9
        second_p = 18
    else:
        middle_shot_id = 2
        first_p = 2
        second_p = 4
    # Training set
    seis_and_vm = data_utils.TensorDataset(torch.from_numpy(data_set[:train_size, ...]).float(),
                                                     torch.from_numpy(label_sets[0][:train_size, ...]).float(),
                                                     torch.from_numpy(label_sets[1][:train_size, ...]).long())

    seis_and_vm_loader = data_utils.DataLoader(seis_and_vm,
                                                         batch_size=train_batch_size,
                                                         pin_memory=True,
                                                         shuffle=True)

    return seis_and_vm_loader

def train_for_stage(cur_epochs, model, training_loader, optimizer, model_type = "DCNet"):
    '''
    Training for designate epochs

    :param cur_epochs:      Designated epochs
    :param model:           Network model objects to be used for training
    :param training_loader: Trainin dataset loader to be fed into the network
    :param optimizer:       Optimizer
    :param model_type:      The main model used, this model is differentiated based on different papers.
                            The available key model keywords are [DCNet| DDNet | DDNet70 | InversionNet]
    :return:                Model save path and runtime
    '''

    loss_of_stage = 0.0
    last_model_save_path = ""
    step = int(train_size / train_batch_size)       # Total number of batches to train
    save_times = 10                                  # How many times do I need to save the intermediate results of the model
    save_epoch = cur_epochs // save_times
    training_time = 0

    model_save_name = "{}_TrSize{}_AllEpo{}".format(dataset_name, train_size, cur_epochs)

    for epoch in range(cur_epochs):
        # Training for the current epoch
        loss_of_epoch = 0.0
        cur_node_time = time.time()
        ############
        # training #
        ############
        for i, (images, labels, contours_labels) in enumerate(training_loader):

            iteration = epoch * step + i + 1
            model.train()

            # Load to GPU
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
                contours_labels = contours_labels.cuda(non_blocking=True)

            # Gradient cache clearing
            optimizer.zero_grad()
            criterion = LossDCNet(weights=[0.2, 0.8] if model_type == "PixelNet" else [1, 0])
            outputs = model(images, model_dim)

            loss = criterion(outputs,labels)

            if np.isnan(float(loss.item())):
                raise ValueError('loss is nan while training')

            # Loss backward propagation
            loss.backward()

            # Optimize
            optimizer.step()

            loss_of_epoch += loss.item()

            if iteration % display_step == 0:
                print('Epochs: {}/{}, Iteration: {}/{} --- Training Loss:{:.6f}'
                      .format(epoch + 1, cur_epochs, iteration, step * cur_epochs, loss.item()))

        ################################
        # The end of the current epoch #
        ################################
        if (epoch + 1) % 1 == 0:

            # Calculate the average loss of the current epoch
            print('Epochs: {:d} finished ! Training loss: {:.5f}'
                  .format(epoch + 1, loss_of_epoch / i))

            # Include the average loss in the array belonging to the current stage
            loss_of_stage = np.append(loss_of_stage, loss_of_epoch / i)

            # Statistics of the time spent in a epoch
            time_elapsed = time.time() - cur_node_time
            print('Epochs consuming time: {:.0f}m {:.0f}s'
                  .format(time_elapsed // 60, time_elapsed % 60))
            training_time += time_elapsed
        #########################################################################
        # When it reaches the point where intermediate results can be stored... #
        #########################################################################
        if (epoch + 1) % save_epoch == 0:
            last_model_save_path = models_dir + model_save_name + '_CurEpo' + str(epoch + 1) + '.pkl'
            torch.save(model.state_dict(), last_model_save_path)
            print('Trained model saved: %d percent completed' % int((epoch + 1) * 100 / cur_epochs))
    print("save loss image")
    save_results(loss=loss_of_stage, epochs=cur_epochs, save_path=results_dir,
                 xtitle='Num. of epochs', ytitle='Num. of epochs',
                 title='Training Loss')

    return last_model_save_path, training_time

def curriculum_learning_training(model_type):
    '''
    Curriculum learning
    '''
    all_training_time = 0
    #priori_model_src = None
    priori_model_src = None
    training_loader, seismic_gathers, velocity_models = load_dataset()

    if epochs != 0:
        dd_net_init, device, optimizer = determine_network(priori_model_src, model_type=model_type)
        training_loader = preparation_stage_task(seismic_gathers, velocity_models)
        priori_model_src, training_time = train_for_stage(epochs, dd_net_init, training_loader, optimizer,
                                               model_type = model_type)
        all_training_time += training_time
        del training_loader

    print("training runtime: {}s".format(all_training_time))
    loss_mat_dir = results_dir + "Training Loss.mat"
    loss_stage1 = scipy.io.loadmat(loss_mat_dir)['loss'][0][0:]

    save_results(loss=loss_stage1,
                 epochs=epochs, save_path=results_dir, xtitle='Num. of epochs',
                 ytitle='Num. of epochs', title='Loss', is_show=False)

if __name__ == "__main__":
    # DCNet
    curriculum_learning_training(model_type="DCNet")
