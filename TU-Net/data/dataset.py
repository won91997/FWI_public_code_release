from torch.utils.data import Dataset, DataLoader, TensorDataset
from math import ceil
from utils import *
from data.show import *
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def batch_read_npy_file(para_dataset_dir,
                        para_start,
                        para_batch_length,
                        para_seismic_flag = "seismic",
                        para_train_or_test = "train"):
    '''
    Batch read seismic gathers and velocity models for .npy file
    :param dataset_dir:             Path to the dataset
    :param start:                   Start reading from the number of data
    :param batch_length:            Starting from the defined first number of data, how long to read
    :param train_or_test:           Whether the read data is used for training or testing ("train" or "test")
    :return:                        a pair: (seismic data, velocity model, contour of velocity model)
                                    Among them, the dimensions of seismic data, velocity model and contour of velocity
                                    model are all (number of read data * 500, channel, height, width)
                                    dataset （500,5,1000,70）  地震数据  未经任何处理
                                    labelset, （500,1,70,70）  速度模型  未经任何处理
    '''

    dataset_list = []
    labelset_list = []

    for i in range(para_start, para_start + para_batch_length):

        ##############################
        ##    Load Seismic Data     ##
        ##############################

        # Determine the seismic data path in the dataset
        filename_seis = para_dataset_dir + '{}_data/{}/seismic{}.npy'.format(para_train_or_test, para_seismic_flag, i)
        print("Reading: {}".format(filename_seis))

        data = np.load(filename_seis)
        dataset_list.append(data)

        ##############################
        ##    Load Velocity Model   ##
        ##############################

        # Determine the velocity model path in the dataset
        filename_label = para_dataset_dir + '{}_data/vmodel/vmodel{}.npy'.format(para_train_or_test, i)
        print("Reading: {}".format(filename_label))
        label = np.load(filename_label)
        labelset_list.append(label)

    dataset = np.concatenate(dataset_list, axis=0)
    labelset = np.concatenate(labelset_list, axis=0)
    return dataset, labelset


def batch_read_npy_file_edge(dataset_dir,
                       start,
                       batch_length,
                       seismic_flag = "seismic",
                       train_or_test = "train"):
    '''
    Batch read seismic gathers and velocity models for .npy file

    :param dataset_dir:             Path to the dataset
    :param start:                   Start reading from the number of data
    :param batch_length:            Starting from the defined first number of data, how long to read
    :param train_or_test:           Whether the read data is used for training or testing ("train" or "test")
    :return:                        a pair: (seismic data, velocity model, contour of velocity model)
                                    Among them, the dimensions of seismic data, velocity model and contour of velocity
                                    model are all (number of read data * 500, channel, height, width)
                                    dataset （500,5,1000,70）  地震数据  未经任何处理
                                    labelset, （500,1,70,70）  速度模型  未经任何处理
                                    edgeset    (500,1,70,70)   二值边缘图像
    '''

    dataset_list = []
    labelset_list = []
    edgeset_list = []

    for i in range(start, start + batch_length):
        ##############################
        ##    Load Seismic Data     ##
        ##############################

        # Determine the seismic data path in the dataset
        filename_seis = dataset_dir + '{}_data/{}/seismic{}.npy'.format(train_or_test, seismic_flag, i)
        # filename_seis = dataset_dir + '{}_data/seismic/seismic{}.npy'.format(train_or_test, i)
        print("Reading: {}".format(filename_seis))

        datas = np.load(filename_seis)
        dataset_list.append(datas)

        ##############################
        ##    Load Velocity Model   ##
        ##############################

        # Determine the velocity model path in the dataset
        filename_label = dataset_dir + '{}_data/vmodel/vmodel{}.npy'.format(train_or_test, i)
        print("Reading: {}".format(filename_label))
        labels = np.load(filename_label)
        labelset_list.append(labels)


        ###################################
        ##    Generating Velocity Edge   ##
        ###################################

        print("Generating velocity model profile......")
        edges = np.zeros([500, OutChannel, ModelDim[0], ModelDim[1]])
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                edges[i, j, ...] = extract_contours(labels[i, j, ...])
        edgeset_list.append(edges)

    dataset = np.concatenate(dataset_list, axis=0)
    labelset = np.concatenate(labelset_list, axis=0)
    edgeset = np.concatenate(edgeset_list, axis=0)

    return dataset, labelset, edgeset


def batch_read_mat_file(dataset_dir,
                       start,
                       batch_length,
                       train_or_test="train",
                       data_channels=29):
    '''
    Batch read seismic gathers and velocity models for .mat file

    :param dataset_dir:             Path to the dataset
    :param start:                   Start reading from the number of data
    :param batch_length:            Starting from the defined first number of data, how long to read
    :param train_or_test:           Whether the read data is used for training or testing ("train" or "test")
    :param data_channels:           The total number of channels read into the data itself
    :return:                        a quadruple: (seismic data, [velocity model, contour of velocity model])
                                    Among them, the dimensions of seismic data, velocity model and contour of velocity model are all (number of read data, channel, width x height)
    '''

    data_set = np.zeros([batch_length, data_channels, DataDim[0], DataDim[1]])
    label_set = np.zeros([batch_length, OutChannel, ModelDim[0], ModelDim[1]])

    for indx, i in enumerate(range(start, start + batch_length)):

        # Load Seismic Data
        # filename_seis = dataset_dir + '{}_data/georec/georec{}.mat'.format(train_or_test, i)
        # filename_seis = dataset_dir + '{}_data/seismic/seismic{}.mat'.format(train_or_test, i)
        if train_or_test == "train":
            filename_seis = dataset_dir + '{}_data/georec_train/georec{}.mat'.format(train_or_test, i)
        else:
            # filename_seis = dataset_dir + '{}_data/georec_test/georec{}.mat'.format(train_or_test, i)
            filename_seis = dataset_dir + '{}_data/seismic_test/seismic{}.mat'.format(train_or_test, i)
        print("Reading: {}".format(filename_seis))
        # sei_data = scipy.io.loadmat(filename_seis)["Rec"]
        sei_data = scipy.io.loadmat(filename_seis)["Rec"]
        # (400, 301, 29) -> (29, 400, 301)
        sei_data = sei_data.swapaxes(0, 2)
        sei_data = sei_data.swapaxes(1, 2)
        for ch in range(data_channels):
            data_set[indx, ch, ...] = sei_data[ch, ...]

        # Load Velocity Model
        if SEGSaltData == True and train_or_test == "test":
            filename_label = dataset_dir + '{}_data/vmodel/vmodel{}.mat'.format(train_or_test, i)
            vm_data = scipy.io.loadmat(filename_label)["data"]
        if SEGSaltData == True and train_or_test == "train":
            filename_label = dataset_dir + '{}_data/vmodel/svmodel{}.mat'.format(train_or_test, i)
            vm_data = scipy.io.loadmat(filename_label)["svmodel"]
        if SEGSimulateData == True and train_or_test == "train":
            filename_label = dataset_dir + '{}_data/vmodel_train/vmodel{}.mat'.format(train_or_test, i)
            vm_data = scipy.io.loadmat(filename_label)["vmodel"]
        else:
            filename_label = dataset_dir + '{}_data/vmodel_test/vmodel{}.mat'.format(train_or_test, i)
            vm_data = scipy.io.loadmat(filename_label)["vmodel"]

            # filename_label = dataset_dir + '{}_data/vmodel/vmodel{}.mat'.format(train_or_test, i)
            # filename_label = dataset_dir + '{}_data/vmodel/vmodel{}.mat'.format(train_or_test, i)
            # vm_data = scipy.io.loadmat(filename_label)["vmodel"]

        print("Reading: {}".format(filename_label))
        label_set[indx, 0, ...] = vm_data

    return data_set, label_set


def batch_read_mat_file_edge(dataset_dir,
                       start,
                       batch_length,
                       train_or_test="train",
                       data_channels=29):
    '''
    Batch read seismic gathers and velocity models for .mat file

    :param dataset_dir:             Path to the dataset
    :param start:                   Start reading from the number of data
    :param batch_length:            Starting from the defined first number of data, how long to read
    :param train_or_test:           Whether the read data is used for training or testing ("train" or "test")
    :param data_channels:           The total number of channels read into the data itself
    :return:                        a quadruple: (seismic data, [velocity model, contour of velocity model])
                                    Among them, the dimensions of seismic data, velocity model and contour of velocity model are all (number of read data, channel, width x height)
    '''

    data_set = np.zeros([batch_length, data_channels, DataDim[0], DataDim[1]])
    label_set = np.zeros([batch_length, OutChannel, ModelDim[0], ModelDim[1]])
    edge_set = np.zeros([batch_length, OutChannel, ModelDim[0], ModelDim[1]])

    for indx, i in enumerate(range(start, start + batch_length)):

        # Load Seismic Data
        # filename_seis = dataset_dir + '{}_data/georec_{}/georec{}.mat'.format(train_or_test, train_or_test, i)
        filename_seis = dataset_dir + '{}_data/seismic/seismic{}.mat'.format(train_or_test, i)
        print("Reading: {}".format(filename_seis))
        sei_data = scipy.io.loadmat(filename_seis)["Rec"]
        # sei_data = scipy.io.loadmat(filename_seis)["data"]
        # (400, 301, 29) -> (29, 400, 301)
        sei_data = sei_data.swapaxes(0, 2)
        sei_data = sei_data.swapaxes(1, 2)
        for ch in range(data_channels):
            data_set[indx, ch, ...] = sei_data[ch, ...]

        # Load Velocity Model
        if SEGSaltData == True and train_or_test == "test":
            filename_label = dataset_dir + '{}_data/vmodel/svmodel{}.mat'.format(train_or_test, i)
            vm_data = scipy.io.loadmat(filename_label)["svmodel"]
        if SEGSaltData == True and train_or_test == "train":
            filename_label = dataset_dir + '{}_data/vmodel/svmodel{}.mat'.format(train_or_test, i)
            vm_data = scipy.io.loadmat(filename_label)["svmodel"]
        if SEGSimulateData == True:
            filename_label = dataset_dir + '{}_data/vmodel/vmodel{}.mat'.format(train_or_test, i)
            vm_data = scipy.io.loadmat(filename_label)["vmodel"]
        print("Reading: {}".format(filename_label))
        label_set[indx, 0, ...] = vm_data
        edge_set[indx, 0, ...] = extract_contours(vm_data)

    return data_set, label_set, edge_set


class Dataset_train(Dataset):
    def __init__(self, para_data_dir, para_train_size, para_start_num, para_seismic_flag, train_or_test):
        # start_num: 1 for train, 11 for test
        print("---------------------------------")
        print("· Loading the datasets...")

        if OpenFWI == True:
            data_set, label_set= batch_read_npy_file(para_data_dir, para_start_num, ceil(para_train_size / 500),
                                                                para_seismic_flag, train_or_test)
            for i in range(data_set.shape[0]):
                vm = label_set[i][0]
                label_set[i][0] = (vm - np.min(vm)) / (np.max(vm) - np.min(vm))
        else:
            data_set, label_set = batch_read_mat_file(para_data_dir, para_start_num, para_train_size, train_or_test)
            # for i in range(data_set.shape[0]):
            #     vm = label_set[i][0]
            #     label_set[i][0] = (vm - np.min(vm)) / (np.max(vm) - np.min(vm))

        ###################################################################################
        #                          速度模型归一化                                            #
        ###################################################################################
        print("Normalization in progress...")
        # Training set
        self.seismic_data = TensorDataset(torch.from_numpy(data_set[:para_train_size, ...]).float())
        self.vmodel = TensorDataset(torch.from_numpy(label_set[:para_train_size, ...]).float())


    def __getitem__(self, index):
        return self.seismic_data[index], self.vmodel[index]

    def __len__(self):
        return len(self.seismic_data)


class Dataset_train_edge(Dataset):
    '''
       # Load the training data. It includes only seismic_data and vmodel.
       param_seismic_flag: for loading different frequency seismic data. It is designed for domain adaption.
    '''

    def __init__(self, para_data_dir, para_train_size, para_start_num, para_seismic_flag, para_train_or_test):
        # para_start_num: 1 for train, 11 for test
        print("---------------------------------")
        print("Loading the datasets...")

        if OpenFWI == True:
            data_set, label_set, edge_set = batch_read_npy_file_edge(para_data_dir, para_start_num, ceil(para_train_size / 500), para_seismic_flag, para_train_or_test)
            # for each sample
            for i in range(data_set.shape[0]):
                vm = label_set[i][0]
                label_set[i][0] = (vm - np.min(vm)) / (np.max(vm) - np.min(vm))
        else:
            data_set, label_set, edge_set = batch_read_mat_file_edge(para_data_dir, para_start_num, para_train_size, para_train_or_test)
            # for i in range(data_set.shape[0]):
            #     # {ndarray:(70,70)}
            #     vm = label_set[i][0]
            #     # print(np.min(vm))
            #     label_set[i][0] = (vm - np.min(vm)) / (np.max(vm) - np.min(vm))

        ###################################################################################
        #                          Vmodel normalization                                   #
        ###################################################################################
        print("Normalization in progress...")

        ###################################################################################
        self.seismic_data = TensorDataset(torch.from_numpy(data_set[:para_train_size, ...]).float())
        self.vmodel = TensorDataset(torch.from_numpy(label_set[:para_train_size, ...]).float())
        self.edge = TensorDataset(torch.from_numpy(edge_set[:para_train_size, ...]).float())

    def __getitem__(self, index):
        return self.seismic_data[index], self.vmodel[index], self.edge[index]

    def __len__(self):
        return len(self.seismic_data)


class Dataset_test(Dataset):
    '''
       # Load the test data. It includes seismic_data, vmodel, and vmodel_max_min.
       # vmodel_max_min is used for inverting normalization.
    '''

    def __init__(self, para_data_dir, para_train_size, para_start_num, para_seismic_flag, para_train_or_test):
        # start_num: 1 for train, 11 for test
        print("---------------------------------")
        print("· Loading the datasets...")
        vmodel_max_min = np.zeros((para_train_size, 2))

        if OpenFWI == True:
            data_set, label_set = batch_read_npy_file(para_data_dir, para_start_num, ceil(para_train_size / 500),
                                                            para_seismic_flag, para_train_or_test)
            # for each sample
            for i in range(data_set.shape[0]):
                vm = label_set[i][0]
                vmodel_max_min[i, 0] = np.max(vm)
                vmodel_max_min[i, 1] = np.min(vm)
                label_set[i][0] = (vm - np.min(vm)) / (np.max(vm) - np.min(vm))
        else:
            data_set, label_set = batch_read_mat_file(para_data_dir, para_start_num, para_train_size, para_train_or_test)
            # for each sample
            for i in range(data_set.shape[0]):
                vm = label_set[i][0]
                vmodel_max_min[i, 0] = np.max(vm)
                vmodel_max_min[i, 1] = np.min(vm)
                # label_set[i][0] = (vm - np.min(vm)) / (np.max(vm) - np.min(vm))

        ###################################################################################
        #                          Vmodel normalization                                   #
        ###################################################################################
        print("Normalization in progress...")


        # ##################################################################################
        # Training set
        self.seismic_data = TensorDataset(torch.from_numpy(data_set[:para_train_size, ...]).float())
        self.vmodel = TensorDataset(torch.from_numpy(label_set[:para_train_size, ...]).float())
        self.vmodel_max_min = TensorDataset(torch.from_numpy(vmodel_max_min[:para_train_size, ...]).float())

    def __getitem__(self, index):
        return self.seismic_data[index], self.vmodel[index], self.vmodel_max_min[index]

    def __len__(self):
        return len(self.seismic_data)


class Dataset_test_edge(Dataset):
    '''
       # Load the test data. It includes seismic_data, vmodel, and vmodel_max_min.
       # vmodel_max_min is used for inverting normalization.
    '''

    def __init__(self, para_data_dir, para_train_size, para_start_num, para_seismic_flag, para_train_or_test):
        # start_num: 1 for train, 11 for test
        print("---------------------------------")
        print("· Loading the datasets...")

        vmodel_max_min = np.zeros((para_train_size, 2))
        if OpenFWI == True:
            data_set, label_set, edge_set = batch_read_npy_file_edge(para_data_dir, para_start_num, ceil(para_train_size / 500),
                                                            para_seismic_flag, para_train_or_test)
            # for each sample
            for i in range(data_set.shape[0]):
                vm = label_set[i][0]
                vmodel_max_min[i, 0] = np.max(vm)
                vmodel_max_min[i, 1] = np.min(vm)
                label_set[i][0] = (vm - np.min(vm)) / (np.max(vm) - np.min(vm))
        else:
            data_set, label_set, edge_set = batch_read_mat_file_edge(para_data_dir, para_start_num, para_train_size, para_train_or_test)
            # for each sample
            for i in range(data_set.shape[0]):
                vm = label_set[i][0]
                vmodel_max_min[i, 0] = np.max(vm)
                vmodel_max_min[i, 1] = np.min(vm)
                # label_set[i][0] = (vm - np.min(vm)) / (np.max(vm) - np.min(vm))


        ###################################################################################
        #                          Vmodel normalization                                   #
        ###################################################################################
        print("Normalization in progress...")

        # ##################################################################################
        # Training set
        self.seismic_data = TensorDataset(torch.from_numpy(data_set[:para_train_size, ...]).float())
        self.vmodel = TensorDataset(torch.from_numpy(label_set[:para_train_size, ...]).float())
        self.edge = TensorDataset(torch.from_numpy(edge_set[:para_train_size, ...]).float())
        self.vmodel_max_min = TensorDataset(torch.from_numpy(vmodel_max_min[:para_train_size, ...]).float())

    def __getitem__(self, index):
        return self.seismic_data[index], self.vmodel[index], self.edge[index], self.vmodel_max_min[index]

    def __len__(self):
        return len(self.seismic_data)



if __name__ == '__main__':
    dataset_dir = 'G:/Data/OpenFWI/CurveFaultA/'
    # batch_read_npyfile(dataset_dir,1,2,train_or_test="train")
    data_set = Dataset_train_edge(dataset_dir, 1000, 1, "seismic", "train")
    data_loader = DataLoader(data_set, batch_size=10, shuffle=True)
    # Traverse the dataset
    for (data_set) in data_loader:
        print("ok ")
