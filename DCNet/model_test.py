# -*- coding: utf-8 -*-
from random import Random
import argparse
import os
import sys
import subprocess

import param_config
from func.datasets_reader import *
from model_train import determine_network

def load_dataset():
    '''
    Load the testing data according to the parameters in "param_config"

    :return:    A triplet: datasets loader, seismic gathers and velocity models
    '''

    print("---------------------------------")
    print("· Loading the datasets...")
    if dataset_name in ['SEGSalt', 'SEGSimulation']:
        # data_set, label_sets = batch_read_matfile(data_dir, 1 if dataset_name == 'SEGSalt'
        #                                                       else 1601, test_size, "test")
        data_set, label_sets = batch_read_matfile(data_dir, 1 if dataset_name == 'SEGSalt'
        else 1601, test_size, "test")
    else:
        data_set, label_sets = batch_read_npyfile(data_dir, 1, test_size // 500, "test")
        for i in range(data_set.shape[0]):
            vm = label_sets[0][i][0]
            max_velocity, min_velocity = np.max(vm), np.min(vm)
            label_sets[0][i][0] = (vm - min_velocity) / (max_velocity - min_velocity)

            #vm = label_sets[i][0]
            #label_sets[i][0] = (vm - np.min(vm)) / (np.max(vm) - np.min(vm))

    print("· Number of seismic gathers included in the testing set: {}.".format(test_size))
    print("· Dimensions of seismic data: ({},{},{},{}).".format(test_size, inchannels, data_dim[0], data_dim[1]))
    print("· Dimensions of velocity model: ({},{},{},{}).".format(test_size, classes, model_dim[0], model_dim[1]))
    print("---------------------------------")

    seis_and_vm = data_utils.TensorDataset(torch.from_numpy(data_set).float(),
                                           torch.from_numpy(label_sets[0]).float())
    seis_and_vm_loader = data_utils.DataLoader(seis_and_vm, batch_size=test_batch_size, shuffle=True)

    return seis_and_vm_loader, data_set, label_sets

def batch_test(model_path, model_type = "DDNet", export_dir=None, benchmark_eval=None, benchmark_eval_json=None):
    '''
    Batch testing for multiple seismic data

    :param model_path:              Model path
    :param model_type:              The main model used, this model is differentiated based on different papers.
    :return:
    '''

    loader, seismic_gathers, velocity_models = load_dataset()

    print("Loading test model:{}".format(model_path))
    model_net, device, optimizer = determine_network(model_path, model_type=model_type)

    mse_record = np.zeros((1, test_size), dtype=float)
    mae_record = np.zeros((1, test_size), dtype=float)
    uqi_record = np.zeros((1, test_size), dtype=float)
    lpips_record = np.zeros((1, test_size), dtype=float)

    counter = 0

    lpips_object = lpips.LPIPS(net='alex', version="0.1")

    pred_all, gt_all = [], []
    cur_node_time = time.time()
    for i, (seis_image, gt_vmodel) in enumerate(loader):

        if torch.cuda.is_available():
            seis_image = seis_image.cuda(non_blocking=True)
            gt_vmodel = gt_vmodel.cuda(non_blocking=True)

        # Prediction
        model_net.eval()
        if model_type in ["DDNet", "DDNet70"]:
            [outputs, _] = model_net(seis_image, model_dim)
        elif model_type == "InversionNet":
            outputs = model_net(seis_image)
        elif model_type == "FCNVMB":
            outputs = model_net(seis_image, model_dim)
        elif model_type == "DCNet":
            outputs = model_net(seis_image, model_dim)
        else:
            print('The "model_type" parameter selected in the batch_test(...) '
                  'is the undefined network model keyword! Please check!')
            exit(0)

        # # Both target labels and prediction tags return to "numpy"
        pd_vmodel = outputs.cpu().detach().numpy()
        pd_vmodel = np.where(pd_vmodel > 0.0, pd_vmodel, 0.0)   # Delete bad points
        gt_vmodel = gt_vmodel.cpu().detach().numpy()
        pred_all.append(pd_vmodel[:, 0])
        gt_all.append(gt_vmodel[:, 0])

        # Calculate MSE, MAE, UQI and LPIPS of the current batch
        for k in range(test_batch_size):

            pd_vmodel_of_k = pd_vmodel[k, 0, :, :]
            gt_vmodel_of_k = gt_vmodel[k, 0, :, :]

            mse_record[0, counter]   = run_mse(pd_vmodel_of_k, gt_vmodel_of_k)
            mae_record[0, counter]   = run_mae(pd_vmodel_of_k, gt_vmodel_of_k)
            uqi_record[0, counter]   = run_uqi(gt_vmodel_of_k, pd_vmodel_of_k)
            lpips_record[0, counter] = run_lpips(gt_vmodel_of_k, pd_vmodel_of_k, lpips_object)

            print('The %d testing MSE: %.7f\tMAE: %.7f\tUQI: %.7f\tLPIPS: %.7f' %
                  (counter, mse_record[0, counter], mae_record[0, counter],
                   uqi_record[0, counter], lpips_record[0, counter]))
            counter = counter + 1
    time_elapsed = time.time() - cur_node_time

    print("The average of MSE: {:.7f}".format(mse_record.mean()))
    print("The average of MAE: {:.7f}".format(mae_record.mean()))
    print("The average of UQI: {:.7f}".format(uqi_record.mean()))
    print("The average of LIPIS: {:.7f}".format(lpips_record.mean()))
    print("-----------------")
    print("Time-consuming testing of batch samples: {:.67f}".format(time_elapsed))
    print("Average test-consuming per sample: {:.7f}".format(time_elapsed / test_size))

    if export_dir:
        os.makedirs(export_dir, exist_ok=True)
        pred_path = os.path.join(export_dir, "pred.npy")
        gt_path = os.path.join(export_dir, "gt.npy")
        np.save(pred_path, np.concatenate(pred_all, axis=0))
        np.save(gt_path, np.concatenate(gt_all, axis=0))
        print(f"Exported predictions: {pred_path}")
        print(f"Exported ground truth: {gt_path}")
        if benchmark_eval:
            cmd = [sys.executable, benchmark_eval, "--pred", pred_path, "--gt", gt_path]
            if benchmark_eval_json:
                cmd.extend(["--out-json", benchmark_eval_json])
            print("Running unified benchmark evaluator...")
            subprocess.run(cmd, check=True)

def single_test(model_path, select_id, save_path, save_name, train_or_test = "test", model_type = "MixFormer"):
    '''
    Batch testing for single seismic data

    :param model_path:              Model path
    :param select_id:               The ID of the selected data. if it is openfwi, here is a pair,
                                    e.g. [11, 100], otherwise it is just a single number, e.g. 56.
    :param train_or_test:           Whether the data set belongs to the training set or the testing set
    :param model_type:              The main model used, this model is differentiated based on different papers.
                                    The available key model keywords are [DDNet70 | DDNet | InversionNet | SwinTransNet]
    :return:
    '''

    print("Loading test model:{}".format(model_path))
    model_net, device, optimizer = determine_network(model_path, model_type = model_type)



    if dataset_name in ['SEGSalt', 'SEGSimulation']:
        seismic_data, velocity_model, _ = single_read_matfile(data_dir, data_dim, model_dim, select_id, train_or_test = train_or_test)
        max_velocity, min_velocity = np.max(velocity_model), np.min(velocity_model)
    else:
        seismic_data, velocity_model, _ = single_read_npyfile(data_dir, select_id, train_or_test = train_or_test)
        max_velocity, min_velocity = np.max(velocity_model), np.min(velocity_model)
        velocity_model = (velocity_model - np.min(velocity_model)) / (np.max(velocity_model) - np.min(velocity_model))

    lpips_object = lpips.LPIPS(net='alex', version="0.1")

    # Convert numpy to tensor and load it to GPU
    seismic_data_tensor = torch.from_numpy(np.array([seismic_data])).float()
    if torch.cuda.is_available():
        seismic_data_tensor = seismic_data_tensor.cuda(non_blocking=True)

    # Prediction
    model_net.eval()
    cur_node_time = time.time()
    if model_type in ["DDNet", "DDNet70"]:
        [predicted_vmod_tensor, _] = model_net(seismic_data_tensor, model_dim)
    elif model_type == "InversionNet":
        predicted_vmod_tensor = model_net(seismic_data_tensor)
    elif model_type == "FCNVMB":
        predicted_vmod_tensor = model_net(seismic_data_tensor, model_dim)
    elif model_type == "DCNet":
        predicted_vmod_tensor = model_net(seismic_data_tensor, model_dim)
    else:
        print('The "model_type" parameter selected in the single_test(...) '
              'is the undefined network model keyword! Please check!')
        exit(0)
    time_elapsed = time.time() - cur_node_time


    predicted_vmod = predicted_vmod_tensor.cpu().detach().numpy()[0][0]  # (1, 1, X, X)
    predicted_vmod = np.where(predicted_vmod > 0.0, predicted_vmod, 0.0)  # Delete bad points


    mse   = run_mse(predicted_vmod, velocity_model)
    mae   = run_mae(predicted_vmod, velocity_model)
    uqi   = run_uqi(velocity_model, predicted_vmod)
    lpi = run_lpips(velocity_model, predicted_vmod, lpips_object)

    print('MSE: %.6f\nMAE: %.6f\nUQI: %.6f\nLPIPS: %.6f' % (mse, mae, uqi, lpi))
    print("-----------------")
    print("Time-consuming testing of a sample: {:.6f}".format(time_elapsed))

    with open('image_values_true.txt', 'w') as f:
        for row in velocity_model:
            f.write(' '.join(f'{pixel:.6f}' for pixel in row) + '\n')

    with open('image_values_pred.txt', 'w') as f:
        for row in predicted_vmod:
            f.write(' '.join(f'{pixel:.6f}' for pixel in row) + '\n')

    # Show
    if dataset_name in ['SEGSalt', 'SEGSimulation']:
        pain_seg_seismic_data(seismic_data[15])
        pain_seg_velocity_model(velocity_model, min_velocity, max_velocity, save_path, "ground_true")
        pain_seg_velocity_model(predicted_vmod, min_velocity, max_velocity, save_path, save_name)
    else:
        pain_openfwi_seismic_data(seismic_data[2])
        minV = np.min(min_velocity + velocity_model * (max_velocity - min_velocity))
        maxV = np.max(min_velocity + velocity_model * (max_velocity - min_velocity))
        pain_openfwi_velocity_model(min_velocity + velocity_model * (max_velocity - min_velocity), minV, maxV, save_path, save_name+"_true")
        pain_openfwi_velocity_model(min_velocity + predicted_vmod * (max_velocity - min_velocity), minV, maxV, save_path, save_name)

def num_test(model_path, num_samples, seed=42, dataType = 'CurveVelA', save_name="test", model_type = "DDNet",save_path="") :

    print("Loading test model:{}".format(model_path))
    model_net, device, optimizer = determine_network(model_path, model_type=model_type)

    # net = PixelModel()
    # net.to(device)
    # net.to(device)
    # model = net.load_state_dict(torch.load(model_path))

    temp_vmod_path = None
    temp_seie_path = None

    num_data = len(np.load(temp_seie_path))
    random = Random(seed)
    idx_samples = random.sample(range(0, num_data), num_samples)
    samples = []
    for idx in idx_samples:
        data = torch.from_numpy(np.load(temp_seie_path)[idx, ...]).float()  # (5, 1000, 70)
        label = np.load(temp_vmod_path)[idx, ...]  # (1, 70, 70)
    label = torch.from_numpy(label).float()
    data = torch.unsqueeze(data, 0)
    label = torch.unsqueeze(label, 0)
    data, label = data.to("cuda:0"), label.to("cuda:0")
    output = model_net(data, model_dim)
    output = output.detach().cpu().squeeze()
    label = label.cpu().squeeze()
    samples.append((output, label))

    fig, axs = plt.subplots(2, num_samples, figsize=(num_samples * 5, 3 * 5))
    vmin_output = min(output.min() for output, _ in samples)
    vmax_output = max(output.max() for output, _ in samples)
    vmin_label = min(label.min() for _, label in samples)
    vmax_label = max(label.max() for _, label in samples)
    vmin = min(vmin_output, vmin_label)
    vmax = max(vmax_output, vmax_label)
    norm = matplotlib.colors.Normalize(vmin, vmax)
    mappable = matplotlib.cm.ScalarMappable(norm)

    for i, (output, label) in enumerate(samples):
        axs[0, i].imshow(output, norm=norm)
        axs[0, i].set_title(f"Prediction {i+1}", {"fontsize": 16})
        axs[1, i].imshow(label, norm=norm)
        axs[1, i].set_title(f"Label {i+1}", {"fontsize": 16})
    cb_ax = fig.add_axes([0.1, 0.06, 0.8, 0.02])
    fig.colorbar(mappable, cax=cb_ax, orientation="horizontal", shrink=0.6)
    fig.suptitle("InversionNet, OpenFWI", y=0.9, fontsize=20, fontweight=500)
    plt.show()
    pathName = save_path + save_name + '.png'
    plt.savefig(pathName)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DCNet model test")
    parser.add_argument("--batch-of-single", type=int, default=1, help="1=batch test, 0=single test")
    parser.add_argument("--model-type", type=str, default="DCNet", help="DCNet|DDNet|InversionNet|FCNVMB")
    parser.add_argument("--model-path", type=str, default=None, help="checkpoint path")
    parser.add_argument("--save-path", type=str, default=None, help="single test save path")
    parser.add_argument("--save-name", type=str, default="test", help="single test save name")
    parser.add_argument("--export-dir", type=str, default=None, help="export pred.npy and gt.npy")
    parser.add_argument("--benchmark-eval", type=str, default=None, help="path to benchmark_eval.py")
    parser.add_argument("--benchmark-eval-json", type=str, default=None, help="optional json output path")
    args = parser.parse_args()

    batch_of_single = args.batch_of_single
    # |DCNet|DDNet|InversionNet|SwinTransNet|
    model_type = args.model_type

    path = args.model_path
    savePath = args.save_path
    saveName = args.save_name

    if batch_of_single == 1:
        ##############
        # Batch test #
        ##############
        batch_test(
            path,
            model_type=model_type,
            export_dir=args.export_dir,
            benchmark_eval=args.benchmark_eval,
            benchmark_eval_json=args.benchmark_eval_json
        )
    else :
        ###############
        # Single test #
        ###############
        if dataset_name in ["SEGSalt", "SEGSimulation"]:
            # 1~10      :SEGSalt
            # 1601~1700 :SEGSimulation  1615
            select_id = 1615
        else:
            # [1~2, 0~499]
            select_id = [1, 224]
        single_test( path, select_id, model_type=model_type, save_path=savePath, save_name=saveName)
