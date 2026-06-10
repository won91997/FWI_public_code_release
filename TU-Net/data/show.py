import matplotlib.pylab as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
mpl.use('TkAgg')
import numpy as np
import matplotlib.ticker as ticker
import cv2


font21 = {
    'family': 'Times New Roman',
    'weight': 'normal',
    'size': 21,
}


font18 = {
    'family': 'Times New Roman',
    'weight': 'normal',
    'size': 18,
}


def pain_seg_seismic_data(para_seismic_data, output_filename):
    """
    Plotting seismic data images of SEG salt datasets
    :param para_seismic_data:  Seismic data (400 x 301) (numpy)
    :param is_colorbar: Whether to add a color bar (1 means add, 0 is the default, means don't add)
    """
    fig, ax = plt.subplots(figsize=(6.2, 8.1), dpi = 120)

    im = ax.imshow(para_seismic_data, extent=[0, 300, 400, 0], cmap=plt.cm.seismic, vmin=-0.4, vmax=0.44)

    ax.set_xlabel('Position (km)', font21)
    ax.set_ylabel('Time (s)', font21)
    ax.set_xticks(np.linspace(0, 300, 5))
    ax.set_yticks(np.linspace(0, 400, 5))
    ax.set_xticklabels(labels = [0,0.75,1.5,2.25,3.0], size=16)
    ax.set_yticklabels(labels = [0.0,0.50,1.00,1.50,2.00], size=16)

    plt.rcParams['font.size'] = 14      # Set colorbar font size
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="3%", pad=0.3)
    plt.colorbar(im, ax=ax, cax=cax, orientation='horizontal')
    plt.subplots_adjust(bottom=0.08, top=0.98, left=0.11, right=0.99)
    plt.savefig(output_filename)
    plt.show()


def pain_openfwi_seismic_data(para_seismic_data, output_filename):
    """
    Plotting seismic data images of openfwi dataset
    :param para_seismic_data:   Seismic data (1000 x 70) (numpy)
    """
    data = cv2.resize(para_seismic_data, dsize=(400, 301), interpolation=cv2.INTER_CUBIC)
    fig, ax = plt.subplots(figsize=(6.2, 8.1), dpi = 120)
    im = ax.imshow(data, extent=[0, 0.7, 1.0, 0], cmap=plt.cm.seismic, vmin=-18, vmax=19)

    ax.set_xlabel('Position (km)', font21)
    ax.set_ylabel('Time (s)', font21)
    ax.set_xticks(np.linspace(0, 0.7, 5))
    ax.set_yticks(np.linspace(0, 1.0, 5))
    ax.set_xticklabels(labels=[0, 0.17, 0.35, 0.52, 0.7], size=16)
    ax.set_yticklabels(labels=[0, 0.25, 0.5, 0.75, 1.0], size=16)

    plt.rcParams['font.size'] = 14      # Set colorbar font size
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="3%", pad=0.3)
    plt.colorbar(im, ax=ax, cax=cax, orientation='horizontal')
    plt.subplots_adjust(bottom=0.08, top=0.98, left=0.11, right=0.99)
    plt.savefig(output_filename)  # 保存图像
    plt.show()
    plt.close()


def plot_velocity(num, output, target, test_result_dir, vmin, vmax):
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    im = ax[0].matshow(output, cmap='viridis', vmin=vmin, vmax=vmax)
    ax[0].set_title('Prediction', y=1.08)
    ax[1].matshow(target, cmap='viridis', vmin=vmin, vmax=vmax)
    ax[1].set_title('Ground Truth', y=1.08)

    for axis in ax:
        axis.set_xticks(range(0, 70, 10))
        axis.set_xticklabels(range(0, 700, 100))
        axis.set_yticks(range(0, 70, 10))
        axis.set_yticklabels(range(0, 700, 100))
        axis.set_ylabel('Depth (m)', fontsize=12)
        axis.set_xlabel('Offset (m)', fontsize=12)

    fig.colorbar(im, ax=ax, shrink=0.75, label='Velocity(m/s)')
    plt.savefig(test_result_dir + 'PD_GT' + str(num) + '.png')
    plt.close('all')


def plot_ground_truth(num, target, test_result_dir, vmin, vmax):
    fig, ax = plt.subplots(figsize=(5.8, 6), dpi=150)
    im = ax.imshow(target, extent=[0, 0.7, 0.7, 0], vmin=vmin, vmax=vmax)
    ax.set_xlabel('Position (km)', font18)
    ax.set_ylabel('Depth (km)', font18)
    ax.set_xticks(np.linspace(0, 0.7, 8))
    ax.set_yticks(np.linspace(0, 0.7, 8))
    ax.set_xticklabels(labels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], size=18)
    ax.set_yticklabels(labels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], size=18)
    plt.rcParams['font.size'] = 14      # Set colorbar font size
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="3%", pad=0.4)
    cb1 = plt.colorbar(im, ax=ax, cax=cax, orientation='horizontal', format=mpl.ticker.StrMethodFormatter('{x:.0f}'))
    tick_locator = ticker.MaxNLocator(nbins=9)
    cb1.locator = tick_locator
    cb1.set_ticks([np.min(vmin), 0.2*(vmax-vmin)+vmin, 0.4*(vmax-vmin)+vmin,
                   0.6*(vmax-vmin)+vmin, 0.8*(vmax-vmin)+vmin, np.max(vmax)])
    plt.subplots_adjust(bottom=0.11, top=0.97, left=0.12, right=0.97)
    plt.savefig(test_result_dir + 'GT' + str(num) + '.png')
    plt.close(fig)


def plot_seg_velocity_compare(num, output, target, test_result_dir, vmin, vmax):
    fig, ax = plt.subplots(1, 2, figsize=(11,5))
    im = ax[0].matshow(output, cmap='viridis', vmin=vmin, vmax=vmax)
    ax[0].set_title('Prediction', y=1.15)
    ax[1].matshow(target, cmap='viridis', vmin=vmin, vmax=vmax)
    ax[1].set_title('Ground Truth', y=1.15)

    for axis in ax:

        axis.set_xticks(range(0, 301, 50))
        axis.set_xticklabels([0.0,0.5,1.0,1.5,2.0,2.5,3.0],size=8)
        axis.set_yticks(range(0, 201, 25))
        axis.set_yticklabels([0.00,0.25,0.50,0.75,1.00,1.25,1.50,1.75,2.00],fontsize=8)

        axis.set_ylabel('Depth (km)', fontsize=11)
        axis.set_xlabel('Position (km)', fontsize=11)

    fig.colorbar(im, ax=ax, shrink=0.55, label='Velocity(m/s)')
    plt.savefig(test_result_dir + 'PD_GT' + str(num))  # 设置保存名字
    plt.close('all')


def plot_seg_prediction_velocity(num, output, test_result_dir, vmin, vmax):
    fig, ax = plt.subplots(figsize=(7.1, 5.2), dpi=150)
    im = ax.matshow(output, cmap='viridis', vmin=vmin, vmax=vmax)
    ax.set_xticks(range(0, 301, 50))
    ax.set_yticks(range(0, 201, 25))
    ax.set_xticklabels(labels=[0.0,0.5,1.0,1.5,2.0,2.5,3.0], size=16)
    ax.set_yticklabels(labels=[0.00,0.25,0.50,0.75,1.00,1.25,1.50,1.75,2.00], size=16)
    ax.set_xlabel('Position (km)', size=16)
    ax.set_ylabel('Depth (km)', size=16)
    plt.rcParams['font.size'] = 14
    ax.tick_params(axis='x', bottom=True, top=False, labelbottom=True, labeltop=False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="3%", pad=0.4)
    cb1 = plt.colorbar(im, ax=ax, cax=cax, orientation='horizontal', format=mpl.ticker.StrMethodFormatter('{x:.0f}'))
    tick_locator = ticker.MaxNLocator(nbins=9)
    cb1.locator = tick_locator
    cb1.set_ticks([np.min(vmin), 0.2*(vmax-vmin)+vmin, 0.4*(vmax-vmin)+vmin,
                   0.6*(vmax-vmin)+vmin, 0.8*(vmax-vmin)+vmin, np.max(vmax)])
    plt.subplots_adjust(bottom=0.11, top=0.97, left=0.12, right=0.97)
    plt.savefig(test_result_dir + 'PD' + str(num))  # 设置保存名字
    plt.close('all')


def plot_seg_truth_velocity(num, output, test_result_dir, vmin, vmax):
    fig, ax = plt.subplots(figsize=(7.1, 5.2), dpi=150)
    im = ax.matshow(output, cmap='viridis', vmin=vmin, vmax=vmax)
    ax.set_xticks(range(0, 301, 50))
    ax.set_yticks(range(0, 201, 25))
    ax.set_xticklabels(labels=[0.0,0.5,1.0,1.5,2.0,2.5,3.0], size=16)
    ax.set_yticklabels(labels=[0.00,0.25,0.50,0.75,1.00,1.25,1.50,1.75,2.00], size=16)
    ax.set_xlabel('Position (km)', size=16)
    ax.set_ylabel('Depth (km)', size=16)
    plt.rcParams['font.size'] = 14
    ax.tick_params(axis='x', bottom=True, top=False, labelbottom=True, labeltop=False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="3%", pad=0.4)
    cb1 = plt.colorbar(im, ax=ax, cax=cax, orientation='horizontal', format=mpl.ticker.StrMethodFormatter('{x:.0f}'))
    tick_locator = ticker.MaxNLocator(nbins=9)
    cb1.locator = tick_locator
    cb1.set_ticks([np.min(vmin), 0.2*(vmax-vmin)+vmin, 0.4*(vmax-vmin)+vmin,
                   0.6*(vmax-vmin)+vmin, 0.8*(vmax-vmin)+vmin, np.max(vmax)])
    plt.subplots_adjust(bottom=0.11, top=0.97, left=0.12, right=0.97)
    plt.savefig(test_result_dir + 'GT' + str(num))  # 设置保存名字
    plt.close('all')


def plot_prediction(num, output, test_result_dir, vmin, vmax):
    fig, ax = plt.subplots(figsize=(5.8, 6), dpi=150)
    im = ax.imshow(output, extent=[0, 0.7, 0.7, 0], vmin=vmin, vmax=vmax)
    ax.set_xlabel('Position (km)', font18)
    ax.set_ylabel('Depth (km)', font18)
    ax.set_xticks(np.linspace(0, 0.7, 8))
    ax.set_yticks(np.linspace(0, 0.7, 8))
    ax.set_xticklabels(labels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], size=18)
    ax.set_yticklabels(labels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], size=18)
    plt.rcParams['font.size'] = 14      # Set colorbar font size
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="3%", pad=0.4)
    cb1 = plt.colorbar(im, ax=ax, cax=cax, orientation='horizontal', format=mpl.ticker.StrMethodFormatter('{x:.0f}'))
    tick_locator = ticker.MaxNLocator(nbins=9)
    cb1.locator = tick_locator
    cb1.set_ticks([np.min(vmin), 0.2*(vmax-vmin)+vmin, 0.4*(vmax-vmin)+vmin,
                   0.6*(vmax-vmin)+vmin, 0.8*(vmax-vmin)+vmin, np.max(vmax)])
    plt.subplots_adjust(bottom=0.11, top=0.97, left=0.12, right=0.97)
    plt.savefig(test_result_dir + 'PD' + str(num) + '.png')
    plt.close(fig)


def plot_velocity_image(num, output1, output2, output3, output4, target, test_result_dir, vmin, vmax):
    fig = plt.figure(figsize=(8.5, 7.8))
    column_index = 35
    pixel_values1, pixel_values2, pixel_values3, pixel_values4, pixel_values5 = [], [], [], [], []
    for y in range(output1.shape[0]):
        pixel_value1 = output1[y, column_index]
        pixel_value2 = output2[y, column_index]
        pixel_value3 = output3[y, column_index]
        pixel_value4 = output4[y, column_index]
        pixel_value5 = target[y, column_index]
        pixel_values1.append(pixel_value1)
        pixel_values2.append(pixel_value2)
        pixel_values3.append(pixel_value3)
        pixel_values4.append(pixel_value4)
        pixel_values5.append(pixel_value5)
    plt.plot(pixel_values5, color='orange', linewidth=2, label='Ground Truth')
    plt.plot(pixel_values4, color='purple', linewidth=2, label='VelocityGAN')
    plt.plot(pixel_values3, color='green', linewidth=2, label='DD-Net70')
    plt.plot(pixel_values2, color='blue', linewidth=2, label='InversionNet')
    plt.plot(pixel_values1, color='red', linewidth=2, label='TU-Net')

    plt.legend(fontsize=18)
    plt.rcParams['font.size'] = 14
    plt.xticks(range(0, 80, 10), fontsize=18)
    ticks = [np.min(vmin), 0.2*(vmax-vmin)+vmin, 0.4*(vmax-vmin)+vmin,
                0.6*(vmax-vmin)+vmin, 0.8*(vmax-vmin)+vmin, np.max(vmax)]
    tick_labels = [int(tick) for tick in ticks]
    plt.yticks(ticks, tick_labels, fontsize=18)
    # plt.yticks([np.min(vmin), 0.2*(vmax-vmin)+vmin, 0.4*(vmax-vmin)+vmin,
    #             0.6*(vmax-vmin)+vmin, 0.8*(vmax-vmin)+vmin, np.max(vmax)], fontsize=18)
    plt.xlabel('Depth (m)', font18)
    plt.ylabel('Velocity (m/s)', font18)
    plt.subplots_adjust(bottom=0.10, top=0.97, left=0.13, right=0.97)
    plt.savefig(test_result_dir + 'PDD' + str(num) + '.png')
    plt.close(fig)


def plot_velocity_image_seg(num, output1, output2, output3, target, test_result_dir, vmin, vmax):
    fig = plt.figure(figsize=(8.5, 7.8))
    column_index = 100
    pixel_values1, pixel_values2, pixel_values3, pixel_values4 = [], [], [], []
    for y in range(output1.shape[0]):
        pixel_value1 = output1[y, column_index]
        pixel_value2 = output2[y, column_index]
        pixel_value3 = output3[y, column_index]
        pixel_value4 = target[y, column_index]
        pixel_values1.append(pixel_value1)
        pixel_values2.append(pixel_value2)
        pixel_values3.append(pixel_value3)
        pixel_values4.append(pixel_value4)
    plt.plot(pixel_values4, color='orange', linewidth=2, label='Ground Truth')
    plt.plot(pixel_values2, color='blue', linewidth=2, label='FCNVMB')
    plt.plot(pixel_values3, color='green', linewidth=2, label='DD-Net')
    plt.plot(pixel_values1, color='red', linewidth=2, label='TU-Net')

    plt.legend(fontsize=18)
    plt.rcParams['font.size'] = 14
    plt.xticks(range(0, 201, 25), fontsize=18)
    ticks = [np.min(vmin), 0.2*(vmax-vmin)+vmin, 0.4*(vmax-vmin)+vmin,
                0.6*(vmax-vmin)+vmin, 0.8*(vmax-vmin)+vmin, np.max(vmax)]
    tick_labels = [int(tick) for tick in ticks]
    plt.yticks(ticks, tick_labels, fontsize=18)
    # plt.yticks([np.min(vmin), 0.2*(vmax-vmin)+vmin, 0.4*(vmax-vmin)+vmin,
    #             0.6*(vmax-vmin)+vmin, 0.8*(vmax-vmin)+vmin, np.max(vmax)], fontsize=18)
    plt.xlabel('Depth (m)', font18)
    plt.ylabel('Velocity (m/s)', font18)
    plt.subplots_adjust(bottom=0.10, top=0.97, left=0.13, right=0.97)
    plt.savefig(test_result_dir + 'PDD' + str(num) + '.png')
    plt.close(fig)


def pain_openfwi_velocity_model(para_velocity_model):
    """
    Plotting seismic data images of openfwi dataset
    :param para_velocity_model: Velocity model (70 x 70) (numpy)
    :return:
    """
    fig, ax = plt.subplots(figsize=(5.8, 6), dpi=150)
    im = ax.imshow(para_velocity_model, extent=[0, 0.7, 0.7, 0])

    ax.set_xlabel('Position (km)', font18)
    ax.set_ylabel('Depth (km)', font18)

    ax.set_xticks(np.linspace(0, 0.7, 8))
    ax.set_yticks(np.linspace(0, 0.7, 8))
    ax.set_xticklabels(labels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], size=18)
    ax.set_yticklabels(labels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], size=18)

    plt.rcParams['font.size'] = 14      # Set colorbar font size
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="3%", pad=0.35)
    plt.colorbar(im, ax=ax, cax=cax, orientation='horizontal', format = mpl.ticker.StrMethodFormatter('{x:.0f}'))
    plt.subplots_adjust(bottom=0.10, top=0.95, left=0.13, right=0.95)
    plt.show()


def pain_seg_velocity_model(para_velocity_model):
    """
    :param para_velocity_model: Velocity model (200 x 301) (numpy)
    :param min_velocity: Upper limit of velocity in the velocity model
    :param max_velocity: Lower limit of velocity in the velocity model
    :return:
    """
    fig, ax = plt.subplots(figsize=(5.8, 4.3), dpi=150)
    im = ax.imshow(para_velocity_model, extent=[0, 3, 2, 0])

    ax.set_xlabel('Position (km)', font18)
    ax.set_ylabel('Depth (km)', font18)
    ax.tick_params(labelsize=14)

    plt.rcParams['font.size'] = 14  # Set colorbar font size
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="3%", pad=0.32)
    plt.colorbar(im, ax=ax, cax=cax, orientation='horizontal')
    plt.subplots_adjust(bottom=0.12, top=0.95, left=0.11, right=0.99)

    plt.show()


def PlotComparison_openfwi_velocity_model(para_raw_data, para_genenrate_data):
    """
        Compare raw seismic data (ground truth) and genenrated ones (prediction) of openfwi vmodel.
        :param para_raw_data: ground truth (70 x 70) (numpy)
        :param para_genenrate_data: prediction vmodel (70 x 70) (numpy)
        :return:
    """
    vmin = np.min(para_raw_data)
    vmax = np.max(para_raw_data)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    im1 = ax1.imshow(para_raw_data, vmin=vmin, vmax=vmax, extent=[0, 3, 2, 0])
    ax1.set_xlabel('Position (km)', font18)
    ax1.set_ylabel('Depth (km)', font18)
    ax1.tick_params(labelsize=14)

    plt.rcParams['font.size'] = 14  # Set colorbar font size
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("top", size="3%", pad=0.32)
    plt.colorbar(im1, ax=ax1, cax=cax, orientation='horizontal')

    im2 = ax2.imshow(para_genenrate_data, vmin=vmin, vmax=vmax, extent=[0, 3, 2, 0])
    ax2.set_xlabel('Position (km)', font18)
    ax2.set_ylabel('Depth (km)', font18)
    ax2.tick_params(labelsize=14)

    plt.rcParams['font.size'] = 14  # Set colorbar font size
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("top", size="3%", pad=0.32)
    plt.colorbar(im2, ax=ax2, cax=cax, orientation='horizontal')

    plt.subplots_adjust(bottom=0.12, top=0.95, left=0.11, right=0.99)
    plt.show()


def PlotComparison_openfwi_seismic_data(para_raw_data, para_genenrate_data):
    """
        Compare raw seismic data images and genenrated ones of openfwi dataset
        :param para_raw_data: 15 Hz Velocity model (1000 x 70) (numpy)
        :param para_genenrate_data: different observation systems, different main frequencies Velocity model (1000 x 70) (numpy)
        :return:
    """
    data = cv2.resize(para_raw_data, dsize=(400, 301), interpolation=cv2.INTER_CUBIC)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 16), dpi=120)
    im = ax1.imshow(data, extent=[0, 0.7, 1.0, 0], cmap=plt.cm.seismic, vmin=-18, vmax=19)

    ax1.set_xlabel('Position (km)', font21)
    ax1.set_ylabel('Time (s)', font21)
    ax1.set_xticks(np.linspace(0, 0.7, 5))
    ax1.set_yticks(np.linspace(0, 1.0, 5))
    ax1.set_xticklabels(labels=[0, 0.17, 0.35, 0.52, 0.7], size=21)
    ax1.set_yticklabels(labels=[0, 0.25, 0.5, 0.75, 1.0], size=21)

    plt.rcParams['font.size'] = 14  # Set colorbar font size
    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes("top", size="3%", pad=0.3)
    plt.colorbar(im, ax=ax1, cax=cax1, orientation='horizontal')
    plt.subplots_adjust(bottom=0.08, top=0.98, left=0.11, right=0.99)

    generate_data = cv2.resize(para_genenrate_data, dsize=(400, 301), interpolation=cv2.INTER_CUBIC)
    im = ax2.imshow(generate_data, extent=[0, 0.7, 1.0, 0], cmap=plt.cm.seismic, vmin=-18, vmax=19)

    ax2.set_xlabel('Position (km)', font21)
    ax2.set_ylabel('Time (s)', font21)
    ax2.set_xticks(np.linspace(0, 0.7, 5))
    ax2.set_yticks(np.linspace(0, 1.0, 5))
    ax2.set_xticklabels(labels=[0, 0.17, 0.35, 0.52, 0.7], size=21)
    ax2.set_yticklabels(labels=[0, 0.25, 0.5, 0.75, 1.0], size=21)

    plt.rcParams['font.size'] = 14  # Set colorbar font size
    divider = make_axes_locatable(ax2)
    cax2 = divider.append_axes("top", size="3%", pad=0.3)
    plt.colorbar(im, ax=ax2, cax=cax2, orientation='horizontal')
    plt.subplots_adjust(bottom=0.08, top=0.98, left=0.11, right=0.99)

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    seismic_flatfaultA = np.load("G:/Data/OpenFWI/FlatFaultA/test_data/seismic/seismic1.npy")
    # seismic_curvefaultA = np.load("G:/Data/OpenFWI/CurveFaultA/test_data/seismic/seismic1.npy")
    # seismic_curvevelA = np.load("G:/Data/OpenFWI/CurveVelA/test_data/seismic/seismic1.npy")
    # seismic_flatvelA = np.load("G:/Data/OpenFWI/FlatVelA/test_data/seismic/seismic1.npy")

    # seismic_SEGSimulation = scipy.io.loadmat("E:/Data/SEGSimulateData/train_data/georec/georec1371.mat")["Rec"]
    # seismic_SEGSalt = scipy.io.loadmat("E:/Data/SEGSaltData/train_data/georec/georec126.mat")["Rec"]

    # pain_openfwi_seismic_data(seismic_flatflautA[3, 2, :, :])
    # pain_openfwi_seismic_data(seismic_flatflautA[10, 2, :, :])

    # pain_openfwi_seismic_data(seismic_curvevelA[18, 2, :, :], output_filename='Seismic_Data_CurveVelA.png')
    # pain_openfwi_seismic_data(seismic_curvefaultA[3, 2, :, :], output_filename='Seismic_Data_CurveFaultA.png')
    pain_openfwi_seismic_data(seismic_flatfaultA[28, 2, :, :], output_filename='5.png')
    # pain_openfwi_seismic_data(seismic_flatvelA[4, 2, :, :], output_filename='Seismic_Data_FlatVelA.png')

    # pain_seg_seismic_data(seismic_SEGSimulation[:, :, 15], output_filename='Seismic_Data_SEGSimulation.png')
    # pain_seg_seismic_data(seismic_SEGSalt[:, :, 15], output_filename='Seismic_Data_SEGSalt.png')
