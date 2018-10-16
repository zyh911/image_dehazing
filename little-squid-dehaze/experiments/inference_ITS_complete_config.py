# inference config file
# Created by zyh in Meitu.

from squid.net import DnCnn
from squid.net import AODNet
from squid.net import DnCnn_AOD

test_snapshot_path = '/root/zyh3/train_tasks/dncnn_configv5/models/snapshot_12_G_model'

# target_net = AODNet()
target_net = DnCnn(layer_num=20)
# target_net = DnCnn_AOD()

test_input_dir = '/root/zyh3/ITS/ITS/val/haze'
# test_input_dir = '/root/zyh3/IndoorTrain/IndoorTrainHazy'
TEST_OUT_FOLDER = '/root/zyh3/ITS_val_complete_dncnn_20layer_out'
# TEST_OUT_FOLDER = '/root/zyh3/IndoorTrain/IndoorTrainHazy_out'
GPU_ID = 0

vis = None