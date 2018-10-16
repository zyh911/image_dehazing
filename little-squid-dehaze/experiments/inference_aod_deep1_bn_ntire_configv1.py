# inference config file
# Created by zyh in Meitu.

from squid.net import DnCnn
from squid.net import AODNet
from squid.net import DnCnn_AOD
from squid.net import AOD_Deep1_Bn_Net
import torch.nn as nn

test_snapshot_path = '/root/zyh3/train_tasks/aod_deep1_bn_ntire_config/models/snapshot_20_G_model'

target_net = AOD_Deep1_Bn_Net()
target_net = nn.DataParallel(target_net).cuda()

test_input_dir = '/root/group-competition/data/dehaze/NTIRE/indoor/IndoorValidationHazy'

TEST_OUT_FOLDER = '/root/group-competition/zyh3/output/aod_deep1_bn_ntire_configv1'

GPU_ID = -1

vis = None

divided = True

psnr = False