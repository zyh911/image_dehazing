# inference config file
# Created by zyh in Meitu.

from squid.net import DnCnn
from squid.net import AODNet
from squid.net import AOD_Deep1_Net
from squid.net import DnCnn_AOD
import torch.nn as nn

test_snapshot_path = '/root/zyh3/train_tasks/aod_deep1_ntire_config/models/snapshot_20_G_model'

target_net = AOD_Deep1_Net()
target_net = nn.DataParallel(target_net).cuda()

test_input_dir = '/root/zyh3/ntire/indoortest'

TEST_OUT_FOLDER = '/root/zyh3/ntire/indoortest_aod_deep1_out_v1'

GPU_ID = -1

vis = None