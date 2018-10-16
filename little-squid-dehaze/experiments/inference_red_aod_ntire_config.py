# inference config file
# Created by zyh in Meitu.

from squid.net import DnCnn
from squid.net import AODNet
from squid.net import DnCnn_AOD
from squid.net import RED_AOD_Net
import torch.nn as nn

test_snapshot_path = '/root/zyh3/train_tasks/red_aod_ntire_config/models/snapshot_7_G_model'

target_net = RED_AOD_Net()
target_net = nn.DataParallel(target_net).cuda()

test_input_dir = '/root/zyh3/ntire/indoortest'

TEST_OUT_FOLDER = '/root/zyh3/ntire/indoortest_red_aod_out'

GPU_ID = -1

vis = None

divided = False