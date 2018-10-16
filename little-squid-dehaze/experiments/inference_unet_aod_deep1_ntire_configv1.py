# inference config file
# Created by zyh in Meitu.

from squid.net import DnCnn
from squid.net import AODNet
from squid.net import DnCnn_AOD
from squid.net import Unet_AOD_Deep1_Net
import torch.nn as nn

test_snapshot_path = '/root/group-competition/zyh3/train_tasks/unet_aod_deep1_ntire_config/models/snapshot_40_G_model'

target_net = Unet_AOD_Deep1_Net()
target_net = nn.DataParallel(target_net).cuda()

test_input_dir = '/root/group-competition/data/dehaze/NTIRE/indoor/test/haze'

TEST_OUT_FOLDER = '/root/group-competition/zyh3/output/unet_aod_deep1_ntire_configv1'

GPU_ID = -1

vis = None

divided = True

psnr = True