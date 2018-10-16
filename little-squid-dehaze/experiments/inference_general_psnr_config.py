# inference config file
# Created by zyh in Meitu.

from squid.net import DnCnn
from squid.net import AODNet
from squid.net import AOD_Deep1_Net
from squid.net import AOD_Deep_Wide_Net
from squid.net import DnCnn_AOD
from squid.net import EDSR_AOD_Net
from squid.net import Unet_AOD_Deep1_Net
from squid.net import Unet_AOD_Net
from squid.net import Unet_AOD_Deep_Wide_Net
import torch.nn as nn

test_snapshot_path = '/root/group-competition/zyh3/train_tasks/small_aod_deep_wide_ntire_config/models/snapshot_1_G_model'

target_net = AOD_Deep_Wide_Net()
target_net = nn.DataParallel(target_net).cuda()

# test_input_dir = '/root/group-competition/data/dehaze/NTIRE/indoor/IndoorValidationHazy'
test_input_dir = '/root/group-competition/data/dehaze/NTIRE/indoor/test/haze'

TEST_OUT_FOLDER = '/root/group-competition/zyh3/output/inference_general_psnr_config'

GPU_ID = -1

vis = None

divided = True

# psnr = False
psnr = True