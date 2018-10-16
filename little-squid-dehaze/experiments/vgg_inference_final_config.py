# inference config file
# Created by zyh in Meitu.

from squid.net import DnCnn
from squid.net import AODNet
from squid.net import AOD_Deep1_Net
from squid.net import AOD_Deep1_New_Net
from squid.net import AOD_To_End_Deep1_Net
from squid.net import AOD_To_End_Deep1_Residual_Net
from squid.net import AOD_Deep3_Net
from squid.net import AOD_Deep2_Net
from squid.net import AOD_Deep1_Wide1_Net
from squid.net import AOD_Deep1_Wide2_Net
from squid.net import AOD_Deep_Wide_Net
from squid.net import DnCnn_AOD
from squid.net import EDSR_AOD_Net
from squid.net import Unet_AOD_Deep1_Net
from squid.net import Unet_AOD_Net
from squid.net import Unet_Residual_Net
from squid.net import Unet_AOD_Deep_Wide_Net
from squid.net import Unet_AOD_Thin_Net
import torch.nn as nn

test_snapshot_path = '/root/group-competition/zyh3/train_tasks/vgg_unet_aod_deep1_ntire_config/models/snapshot_10_G_model'

target_net = Unet_AOD_Deep1_Net()
test_input_dir = '/root/group-competition/data/dehaze/NTIRE/indoor/IndoorTestHazy'
# test_input_dir = '/root/group-competition/data/dehaze/NTIRE/indoor/test/haze'

TEST_OUT_FOLDER = '/root/group-competition/zyh3/indoor_final'

GPU_ID = 0

vis = None

divided = True

psnr = False
# psnr = True