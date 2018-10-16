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

pic_num = '44'
epoch_num = '100'
snapshot_paths = []
snapshot_path_0 = '/root/group-competition/zyh3/train_tasks/outside_dncnn_ntire_config_' + pic_num + '_0/models/snapshot_' + epoch_num + '_G_model'
snapshot_path_1 = '/root/group-competition/zyh3/train_tasks/outside_dncnn_ntire_config_' + pic_num + '_1/models/snapshot_' + epoch_num + '_G_model'
snapshot_path_2 = '/root/group-competition/zyh3/train_tasks/outside_dncnn_ntire_config_' + pic_num + '_2/models/snapshot_' + epoch_num + '_G_model'
snapshot_path_3 = '/root/group-competition/zyh3/train_tasks/outside_dncnn_ntire_config_' + pic_num + '_3/models/snapshot_' + epoch_num + '_G_model'
snapshot_path_4 = '/root/group-competition/zyh3/train_tasks/outside_dncnn_ntire_config_' + pic_num + '_4/models/snapshot_' + epoch_num + '_G_model'
snapshot_paths = [snapshot_path_0, snapshot_path_1, snapshot_path_2, snapshot_path_3, snapshot_path_4]

target_net = DnCnn(layer_num=20)
target_net = nn.DataParallel(target_net).cuda()

ff = open('/root/group-competition/zyh3/ntire/outdoor/test/test_bb_' + pic_num + '.txt')
dic = dict()
dic[0] = []
dic[1] = []
dic[2] = []
dic[3] = []
dic[4] = []
for line in ff:
	lline = line.strip()
	items = lline.split()
	k = int(items[1])
	dic[k].append(items[0])

test_input_dir = '/root/group-competition/zyh3/ntire/outdoor/test/' + pic_num + '_crop'

TEST_OUT_FOLDER = '/root/group-competition/zyh3/ntire/outdoor/test/' + pic_num + '_output'

ff.close()

GPU_ID = -1

vis = None

divided = False

psnr = False
# psnr = True