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

pic_num = '40'
epoch_num = '10'
snapshot_paths = []
snapshot_path_0 = '/root/group-competition/zyh3/train_tasks/outside_dncnn_ntire_config_mixed_0/models/snapshot_' + epoch_num + '_G_model'
snapshot_path_1 = '/root/group-competition/zyh3/train_tasks/outside_dncnn_ntire_config_mixed_1/models/snapshot_' + epoch_num + '_G_model'
snapshot_path_2 = '/root/group-competition/zyh3/train_tasks/outside_dncnn_ntire_config_mixed_2/models/snapshot_' + epoch_num + '_G_model'
snapshot_path_3 = '/root/group-competition/zyh3/train_tasks/outside_dncnn_ntire_config_mixed_3/models/snapshot_' + epoch_num + '_G_model'
snapshot_path_4 = '/root/group-competition/zyh3/train_tasks/outside_dncnn_ntire_config_mixed_4/models/snapshot_' + epoch_num + '_G_model'
snapshot_path_5 = '/root/group-competition/zyh3/train_tasks/outside_dncnn_ntire_config_mixed_5/models/snapshot_' + epoch_num + '_G_model'
snapshot_path_6 = '/root/group-competition/zyh3/train_tasks/outside_dncnn_ntire_config_mixed_6/models/snapshot_' + epoch_num + '_G_model'
snapshot_path_7 = '/root/group-competition/zyh3/train_tasks/outside_dncnn_ntire_config_mixed_7/models/snapshot_' + epoch_num + '_G_model'
snapshot_path_8 = '/root/group-competition/zyh3/train_tasks/outside_dncnn_ntire_config_mixed_8/models/snapshot_' + epoch_num + '_G_model'
snapshot_path_9 = '/root/group-competition/zyh3/train_tasks/outside_dncnn_ntire_config_mixed_9/models/snapshot_' + epoch_num + '_G_model'
snapshot_path_10 = '/root/group-competition/zyh3/train_tasks/outside_dncnn_ntire_config_mixed_10/models/snapshot_' + epoch_num + '_G_model'
snapshot_path_11 = '/root/group-competition/zyh3/train_tasks/outside_dncnn_ntire_config_mixed_11/models/snapshot_' + epoch_num + '_G_model'
snapshot_path_12 = '/root/group-competition/zyh3/train_tasks/outside_dncnn_ntire_config_mixed_12/models/snapshot_' + epoch_num + '_G_model'
snapshot_path_13 = '/root/group-competition/zyh3/train_tasks/outside_dncnn_ntire_config_mixed_13/models/snapshot_' + epoch_num + '_G_model'
snapshot_path_14 = '/root/group-competition/zyh3/train_tasks/outside_dncnn_ntire_config_mixed_14/models/snapshot_' + epoch_num + '_G_model'
snapshot_path_15 = '/root/group-competition/zyh3/train_tasks/outside_dncnn_ntire_config_mixed_15/models/snapshot_' + epoch_num + '_G_model'
snapshot_path_16 = '/root/group-competition/zyh3/train_tasks/outside_dncnn_ntire_config_mixed_16/models/snapshot_' + epoch_num + '_G_model'
snapshot_path_17 = '/root/group-competition/zyh3/train_tasks/outside_dncnn_ntire_config_mixed_17/models/snapshot_' + epoch_num + '_G_model'
snapshot_path_18 = '/root/group-competition/zyh3/train_tasks/outside_dncnn_ntire_config_mixed_18/models/snapshot_' + epoch_num + '_G_model'
snapshot_path_19 = '/root/group-competition/zyh3/train_tasks/outside_dncnn_ntire_config_mixed_19/models/snapshot_' + epoch_num + '_G_model'
snapshot_paths = [snapshot_path_0, snapshot_path_1, snapshot_path_2, snapshot_path_3, snapshot_path_4, snapshot_path_5, snapshot_path_6, snapshot_path_7, snapshot_path_8, snapshot_path_9, snapshot_path_10, snapshot_path_11, snapshot_path_12, snapshot_path_13, snapshot_path_14, snapshot_path_15, snapshot_path_16, snapshot_path_17, snapshot_path_18, snapshot_path_19]

target_net = DnCnn(layer_num=20)
target_net = nn.DataParallel(target_net).cuda()

ff = open('/root/group-competition/zyh3/ntire/outdoor/test/test_bb_' + pic_num + '_mixed_20.txt')
dic = dict()
dic[0] = []
dic[1] = []
dic[2] = []
dic[3] = []
dic[4] = []
dic[5] = []
dic[6] = []
dic[7] = []
dic[8] = []
dic[9] = []
dic[10] = []
dic[11] = []
dic[12] = []
dic[13] = []
dic[14] = []
dic[15] = []
dic[16] = []
dic[17] = []
dic[18] = []
dic[19] = []
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