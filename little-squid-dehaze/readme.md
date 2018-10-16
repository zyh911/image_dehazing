# little-squid 小乌贼框架

## 通用模块 

### loss

每个文件一种loss，每个loss之间是平等独立的，拷走即用。

### net

每个文件一种net结构，每个net结构之间是平等独立的，拷走即用。

### model

每个文件是一个model，它组合了各种loss，各种net，定义了拟合和预测等接口。每个model之间是平等独立的，拷走即用。

### data

每个文件一个dataset处理，每个dataset之间平等独立，拷走即用。
每种dataset的预处理和后处理(例如outmask的转换)都在单个文件中，例如：分割dataset，预处理数据集可直接调用 data/seg_data.py src_dir dst_dir 。
这样某个dataset相关的所有操作都在单个文件中。

### utils

可视化，log等基础模块。


## 使用:

### step0: 定义需要的模块

定义net，loss等模块；如果没有现成的，需要自己写。

### step1: 定义config文件

定义训练测试流程的迭代数，数据集路径，model的插拔，dataloader的插拔等。

注：有3类config文件

1. 文件名不带ntire和inference的config文件，如果要用需要大改。
2. 文件名带ntire，不带inference的config文件，是在ntire数据集上训练的config文件，可以直接跑，详情见step2.
3. 文件名带inference的config文件，是step3中使用的config文件，详情见step3.

### step2: 训练train(默认包含inference)——调用train.py

注：文件名含ntire的config文件默认是在/group-competition目录下跑，只需要改PERSONAL_TASKS变量就可以跑（不改也可以，不过结果会保存在/group-competition/zyh3目录下），INDOOR_OR_OUTDOOR_DIR变量是选择indoor数据或者outdoor数据的，简单选择下就可以用。

例如：
```
python -m squid.train experiments/unet_ntire_config.py
```

读取config，驱动 train, validate, test 这3种主流程，同时支持snapshot断点，用若干样本对模型效果进行监控等功能。


### (可选)step3: 推断inference——调用inference.py

注：文件名含inference的config文件，如果需要用，需要仿照已有的文件自己写。

例如：
```
python -m squid.inference experiments/inference_aod_ntire_config.py
```

读取config，执行inference流程，默认为debug模式(即会打印log)。
