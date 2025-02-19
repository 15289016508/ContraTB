# ContraTB
本项目旨在利用对比学习技术，提高二线结核药物耐药性检测的准确性。项目包含三个主要训练脚本：

train_attention.py: 用于普通学习的训练脚本。

train_sup.py: 用于对比学习的训练脚本。

train_joint.py: 用于联合学习的训练脚本，该脚本将加载预训练好的模型进行进一步训练。

文件结构
复制
项目名称/
│
├── train_attention.py       # 普通学习训练脚本
├── train_sup.py             # 对比学习训练脚本
├── train_joint.py           # 联合学习训练脚本
├── README.md                # 项目说明文件
└── requirements.txt         # 项目依赖文件
环境要求
Python 3.6+

PyTorch 1.7+

torchvision

numpy

pandas


普通学习训练
运行以下命令开始普通学习训练：

bash
复制
python train_attention.py
对比学习训练
运行以下命令开始对比学习训练：

bash
复制
python train_sup.py
联合学习训练
首先，确保已经通过上述任一训练脚本获得了预训练模型。然后运行以下命令开始联合学习训练：

bash
复制
python train_joint.py 
