import sys
sys.path.append('./')

import argparse
import torch
from tools.imagedb import FaceDataset
from torchvision import transforms
from models.onet import ONet
from training.onet.trainer import ONetTrainer
from training.onet.config import Config
from tools.logger import Logger
from checkpoint import CheckPoint
import os
# import config

from pzmllog import NewLogger

if __name__ == '__main__':

    # Get config
    config = Config()
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)

    # jetML平台配置
    log = NewLogger(
        config={
            # 用户 ID
            'access_token': "uynbxeqzxwav488zjfykhe4j",
            # 项目 ID
            'project': "1434",
            # 实验描述和说明信息
            "description": "onet_train_test",
            # 自定义实验名称
            "experiment_name": "onet_train_test",
            # 仓库 ID
            "repository_id": "4625f512db944cff867fa3a0b3786d20",
            # tomcat的启动端口
            'port': "5560"
        },
        # 超参数集
        info={
            "learning_rate": config.lr,
            "epoch": config.nEpochs,
            "batch_size": config.batchSize
        }
    )

    # Set device
    # os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU  # 注释掉或移除这行
    use_cuda = False  # 直接设置为False，不使用GPU
    print(torch.cuda.is_available())  # 这行可以保留，但不会使用GPU
    # assert use_cuda  # 这行可以注释掉或移除，因为我们不使用GPU
    torch.manual_seed(config.manualSeed)  # 保留，用于设置CPU的随机种子
    # torch.cuda.manual_seed(config.manualSeed)  # 注释掉或移除这行
    device = torch.device("cpu")  # 直接设置为使用CPU
    print(device)
    # torch.backends.cudnn.benchmark = True  # 注释掉或移除这行，因为不使用GPU
    
    # Set dataloader
    kwargs = {'num_workers': config.nThreads, 'pin_memory': True} if use_cuda else {}
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    train_loader = torch.utils.data.DataLoader(
        FaceDataset(config.annoPath, transform=transform, is_train=True), batch_size=config.batchSize, shuffle=True, **kwargs)

    # Set model
    model = ONet()
    model = model.to(device)

    # Set checkpoint
    checkpoint = CheckPoint(config.save_path)

    # Set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.step, gamma=0.1)

    # Set trainer
    logger = Logger(config.save_path)
    trainer = ONetTrainer(config.lr, train_loader, model, optimizer, scheduler, logger, device)

    for epoch in range(1, config.nEpochs + 1):
        # 开始实验
        log.Run()

        trainer.train(epoch, log)
        checkpoint.save_model(model, index=epoch)

        # 源码有bug，只能提交运行目录下的模型文件
        # 出错函数 copy_file_to_dir
        # # 记录模型
        # log.Save([model_path])
        # 结束实验
        log.End()

    # 结束整个过程
    log.Submit()