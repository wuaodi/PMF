"""
PMF配置选项 - H200单卡版本
"""
import os 
import yaml
import sys 
import shutil

sys.path.insert(0, "../../")


class Option(object):
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = yaml.safe_load(open(config_path, "r"))

        # 通用选项
        self.save_path = self.config["save_path"]
        self.seed = self.config["seed"]
        self.gpu = self.config["gpu"]
        self.n_gpus = 1  # 单卡
        
        self.print_frequency = self.config["print_frequency"]
        self.n_threads = self.config["n_threads"]
        self.experiment_id = self.config["experiment_id"]
       
        # 数据配置
        self.dataset = self.config["dataset"]
        self.nclasses = self.config["nclasses"]
        self.data_root = self.config["data_root"]
        self.data_config = self.config["data_config"]
        self.has_label = self.config["has_label"]
        
        # 训练配置
        self.n_epochs = self.config["n_epochs"]
        self.batch_size = self.config["batch_size"]
        self.lr = self.config["lr"]
        self.warmup_epochs = self.config["warmup_epochs"]
        self.momentum = self.config["momentum"]
        self.weight_decay = self.config["weight_decay"]
        self.val_only = self.config["val_only"]
        self.is_debug = self.config["is_debug"]
        self.val_frequency = self.config["val_frequency"]

        # 模型选项
        self.lambda_ = self.config["lambda"]
        self.gamma = self.config["gamma"]
        self.tau = self.config["tau"]
        self.img_backbone = self.config["img_backbone"]
        self.base_channels = self.config["base_channels"]
        self.imagenet_pretrained = self.config["imagenet_pretrained"]

        # checkpoint
        self.checkpoint = self.config["checkpoint"]
        self.pretrained_model = self.config["pretrained_model"]

        self._prepare()

    def _prepare(self):
        batch_size = self.batch_size[0]
        self.save_path = os.path.join(
            self.save_path, 
            f"log_{self.dataset}_PMFNet-{self.img_backbone}_bs{batch_size}-lr{self.lr}_{self.experiment_id}"
        )

    def check_path(self):
        if os.path.exists(self.save_path):
            print(f"file exist: {self.save_path}")
            action = input("Select Action: d(delete) / q(quit): ").lower().strip()
            if action == "d":
                shutil.rmtree(self.save_path)
            else:
                raise OSError(f"Directory exits: {self.save_path}")
        
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)
