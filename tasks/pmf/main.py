"""
PMF训练入口 - H200单卡版本 (PyTorch 2.1+)
"""
import argparse
import datetime
from option import Option
import os
import torch
import time
import trainer
import pc_processor


class Experiment(object):
    def __init__(self, settings: Option):
        self.settings = settings
        
        # 设置GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = self.settings.gpu
        torch.cuda.set_device(0)
        
        # 设置随机种子
        torch.manual_seed(self.settings.seed)
        torch.cuda.manual_seed(self.settings.seed)
        torch.backends.cudnn.benchmark = True
        
        # 初始化记录器
        self.recorder = pc_processor.checkpoint.Recorder(
            self.settings, self.settings.save_path)
        
        self.epoch_start = 0
        
        # 初始化模型
        self.model = pc_processor.models.PMFNet(
            pcd_channels=5,
            img_channels=3,
            nclasses=self.settings.nclasses,
            base_channels=self.settings.base_channels,
            image_backbone=self.settings.img_backbone,
            imagenet_pretrained=self.settings.imagenet_pretrained
        )
        
        # 初始化trainer
        self.trainer = trainer.Trainer(self.settings, self.model, self.recorder)
        
        # 加载checkpoint
        self._loadCheckpoint()

    def _loadCheckpoint(self):
        if self.settings.pretrained_model is not None and self.settings.checkpoint is not None:
            raise ValueError("不能同时使用pretrained_model和checkpoint")
            
        if self.settings.pretrained_model is not None:
            if not os.path.isfile(self.settings.pretrained_model):
                raise FileNotFoundError(f"pretrained model not found: {self.settings.pretrained_model}")
            
            state_dict = torch.load(self.settings.pretrained_model, map_location="cpu")
            new_state_dict = self.model.state_dict()
            
            for k, v in state_dict.items():
                if k in new_state_dict.keys() and new_state_dict[k].size() == v.size():
                    new_state_dict[k] = v
            
            self.model.load_state_dict(new_state_dict)
            self.recorder.logger.info(f"Loaded pretrained weight from: {self.settings.pretrained_model}")

        if self.settings.checkpoint is not None:
            if not os.path.isfile(self.settings.checkpoint):
                raise FileNotFoundError(f"checkpoint not found: {self.settings.checkpoint}")
            
            checkpoint_data = torch.load(self.settings.checkpoint, map_location="cpu")
            self.model.load_state_dict(checkpoint_data["model"])
            self.trainer.optimizer.load_state_dict(checkpoint_data["optimizer"])
            self.trainer.aux_optimizer.load_state_dict(checkpoint_data["aux_optimizer"])
            self.epoch_start = checkpoint_data["epoch"] + 1
            self.recorder.logger.info(f"Resumed from epoch {self.epoch_start}")

    def run(self):
        t_start = time.time()
        
        if self.settings.val_only:
            self.trainer.run(0, mode="Validation")
            return
        
        best_val_result = None

        for epoch in range(self.epoch_start, self.settings.n_epochs):
            # 训练
            self.trainer.run(epoch, mode="Train")
            torch.cuda.empty_cache()
            
            # 验证
            if epoch % self.settings.val_frequency == 0 or epoch == self.settings.n_epochs - 1:
                val_result = self.trainer.run(epoch, mode="Validation")
                
                # 保存最佳模型
                if best_val_result is None:
                    best_val_result = val_result
                
                model_state = self.model.state_dict()
                for k, v in val_result.items():
                    if v >= best_val_result[k]:
                        self.recorder.logger.info(f"Better {k} model: {v}")
                        best_val_result[k] = v
                        if k == "IOU":
                            saved_path = os.path.join(self.recorder.checkpoint_path, f"best_{k}_model.pth")
                            torch.save(model_state, saved_path)

            # 保存checkpoint
            saved_path = os.path.join(self.recorder.checkpoint_path, "checkpoint.pth")
            checkpoint_data = {
                "model": self.model.state_dict(),
                "optimizer": self.trainer.optimizer.state_dict(),
                "aux_optimizer": self.trainer.aux_optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint_data, saved_path)
            
            if best_val_result is not None:
                log_str = "Best Result: " + " ".join([f"{k}: {v:.4f}" for k, v in best_val_result.items()])
                self.recorder.logger.info(log_str)
        
        cost_time = time.time() - t_start
        self.recorder.logger.info(f"Total training time: {datetime.timedelta(seconds=cost_time)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PMF Training")
    parser.add_argument("config_path", type=str, help="path to config file")
    args = parser.parse_args()
    
    exp = Experiment(Option(args.config_path))
    print("=== Environment initialized ===")
    exp.run()
