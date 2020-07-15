# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
TridentNet Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

import os
import time
import torch

import numpy as np

from detectron2.utils import comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.utils.events import JSONWriter, TensorboardXWriter
from continuous.custom import add_custom_config
from continuous.custom.data import build_classification_train_loader, build_classification_test_loader

from continuous.utils.metrics import AverageMeter, ProgressMeter, accuracy
from continuous.utils.events import CommonMetricPrinterCustom
from continuous.custom.evaluation import ClassificationEvaluator


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
class Trainer(DefaultTrainer):
    top1 = AverageMeter('Acc@1', ':6.2f')
    def build_writers(self):
        """
        Build a list of writers to be used. By default it contains
        writers that write metrics to the screen,
        a json file, and a tensorboard event file respectively.
        If you'd like a different list of writers, you can overwrite it in
        your trainer.

        Returns:
            list[EventWriter]: a list of :class:`EventWriter` objects.

        It is now implemented by:

        .. code-block:: python

            return [
                CommonMetricPrinter(self.max_iter),
                JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
                TensorboardXWriter(self.cfg.OUTPUT_DIR),
            ]

        """
        # Here the default print/log frequency of each writer is used.
        return [
            # It may not always print what you want to see, since it prints "common" metrics only.
            CommonMetricPrinterCustom(self.max_iter),
            JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(self.cfg.OUTPUT_DIR),
        ]

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        """
        If you want to do something with the losses, you can wrap the model.
        """
        loss_dict, outputs, targets = self.model(data)
        losses = sum(loss_dict.values())
        self._detect_anomaly(losses, loss_dict)

        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        self.top1.update(acc1[0], targets.size(0))

        metrics_dict = loss_dict
        metrics_dict["data_time"] = data_time
        metrics_dict["acc"] = self.top1.avg
        self._write_metrics(metrics_dict)

        """
        If you need to accumulate gradients or something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()
        losses.backward()

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method.
        """
        self.optimizer.step()
    @classmethod
    def build_train_loader(cls, cfg):
        return build_classification_train_loader(cfg)
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_classification_test_loader(cfg, dataset_name)
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return ClassificationEvaluator(True)

    def _write_metrics(self, metrics_dict: dict):
        """
        Args:
            metrics_dict (dict): dict of scalar metrics
        """
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                # data_time among workers can have high variance. The actual latency
                # caused by data_time is the maximum among workers.
                data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            classification_loss = np.mean([x.pop("classification") for x in all_metrics_dict])
            self.storage.put_scalar("total_loss", classification_loss)

            classification_acc = np.mean([x.pop("acc") for x in all_metrics_dict])
            self.storage.put_scalar("total_acc", classification_acc)

            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_custom_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    args.config_file = 'configs/classification/Base-Classification.yaml'
    args.num_gpus = 1
    # args.resume=True
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
