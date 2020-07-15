# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import contextlib
import copy
import io
import itertools
import logging
from collections import OrderedDict
import torch
import numpy as np

import detectron2.utils.comm as comm
from detectron2.evaluation.evaluator import DatasetEvaluator
from continuous.utils.metrics import AverageMeter, ProgressMeter, accuracy


class ClassificationEvaluator(DatasetEvaluator):

    def __init__(self, distributed):
        self._distributed = distributed

        self._cpu_device = torch.device("cpu")

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        top1 = AverageMeter('Acc@1', ':6.2f')

        targets = []
        for x in inputs:
            targets.append(x[1])
        targets = torch.tensor(targets)

        acc1, acc5 = accuracy(outputs['linear'].detach().cpu(), targets, topk=(1, 5))

        top1.update(acc1[0], targets.size(0))
        self._predictions.append(top1.avg)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        self._results = OrderedDict()

        self._results['val'] = {'acc':np.mean([x for x in predictions])}
        return copy.deepcopy(self._results)
