from collections import defaultdict
from typing import List

import torch

from dataloaders import ClassConditionalLoader
from distributed_util import reduce_tensor


class ClassifierTester:
    def __init__(self, words_file: str, num_gpus: int) -> None:
        self.class_conditional_loader = ClassConditionalLoader(words_file=words_file)
        self.num_gpus = num_gpus
        self.loss = torch.nn.CrossEntropyLoss()
        self.reset()
    
    def reset(self):
        self.logs = {
            key: defaultdict(int)
            for key in ['train', 'val', 'test']
        }
    
    def evaluate(self, set: str):
        return self.logs[set]['loss_cumulative'] / self.logs[set]['step_count']

    def last_loss(self, set: str):
        return self.logs[set]['last_loss']

    def _update_logs(self, set: str, value: float):
        self.logs[set]['last_loss'] = value
        self.logs[set]['loss_cumulative'] += value
        self.logs[set]['step_count'] += 1

    def __call__(
        self, 
        classifier: torch.nn.Module, 
        input: torch.Tensor, 
        labels: List[str], 
        set: str,
    ) -> float:
        with torch.no_grad():
            pred_classes = classifier(input.cuda())

        real_classes = self.class_conditional_loader.batch_call(labels).cuda()
        # real_classes = real_classes.squeeze(1)

        loss_val = self.loss(pred_classes, real_classes)

        if self.num_gpus > 1 and set == 'train':
            loss_val = reduce_tensor(loss_val.data, self.num_gpus)
        
        loss_val = loss_val.item()

        self._update_logs(set, loss_val)

        return loss_val