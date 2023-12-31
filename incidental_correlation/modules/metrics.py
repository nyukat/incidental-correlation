import torch
from pytorch_lightning.metrics import Metric


class Accuracy(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits, target):
        logits, target = (
            logits.detach().to(self.correct.device),
            target.detach().to(self.correct.device),
        )
        preds = (torch.sigmoid(logits) > 0.5).int()
        if target.numel() == 0:
            return 1

        assert preds.shape == target.shape

        self.correct += torch.sum(preds == target.int())
        self.total += target.numel()

    def compute(self):
        return self.correct / self.total


class VQAScore(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits, target):
        logits, target = (
            logits.detach().float().to(self.score.device),
            target.detach().float().to(self.score.device),
        )
        logits = torch.max(logits, 1)[1] # Column with max logit value
        one_hots = torch.zeros(*target.size()).to(target)
        one_hots.scatter_(1, logits.view(-1, 1), 1) # Array of zeros, except for one at the column with max logit value
        scores = one_hots * target # Array of zeros, except for the target at the column with max logit value

        self.score += scores.sum()
        self.total += len(logits)

    def compute(self):
        return self.score / self.total # Average target at the column with max logit value