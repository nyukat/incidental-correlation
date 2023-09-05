import torch.nn as nn
import torch.nn.functional as F
from modules.nn_utils import MLP


class MultimodalClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, loss_fn):
        super().__init__()
        self.loss_fn = loss_fn
        self.model = MLP(input_dim, hidden_dims, output_dim)


    def forward(self, x, y_true):
        y_pred = self.model(x)
        loss = self.loss_fn(y_pred, y_true).mean()
        return {
            "loss": loss,
            "logits": y_pred
        }


class UnimodalClassifier(nn.Module):
    def __init__(self, image_dim, text_dim, hidden_dims, output_dim, loss_fn):
        super().__init__()
        self.loss_fn = loss_fn
        self.image_model = MLP(image_dim, hidden_dims, output_dim)
        self.text_model = MLP(text_dim, hidden_dims, output_dim)


    def forward(self, x_image, x_text, y_true):
        logits_image = F.log_softmax(self.image_model(x_image))
        logits_text = F.log_softmax(self.text_model(x_text))
        logits = logits_image + logits_text
        loss = self.loss_fn(logits, y_true).mean()
        return {
            "loss": loss,
            "logits": logits
        }