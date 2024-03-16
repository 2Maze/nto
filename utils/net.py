import torchvision
import torchmetrics
import torch
import torch.nn as nn
import lightning as L
import torch.nn.functional as F

from utils.utils import calculate_class_weights_from_dataframe


class LitModule(L.LightningModule):
    def __init__(self, num_classes, label_maps):
        super().__init__()

        self.label_maps = label_maps

        # Model
        self.model = torchvision.models.inception_v3(pretrained=True)
        self.model.fc = nn.Sequential(nn.Linear(2048, num_classes, bias=True))
                                      #nn.ReLU(inplace=True),
                                      #nn.Dropout(0.3),
                                      #nn.Linear(1024, num_classes, bias=True))
        self.weights = torch.tensor([0.0171, 0.0093, 0.0108, 0.0280, 0.0097, 0.0144, 0.0097, 0.0266, 0.0095,
                                     0.0177, 0.0190, 0.0106, 0.0177, 0.0130, 0.0097, 0.0118, 0.0130, 0.0171,
                                     0.0098, 0.0089, 0.0090, 0.0116, 0.0108, 0.0083, 0.0089, 0.0097, 0.0144,
                                     0.0106, 0.0118, 0.0095, 0.0231, 0.0106, 0.0177, 0.0190, 0.0089, 0.0100,
                                     0.0073, 0.0075, 0.0197, 0.0152, 0.0133, 0.0084, 0.0102, 0.0130, 0.0221,
                                     0.0190, 0.0108, 0.0177, 0.0108, 0.0084, 0.0098, 0.0090, 0.0204, 0.0104,
                                     0.0104, 0.0102, 0.0073, 0.0127, 0.0148, 0.0144, 0.0133, 0.0100, 0.0124,
                                     0.0104, 0.0108, 0.0221, 0.0221, 0.0483, 0.0140, 0.0111, 0.0097, 0.0106,
                                     0.0140, 0.0108])

        # Metrics
        self.acc = torchmetrics.classification.Accuracy(task='multiclass', num_classes=num_classes)
        self.f1 = torchmetrics.classification.F1Score(task='multiclass', num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.forward(x)
        loss = F.cross_entropy(outputs.logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.forward(x)
        loss = F.cross_entropy(outputs, y)
        self.acc(outputs, y)
        self.f1(outputs, y)
        self.log('val_loss', loss)

    def on_validation_epoch_end(self):
        self.log('val_acc', self.acc)
        self.log('val_f1', self.f1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80], gamma=0.1)
        return [optimizer], [scheduler]
