import pandas as pd
import lightning as L

from torch.utils.data import DataLoader
from torchvision.transforms import v2
from utils.net import LitModule
from utils.dataset import AttractionsDataset
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch import seed_everything


def main():
    seed_everything(42, workers=True)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = v2.Compose([
        v2.Resize(299),
        v2.RandomResizedCrop(299),
        v2.ToTensor(),
        v2.Normalize(mean, std)
    ])

    test_transform = v2.Compose([
        v2.Resize(299),
        v2.CenterCrop(299),
        v2.ToTensor(),
        v2.Normalize(mean, std)
    ])
    dataset_path = './data/photos_v3.csv'
    data = pd.read_csv(dataset_path)
    labels = list(data['name'].unique())
    label_maps = {label: i for i, label in enumerate(labels)}
    train_dataset = AttractionsDataset('data/train.csv', label_maps=label_maps, transform=train_transform)
    val_dataset = AttractionsDataset('data/val.csv', label_maps=label_maps, transform=test_transform)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=32, num_workers=2)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=32, num_workers=2)

    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints_f',
        filename="classifier_{epoch:02d}",
        every_n_epochs=2,
        save_top_k=-1,
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    model = LitModule(len(train_dataset.label_maps), label_maps)
    trainer = L.Trainer(accelerator='gpu',
                        devices=1,
                        max_epochs=100,
                        callbacks=[checkpoint_callback, lr_monitor])
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == '__main__':
    main()
