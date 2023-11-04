import os
import torch
import pytorch_lightning as pl
from clasification_model_new import GraspClassifier
from dataset_new import GraspDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from PIL import Image
import matplotlib.pyplot as plt

dataset_path = 'xxx'
checkpoint_path = 'xxx'


data_module = GraspDataModule(dataset_path=dataset_path)
data_module.prepare_data()


class_weights = data_module.class_weights
model = GraspClassifier(num_classes=2, class_weights=class_weights)

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    filename='model_{epoch:02d}_{val_loss:.4f}',
    dirpath= checkpoint_path,
    save_top_k=1,
    mode='min'
)

trainer = pl.Trainer(max_epochs=25, gpus=1, callbacks=[checkpoint_callback])
trainer.fit(model, data_module)
