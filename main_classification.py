import os
import torch
import pytorch_lightning as pl
from clasification_model_new import GraspClassifier
from dataset_new import GraspDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from PIL import Image
import matplotlib.pyplot as plt

dataset_path = '/home/jannik/media/GitHub/model_jannik/utils/classification'
checkpoint_path = '/home/jannik/media/GitHub/model_jannik/utils/'


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
#trainer.test(model, datamodule=data_module)

#input_image = Image.open('/media/local/jannik/eval_with_gripper_abs_rel/segmentation_analyse/lift_pink_block_table/classification/open_gripper/gripper_closed_6_79.png').convert('RGB')
input_image = Image.open('/home/jannik/media/GitHub/model_jannik/utils/classification/closed/2_34.png').convert('RGB')
input_tensor = data_module.data_transform(input_image)
input_tensor = input_tensor.unsqueeze(0)
ckpt_files = [file for file in os.listdir(checkpoint_path) if file.endswith('.ckpt')]
model_save_path = os.path.join(checkpoint_path, ckpt_files[0])
model = GraspClassifier.load_from_checkpoint(model_save_path, num_classes=2, class_weights=class_weights)
model.eval()
with torch.no_grad():
    predicted_label, predicted_probabilities = model.predict(input_tensor)