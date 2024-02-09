import nemo
import copy
import torch
import nemo.collections.asr as nemo_asr
import pytorch_lightning as pl

from nemo.utils.exp_manager import exp_manager
from omegaconf import DictConfig, OmegaConf

checkpoint_path = '~/QuartzNet15x5_ru--val_wer=0.3912-epoch=116.ckpt'

train_manifest_path = '~/some/path/to/train_manifest.json'
validation_manifest_path = '~/some/path/to/manifest.json'
test_manifest_path = '~/some/path/to/test_manifest.json'

model = nemo_asr.models.EncDecCTCModel.load_from_checkpoint(checkpoint_path)

torch.set_float32_matmul_precision('medium')

trainer = pl.Trainer(
             devices=1,
             max_epochs=41,
             max_steps=-1,
             num_nodes=1,
             accelerator='gpu',
             strategy='ddp',
             accumulate_grad_batches=1,
             log_every_n_steps=100,
             val_check_interval=1.0
          )

model.set_trainer(trainer)

params = model.cfg
new_opt = copy.deepcopy(params.optim)
new_opt.lr = 0.0001

params['train_ds']['is_tarred'] = False
params['train_ds']['shuffle'] = True
OmegaConf.update(params, "train_ds", {'num_workers': 4}, force_add=True)
OmegaConf.update(params, "train_ds", {'pin_memory': True}, force_add=True)
params['train_ds']['min_duration'] = 0.1
params['train_ds']['manifest_filepath'] = train_manifest_path
params['validation_ds']['manifest_filepath'] = validation_manifest_path
params['validation_ds']['shuffle'] = False
OmegaConf.update(params, "validation_ds", {'num_workers': 4}, force_add=True)
OmegaConf.update(params, "validation_ds", {'pin_memory': True}, force_add=True)
params['test_ds']['manifest_filepath'] = test_manifest_path

model.setup_optimization(optim_config=new_opt)
model.setup_training_data(train_data_config=params['train_ds'])
model.setup_validation_data(val_data_config=params['validation_ds'])
model.setup_test_data(test_data_config=params['test_ds'])

trainer.fit(model)