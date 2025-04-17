# Training

Before start training, please make sure you have prepared the dataset listed [here](./INSTALL.md/#table-of-contents).

```python
python train.py -dc path_to_data_config -ec path_to_exp_config -m path_to_ckpt_model

# examples:
# training on KITTI360 static scene
python train.py -dc configs/kitti360/static/3.yaml -ec configs/exp.yaml

# training on Waymo dynamic scene
python train.py -dc configs/waymo/dynamic/1.yaml -ec configs/exp.yaml

# resume training
python train.py -dc configs/waymo/dynamic/1.yaml -ec configs/exp.yaml -m output/scene_wd1/models/model_it_25000.pth

# refine the unet using a specific model
python train.py -dc configs/waymo/dynamic/1.yaml -ec configs/exp.yaml -m output/scene_wd1/models/model_it_25000.pth -r
```

<a id="training-args"></a>
<details>
  <summary><span style="font-weight: bold;">🔍 Click here to see the detailed Command Line Arguments for training.</span></summary>
  <hr>

#### -dc, --data_config_path
Data configuration file (required).
#### -ec, --exp_config_path
Experiment configuration file (required).
#### -m, --model
Path to load checkpoint model (optional). You can resume a interrupted training by specifying a checkpoint model.
#### -r, --only_refine
Skip the gaussian optimization stage, directly refine the unet using a specific model. You must specify the model path.

  <hr>
</details>
