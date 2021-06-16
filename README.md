# Machine Learning Project1

## Code Structure
* FracNet/
    * [`dataset/`](./dataset): PyTorch dataset and transforms.
    * [`models/`](./models): PyTorch 3D UNet model and losses.
    * [`utils/`](./utils): Utility functions.
    * [`main.py`](main.py): Training script.

## Requirements
```
SimpleITK==1.2.4
fastai==1.0.59
fastprogress==0.1.21
matplotlib==3.1.3
nibabel==3.0.0
numpy>=1.18.5
pandas>=0.25.3
scikit-image==0.16.2
torch==1.4.0
tqdm==4.38.0
```
### Training
To train the model, run the following in command line:
```bash
python -m main --train_image_dir <training_image_directory> --train_label_dir <training_label_directory> --val_image_dir <validation_image_directory> --val_label_dir <validation_label_directory>
```

### Prediction
To generate prediction, run the following in command line:
```bash
python -m predict --image_dir <image_directory> --pred_dir <predition_directory> --model_path <model_weight_path>
```


