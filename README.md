# contrastive-learning


## Install dependencies in conda

```bash
conda create --yes --name venv python=3.6
conda activate venv
conda install --yes --file requirements-conda
conda install --yes pytorch torchvision cudatoolkit=11.0 -c pytorch 
python -m pip install -r requirements-pip
```

## Examples

Pretrain a model with bg augmentation from dtd stored in data/datasets/dtd
```bash
python run.py --config_file configs/deep_supervision_triplet_blur.json --texture_directory data/datasets/dtd --target cone --output_dir data/test
```
