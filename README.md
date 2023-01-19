# Dog breed classification


## Requirements

- python >= 3.7.x


## Instalation

### Steps
1. Download the kaggle dataset
Use the instruction from here https://github.com/Kaggle/kaggle-api to install the Kaggle API
```bash
kaggle competitions download -c dog-breed-identification
```


2. Run project installation
```bash
python3 -m pip install -e 
```

3. To visualize some images in the dataset run 
```bash
python -m cli.commands show_multiple --image_path "data/train" --num_img 4
```

4. To start the training process: 
```bash
python -m cli.commands train_model --image_path "data/train"
```
To see available options run: 
```bash
python -m cli.commands train_model --help
```