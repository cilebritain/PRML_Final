# PRML_Final
**PRML Final Project, Spring 2020, Fudan University.**

**By Yuchun Dai and Yipei Xu.**

We implement LSTM, Tree-LSTM, Chemprop method in this project.

### Data

the task data can been seen under `data` folder, which comes from https://www.aicures.mit.edu/tasks , Our prediction results for each fold are also under the folder.

### Model 

Our model can been seen under `model` folder, you can run the following command to run the model:

### Result

Our result are under `result` folder, you can take it as a example if you want to implement our method to the task

### Reference Paper

Our reference paper are mainly *Chemprop* and *Seq2Seq-fingerprint*, you can see the paper pdf and github link under `ref-paper` folder.

### Run Code

To run our LSTM&&Tree-LSTM model, please open the fold *LSTM&&Tree-LSTM* and input the following command in terminal:
```
python source.py --data_dir=<the fold saving train.csv, dev.csv and test.csv> --model=<'lstm' or 'tree'> --preds_dir=<the location saving prediction>
```
like
```
python source.py --data_dir='../../data/train_cv/fold_1' --model='tree' --preds_dir='../../data/train_cv/fold_1/preds.csv'
```
To run the chemprop model, please open the fole *chemprop* and input the following command in terminal:
```
python train.py --data_path <the location of train_data>  --separate_val_path <the location of validate_data>  --separate_test_path <the location of test_data> --save_dir <the location of checkpoint to save the model> 

python predict.py --test_path <the location of test_data> --checkpoint_dir <the location of checkpoint to save the model> --preds_path <the location to save the prediction>

```
like
```
python train.py --data_path ../../data/train_cv/fold_0/train.csv  --separate_val_path ../../data/train_cv/fold_0/dev.csv  --separate_test_path ../../data/train_cv/fold_0/test.csv --save_dir ../../data/checkpoints 

python predict.py --test_path ../../data/train_cv/fold_0/test.csv --checkpoint_dir ../../data/checkpoints --preds_path ../../data/train_cv/fold_0/preds_fl.csv 

```


### Report

Our project report is the `report.pdf` file.



7.9.2020
