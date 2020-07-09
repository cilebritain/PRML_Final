import argparse
from treeLSTM.source import CrossValidate

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../../data/train_cv')
    parser.add_argument('--save_dir', type=str, default='./checkpoint/checkpoint')  
    parser.add_argument('--model_kind', type=str, choices=['lstm', 'tree'], default='lstm')
    arg = parser.parse_args()
    CrossValidate(datapath=arg.data_path, model_kind=arg.model_kind, fold_num=0)