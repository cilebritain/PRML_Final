import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../../data/train_cv/fold_1')
    parser.add_argument('--model', type=str, choices=['lstm', 'tree'], default='tree')
    parser.add_argument('--preds_dir', type=str, default='../../data/train_cv/fold_1/preds.csv')
    arg = parser.parse_args()

    CrossValidate(datapath=arg.data_dir, model_kind=arg.model, preds_dir=arg.preds_dir)