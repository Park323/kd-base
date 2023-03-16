import os
import os.path as osp
import glob
import re
import yaml
import argparse
import subprocess

# TASKS = ['ESC-50','SpeechCommandsV2']
dataname = 'ESC-50'

if __name__ == "__main__":
    ## Parse arguments
    parser = argparse.ArgumentParser(description = "Run a whole training sequence.")

    parser.add_argument('-c', '--config', type=str,   default='./configs/mnist.yaml',   help='Config YAML file')
    parser.add_argument('-x', '--exp',     type=str,   default='exp1',   help='name dir')
    # parser.add_argument('-e', '--epoch', type=int, default=100)
    # parser.add_argument('-p', '--profiler', type=str, default=None)

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config_scripts = f.read()
    train_fold = re.search("(?<=&train_fold)[^\n]*", config_scripts).group()
    train_fold = list(map(int, re.sub("[\[\]]", "", train_fold.strip()).split(',')))
    test_fold = re.search("(?<=&test_fold)[^\n]*", config_scripts).group()
    test_fold = list(map(int, re.sub("[\[\]]", "", test_fold.strip()).split(',')))
    fold_idx = train_fold + test_fold

    # Generate configs for training
    exp_path = f'exp/{args.exp}'
    if not osp.exists(exp_path):
        os.makedirs(exp_path)

    import subprocess
    for ex_fold in fold_idx:
        with open(f'{exp_path}/exclude_{ex_fold}.yaml', 'w') as f:
            fold_path = f"{exp_path}/exclude_{ex_fold}"
            if not osp.exists(fold_path):
                os.makedirs(fold_path)
            config_scripts=re.sub("(?<=&result_dir )[^\n]*", f"{exp_path}/exclude_{ex_fold}", config_scripts)
            config_scripts=re.sub("(?<=&train_fold )[^\n]*", f"{[fold_id for fold_id in fold_idx if fold_id != ex_fold]}", config_scripts)
            config_scripts=re.sub("(?<=&test_fold )[^\n]*", f"[{ex_fold}]", config_scripts)
            ckpt = glob.glob(f"checkpoint/HTSAT_ESC_exp=1_fold={ex_fold-1}*")[0]
            config_scripts=re.sub("(?<=resume_checkpoint: )[^\n]*", ckpt, config_scripts)
            f.write(config_scripts)

        command = f'python main.py --config {exp_path}/exclude_{ex_fold}.yaml --mode train'
        subprocess.run(command.split())