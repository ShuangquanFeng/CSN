import argparse
import torch
from config.datasets import *
from config.params import *
from lib.preprocess_datasets import *

parser = argparse.ArgumentParser(description='preprocess the UNBC-McMaster dataset')
parser.add_argument('--gpu', default=0, type=int, help="the GPU to use")

# Preprocesss the UNBC-McMaster dataset
def main():
    args = parser.parse_args()
    device = torch.device('cuda:' + str(args.gpu))
    preprocess_datasets(UNBCMCMASTER_IMG_ROOTDIR, PREPROCESS_STEPS,device)
    
if __name__ == '__main__':
    main()