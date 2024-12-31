import argparse
import torch
from typing import List, Optional
import sys
from functools import partial
import pickle
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import numpy.random
from torch.utils.data import Dataset
from models.ssl import SimCLR
import torchvision.transforms as transforms
from utils import datautils
from revisitop.dataset import configdataset
from revisitop.evaluate import compute_map
import torch.distributed as dist

test_transform = transforms.Compose([
                datautils.CenterCropAndResize(proportion=0.875, size=224),
                transforms.ToTensor(),
                #lambda x: (255 * x).byte(),
            ])

class OxfordParisDataset(Dataset):
    def __init__(self, gnd_path, data_root, transform, noise, query=False, seed=42, denoised=False):
        self.gnd_path = gnd_path
        self.data_root = data_root
        self.query = query
        with open(gnd_path, "rb") as f:
            gnd = pickle.load(f)
            if self.query:
                self.data = gnd["qimlist"]
            else:
                self.data = gnd["imlist"]
        self.transform = transform
        self.denoised=denoised

    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, index):
        if self.denoised:
            file_name = self.data_root + "/" + self.data[index] + ".png"
        else:
            file_name = self.data_root + "/" + self.data[index] + ".jpg"
        img = Image.open(file_name).convert(mode="RGB")
        img = self.transform(img)
        return img

def get_args_parser():
    parser = argparse.ArgumentParser(
        description=description
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default=""
    )
    parser.add_argument(
        "--test_dataset",
        choices=["roxford5k", "rparis6k"]
    )
    parser.add_argument(
        "--noise",
        choices=["gauss_50", "gauss_100", "gauss_255", "shot_10", "shot_3", "shot_1", "speckle_0.4", "speckle_0.7", "speckle_1.0", "identity_0"]
    )
    parser.add_argument(
        "--log_file",
        default="",
        required=True
    )
    parser.add_argument(
        "--step",
        required=True
    )
    return parser

def main(args):
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:1235',
        world_size=1,
        rank=0,
    )
    device = torch.device('cuda:0')
    denoised = False
    cfg = configdataset(args.test_dataset, os.path.join(args.data_root, 'datasets'))
    f = open(args.log_file, "a")
    f.write("##########################################################################\n")
    img_data_root = os.path.join(args.data_root, "datasets", args.test_dataset, "jpg")
        
        
    dataset_pkl = os.path.join(args.data_root, "datasets", args.test_dataset, "gnd_"+args.test_dataset+".pkl")
    transform = test_transform
    database_dataset = OxfordParisDataset(dataset_pkl, img_data_root, transform, args.noise, query=False, seed=42, denoised=denoised)

    database_data_loader = torch.utils.data.DataLoader(
        database_dataset,
        batch_size=32,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        persistent_workers=False
    )

    query_dataset = OxfordParisDataset(dataset_pkl, img_data_root, transform, args.noise, query=True, seed=100, denoised=denoised)

    query_data_loader = torch.utils.data.DataLoader(
        query_dataset,
        batch_size=32,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        persistent_workers=False
    )
    ckpt = torch.load(f"logs/exman-train.py/runs/000023/checkpoint-{args.step}.pth.tar")

    model = SimCLR.load(ckpt, device=device)

    #autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=autocast_dtype)
    #feature_model = ModelWithNormalize(model)
    model.eval()
    #model.cuda()
    with torch.no_grad():
        ### if normalize
        model.eval()
        database_feature_list = []
        for data in tqdm(database_data_loader):
            data = data.cuda()
            database_features = model(data, out='h').float()
            database_feature_list.append(database_features)
        database_feature_mtx = torch.cat(database_feature_list, dim=0).t()

        X = database_feature_mtx.cpu().numpy()

        query_feature_list = []
        for query in tqdm(query_data_loader):
            query=query.cuda()
            query_features = model(query, out='h').float()
            query_feature_list.append(query_features)
        query_feature_mtx = torch.cat(query_feature_list, dim=0).t()

        Q = query_feature_mtx.cpu().numpy()
    print(X.shape)
    print(Q.shape)
    
    # perform search
    f.write('>> {}: Retrieval...\n'.format(args.test_dataset))
    sim = np.dot(X.T, Q)
    ranks = np.argsort(-sim, axis=0)

    # revisited evaluation
    gnd = cfg['gnd']

    # evaluate ranks
    ks = [1, 5, 10]

    # search for easy
    gnd_t = []
    for i in range(len(gnd)):
        g = {}
        g['ok'] = np.concatenate([gnd[i]['easy']])
        g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['hard']])
        gnd_t.append(g)
    mapE, apsE, mprE, prsE = compute_map(ranks, gnd_t, ks)

    # search for easy & hard
    gnd_t = []
    for i in range(len(gnd)):
        g = {}
        g['ok'] = np.concatenate([gnd[i]['easy'], gnd[i]['hard']])
        g['junk'] = np.concatenate([gnd[i]['junk']])
        gnd_t.append(g)
    mapM, apsM, mprM, prsM = compute_map(ranks, gnd_t, ks)

    # search for hard
    gnd_t = []
    for i in range(len(gnd)):
        g = {}
        g['ok'] = np.concatenate([gnd[i]['hard']])
        g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['easy']])
        gnd_t.append(g)
    mapH, apsH, mprH, prsH = compute_map(ranks, gnd_t, ks)

    f.write('>> {}: mAP E: {}, M: {}, H: {}\n'.format(args.test_dataset, np.around(mapE*100, decimals=2), np.around(mapM*100, decimals=2), np.around(mapH*100, decimals=2)))
    f.write('>> {}: mP@k{} E: {}, M: {}, H: {}\n'.format(args.test_dataset, np.array(ks), np.around(mprE*100, decimals=2), np.around(mprM*100, decimals=2), np.around(mprH*100, decimals=2)))
    f.write("##########################################################################\n")


            
        

    #sample_input = torch.randn(1, 3, 224, 224).cuda()
    #sample_output = feature_model(sample_input)[-1:]
    #class_token = sample_output[0][1]
    #assert class_token.shape == (384,)

if __name__ == "__main__":
    description = "DINOv2 linear evaluation"
    args_parser = get_args_parser()
    args = args_parser.parse_args()
    sys.exit(main(args))