import argparse
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from model_electra import ER
from train_utils import train_plmr, dev_plmr, annotation_plmr
from data.beer import BeerAnnotationData, BeerDataCorrelated
from data.hotel import HotelDataset, HotelAnnotationData
from model import PLMR
from transformers import BertTokenizerFast, ElectraTokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=77, help='')
    parser.add_argument('--gpu', type=int, default=0, help='')
    parser.add_argument('--lr_trans', type=float, default=0.000005, help='')
    parser.add_argument('--lr_mlp', type=float, default=0.00002, help='')
    parser.add_argument('--num_labels', type=int, default=2, help='')
    parser.add_argument('--cls_lambda', type=float, default=1.0, help='')
    parser.add_argument('--D_lambda', type=float, default=1, help='')
    parser.add_argument('--full_text_lambda', type=float, default=2.0, help='')
    parser.add_argument('--continuity_lambda', type=float, default=10, help='')
    parser.add_argument('--sparsity_lambda', type=float, default=5, help='')
    parser.add_argument('--sparsity_percentage', type=float, default=0.2, help='')
    parser.add_argument('--dim_reduction_start', type=int, default=3, help='')
    parser.add_argument('--dim_reduction_end', type=int, default=7, help='')
    parser.add_argument('--max_length', type=int, default=256, help='')
    parser.add_argument('--data_dir', type=str, default='/newdisk/ylb/data/beer', help='')
    parser.add_argument('--data_type', type=str, default='beer', help='')
    parser.add_argument('--aspect', type=int, default=0, help='')
    parser.add_argument('--annotation_path', type=str, default='/newdisk/ylb/data/beer/annotations.json', help='')
    parser.add_argument('--batch_size', type=int, default=64, help='')
    parser.add_argument('--balance', type=bool, default=True, help='')
    parser.add_argument('--epochs', type=int, default=12, help='')
    parser.add_argument('--model', type=str, default='bert-base', help='')
    args = parser.parse_args()
    return args

args = parse_args()
args.hidden_dim = 768
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
device = torch.device("cuda:{}".format(args.gpu) )

if args.model == 'bert-base':
    tokenizer = BertTokenizerFast.from_pretrained('/newdisk/ylb/Transformers/bert-base-uncased')
    model = PLMR(args)
elif args.model == 'electra-base':
    tokenizer = ElectraTokenizer.from_pretrained('/newdisk/ylb/Transformers/electra-base')
    model = ER(args)



if args.data_type == 'beer':
    print('beer')
    args.data_dir = '/newdisk/ylb/data/beer'
    args.annotation_path = '/newdisk/ylb/data/beer/annotations.json'
    train_data = BeerDataCorrelated(tokenizer, args.data_dir, args.aspect, 'train', max_length=args.max_length, balance=True)
    dev_data = BeerDataCorrelated(tokenizer, args.data_dir, args.aspect, 'heldout', max_length=args.max_length, balance=True)
    annotation_data = BeerAnnotationData(tokenizer, args.annotation_path, args.aspect, max_length=args.max_length)
elif args.data_type == 'hotel':
    print('hotel')
    args.data_dir = '/newdisk/ylb/data/hotel'
    args.annotation_path = '/newdisk/ylb/data/hotel/annotations'
    train_data = HotelDataset(tokenizer, args.data_dir, args.aspect, 'train', balance=True)
    dev_data = HotelDataset(tokenizer, args.data_dir, args.aspect, 'dev', balance=True)
    annotation_data = HotelAnnotationData(tokenizer, args.annotation_path, args.aspect, max_length=args.max_length)

train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
dev_loader = DataLoader(dev_data, batch_size=args.batch_size)
annotation_loader = DataLoader(annotation_data, batch_size=args.batch_size)

model.to(device)

para0 = []
for i in range(0, args.dim_reduction_start + 1):
    for p in model.bert.encoder.layer[i].parameters():
        if p.requires_grad == True:
            para0.append(p)
optimizer0 = torch.optim.Adam(para0, lr=args.lr_trans)

para1 = []
for i in range(args.dim_reduction_start + 1, args.dim_reduction_end + 1):
    for p in model.bert.encoder.layer[i].parameters():
        if p.requires_grad == True:
            para1.append(p)
optimizer1 = torch.optim.Adam(para1, lr=args.lr_trans)

para2 = []
for i in range(args.dim_reduction_end + 1, 12):
    for p in model.bert.encoder.layer[i].parameters():
        if p.requires_grad == True:
            para2.append(p)
for p in model.bert.pooler.parameters():
    if p.requires_grad == True:
        para2.append(p)
optimizer2 = torch.optim.Adam(para2, lr=args.lr_trans)

para3 = []
for p in model.cls_predictor.parameters():
    if p.requires_grad == True:
        para3.append(p)
optimizer3 = torch.optim.Adam(para3, lr=args.lr_mlp)

para_dim_reduction = []
for p in model.bert.encoder.dim_reduction_predictors.parameters():
    if p.requires_grad == True:
        para_dim_reduction.append(p)
optimizer_dim_reduction = torch.optim.Adam(para_dim_reduction, lr=args.lr_mlp)

for epoch in range(args.epochs):
    print("\n当前时间是:", datetime.now())
    start = time.time()
    model.train()
    print("train model")
    precision, recall, f1_score, accuracy= (
        train_plmr(model, optimizer0, optimizer1, optimizer2, optimizer3, optimizer_dim_reduction, train_loader,
                        device, args))

    print("traning epoch:{} precision:{:.4f} recall:{:.4f} accuracy:{:.4f} f1-score:{:.4f}".format(epoch, precision, recall, accuracy, f1_score))

    model.eval()
    print("dev model")
    precision, recall, f1_score, accuracy = dev_plmr(model, dev_loader, device)
    print("precision:{:.4f} recall:{:.4f} accuracy:{:.4f} f1-score:{:.4f}".format(precision, recall, accuracy, f1_score))

    print("Annotation")
    precision, recall, f1_score, accuracy, sparsity, R_precision, R_recall, R_f1 = annotation_plmr(model, annotation_loader, device, args)
    print("precision:{:.4f} recall:{:.4f} accuracy:{:.4f} f1-score:{:.4f}".format(precision, recall, accuracy, f1_score))
    print("the Annotation result sparsity :{:.4f}  R_precision :{:.4f}  R_recall :{:.4f}  R_f1 :{:.4f} ".format(sparsity, R_precision, R_recall, R_f1))






