import os
import pandas as pd
from torch.utils.data import DataLoader
from irmas_dataset import IRMASDataset
# from doraemon.evaluation import plot_confusion_matrix, plot_TSNE, plot_prediction_map
import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import label_ranking_average_precision_score as LRAP
from tqdm import tqdm
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from utils import import_source_file
import argparse
import glob
from pathlib import Path
import warnings
import yaml

warnings.filterwarnings("ignore")
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


class Evaluator():
    def __init__(self,
                 model,
                 ckpt,
                 valid_meta_path,
                 valid_data_dir,
                 test_meta_path,
                 test_data_dir,
                 ):
        self.device = torch.device("cuda")
        self.ckpt = ckpt
        cfg_path = os.path.join(os.path.dirname(ckpt),'config.yaml')
        with open(cfg_path,'r') as cfg_f:
            cfg = yaml.safe_load(cfg_f)

        self.model = import_source_file(Path("./models/{}.py".format(model.lower())),
                                        "mm").IRNet(
            ch_expand=cfg['ch_expand']
        )

        self.prepare_model()

        self.valid_meta_path = valid_meta_path
        self.valid_data_dir = valid_data_dir

        self.test_meta_path = test_meta_path
        self.test_data_dir = test_data_dir

        self.valid_loader = self.get_valid_data()
        self.test_loader = self.get_test_data()

    @torch.no_grad()
    def evaluate_model(self, note=''):
        self.plot(note)
        self.test(note)

    @torch.no_grad()
    def test(self, note='', activation='sigmoid', pp='average', tem=1.0, seg_dur=3, add_up=False):
        print('------------Testing------------')
        print('[INFO]Activation Function: {}'.format(activation))
        print('[INFO]Post Processing: {}'.format(pp))
        if add_up:
            print('[INFO]Adding up all the segments for inference')
        if activation == 'softmax' and tem != 1.0:
            print('[INFO]Softmax Temperature: {}'.format(tem))
        if not note:
            note = self.ckpt.strip().split('/')[-1].split('.')[0]
        thresholds = np.arange(0.0, 1.0, 0.02)
        res = {}
        save_dir = self.ckpt.split(note)[0]
        preds_list = []
        labels_list = []
        for batch in tqdm(self.test_loader):
            signals = batch['feature'].cuda()
            labels = batch['one_hot_label']
            # labels = labels.repeat_interleave(signals.shape[1],dim=0)  # repeat labels to match length
            signals = signals.squeeze(0)
            if add_up:
                signals = signals.unfold(0, seg_dur * 22050, seg_dur * 22050)
                signals = torch.sum(signals, dim=0, keepdim=True) / signals.shape[0]
            else:
                signals = signals.unfold(0, seg_dur * 22050, seg_dur * 22050 // 2)
            # signals = signals.reshape(-1, signals.shape[-1])  # fold signals

            feats, outputs = self.model(signals)
            outputs /= tem

            if activation == 'sigmoid':
                preds = torch.sigmoid(outputs)
            elif activation == 'softmax':
                preds = F.softmax(outputs, dim=-1)
            else:
                raise NotImplementedError

            if pp == 'average':
                preds = torch.mean(preds, dim=0, keepdim=True)

            elif pp == 'average2':
                preds = torch.mean(preds, dim=0, keepdim=True)
                preds = preds / preds.amax()

            elif pp == 'maxpool':
                # preds = preds.transpose(0, 1)
                # preds = F.max_pool1d(preds, 6, 3)
                # preds = preds.transpose(0, 1)
                # preds = torch.sum(preds, dim=0, keepdim=True)
                preds = torch.max(preds, dim=0, keepdim=True)[0]
                # preds = (preds - preds.min()) / (preds.max() - preds.min())

            elif pp == 'norm':
                preds = torch.sum(preds, dim=0, keepdim=True)
                preds = preds / preds.amax()
            else:
                raise NotImplementedError

            # preds = preds / preds.amax()
            if add_up:
                preds = preds / preds.amax()

            preds_list.append(preds.detach().cpu())
            labels_list.append(labels)

        preds_list = torch.cat(preds_list, 0)
        labels_list = torch.cat(labels_list, 0)
        
        lrap = LRAP(labels_list, preds_list)


        for i, th in enumerate(thresholds):
            predictions = self.apply_threshold(preds=preds_list, threshold=th)
            results = self.compute_score(preds=predictions, labels=labels_list)
            results.update({'threshold': "{:.2f}".format(th)})
            res[i] = results

        res = pd.DataFrame.from_dict(res, orient='index', columns=res[0].keys())
        best_f1 = res['f1_score_micro'].max()
        # best_f1 = 'lol'
        if note:
            note += '-LRAP-{:.3f}-{}-{}-'.format(lrap,activation, pp)
        save_name = save_dir + note + 'evaluation_f1={}--tem={}'.format(best_f1, tem) + '.csv'
        print('best-f1-micro = {}, tem = {}'.format(best_f1, tem))
        res.to_csv(Path(save_name))

    @torch.no_grad()
    def ensemble(self, model_lst, note='', activation='sigmoid', pp='average', seg_dur=3, add_up=False):
        print('------------Model Ensemble------------')
        print('[INFO]Activation Function: {}'.format(activation))
        print('[INFO]Post Processing: {}'.format(pp))
        if add_up:
            print('[INFO]Adding up all the segments for inference')
        if not note:
            note = self.ckpt.strip().split('/')[-1].split('.')[0]

        prediction_table = np.zeros([len(model_lst), len(self.test_loader),11])

        thresholds = np.arange(0.0, 1.0, 0.02)

        for model_idx, model_path in enumerate(model_lst):
            res = {}
            save_dir = self.ckpt.split(note)[0]
            preds_list = []
            labels_list = []
            for batch in tqdm(self.test_loader):
                signals = batch['feature'].cuda()
                labels = batch['one_hot_label']
                # labels = labels.repeat_interleave(signals.shape[1],dim=0)  # repeat labels to match length
                signals = signals.squeeze(0)
                if add_up:
                    signals = signals.unfold(0, seg_dur * 22050, seg_dur * 22050)
                    signals = torch.sum(signals, dim=0, keepdim=True) / signals.shape[0]
                else:
                    signals = signals.unfold(0, seg_dur * 22050, seg_dur * 22050 // 2)
                # signals = signals.reshape(-1, signals.shape[-1])  # fold signals

                feats, outputs = self.model(signals)

                if activation == 'sigmoid':
                    preds = torch.sigmoid(outputs)
                elif activation == 'softmax':
                    preds = F.softmax(outputs, dim=-1)
                else:
                    raise NotImplementedError

                if pp == 'average':
                    preds = torch.mean(preds, dim=0, keepdim=True)

                elif pp == 'average2':
                    preds = torch.mean(preds, dim=0, keepdim=True)
                    preds = preds / preds.amax()

                elif pp == 'maxpool':
                    preds = preds.transpose(0, 1)
                    preds = F.max_pool1d(preds, 6, 3)
                    preds = preds.transpose(0, 1)
                    preds = torch.sum(preds, dim=0, keepdim=True)
                    preds = (preds - preds.min()) / (preds.max() - preds.min())

                elif pp == 'norm':
                    preds = torch.sum(preds, dim=0, keepdim=True)
                    preds = preds / preds.amax()
                else:
                    raise NotImplementedError

                # preds = preds / preds.amax()
                if add_up:
                    preds = preds / preds.amax()

                preds_list.append(preds.detach().cpu())
                labels_list.append(labels)

            preds_list = torch.cat(preds_list, 0)
            labels_list = torch.cat(labels_list, 0)

            lrap = LRAP(labels_list, preds_list)

            for i, th in enumerate(thresholds):
                predictions = self.apply_threshold(preds=preds_list, threshold=th)
                results = self.compute_score(preds=predictions, labels=labels_list)
                results.update({'threshold': "{:.2f}".format(th)})
                res[i] = results

            res = pd.DataFrame.from_dict(res, orient='index', columns=res[0].keys())
            best_f1 = res['f1_score_micro'].max()
            # best_f1 = 'lol'
            if note:
                note += '-LRAP-{:.3f}-{}-{}-'.format(lrap, activation, pp)
            save_name = save_dir + note + 'evaluation_f1={}'.format(best_f1) + '.csv'
            print('best-f1-micro = {}'.format(best_f1))
            res.to_csv(Path(save_name))

    @torch.no_grad()
    def plot(self, note='', dpi=250, normalize=True, tsne_cmap='hls'):
        if not note:
            note = self.ckpt.split('checkpoints/')[1].split('-')[0]
        save_dir = self.ckpt.split('checkpoints')[0]
        feats_list = []
        preds_list = []
        labels_list = []
        for batch in tqdm(self.valid_loader):
            signals = batch['feature'].cuda()
            labels = batch['label']
            # labels = labels.repeat_interleave(signals.shape[1])  # repeat labels to match length
            # signals = signals.reshape(-1, signals.shape[-1])  # fold signals

            feats, outputs = self.model(signals)
            _, preds = torch.max(outputs.data, 1)
            feats_list.append(feats.detach().cpu())
            preds_list.append(preds.detach().cpu())
            labels_list.append(labels)

        feats_list = torch.cat(feats_list, 0)
        preds_list = torch.cat(preds_list, 0)
        labels_list = torch.cat(labels_list, 0)

        acc = accuracy_score(labels_list, preds_list)
        note += '-acc-{:.2f}'.format(acc * 100)

        print('overall accuracy = {}'.format(acc))

        plot_confusion_matrix(labels=labels_list,
                              preds=preds_list,
                              note=note,
                              save_dir=save_dir,
                              normalize=normalize,
                              dpi=dpi
                              )
        plot_TSNE(feats_list, labels_list, save_dir, note, dpi, tsne_cmap)

    @torch.no_grad()
    def plot_pred_map(self, note='', activation='sigmoid', pp='average'):
        print('Activation Function: {}'.format(activation))
        print('Post Processing: {}'.format(pp))
        if not note:
            note = self.ckpt.split('checkpoints/')[1].split('-')[0]
        thresholds = np.arange(0.0, 1.0, 0.02)
        res = {}
        # save_dir = self.ckpt.split('checkpoints')[0]
        save_dir = 'pred_maps/'
        preds_list = []
        labels_list = []
        for batch in tqdm(self.test_loader):
            signals = batch['feature'].cuda()
            labels = batch['one_hot_label']
            path = batch['path']
            # labels = labels.repeat_interleave(signals.shape[1],dim=0)  # repeat labels to match length
            signals = signals.reshape(-1, signals.shape[-1])  # fold signals

            feats, outputs = self.model(signals)

            if activation == 'sigmoid':
                preds = torch.sigmoid(outputs)
            elif activation == 'softmax':
                preds = F.softmax(outputs, dim=-1)
            else:
                raise NotImplementedError

            plot_prediction_map(preds, labels, path, save_dir)

            if pp == 'average':
                preds = torch.mean(preds, dim=0, keepdim=True)

            elif pp == 'maxpool':
                preds = preds.transpose(0, 1)
                preds = F.max_pool1d(preds, 6, 3)
                preds = preds.transpose(0, 1)
                preds = torch.sum(preds, dim=0, keepdim=True)
                preds = (preds - preds.min()) / (preds.max() - preds.min())

            elif pp == 'norm':
                print('hey')
            else:
                raise NotImplementedError

            # preds = preds / preds.amax()

            preds_list.append(preds.detach().cpu())
            labels_list.append(labels)

        preds_list = torch.cat(preds_list, 0)
        labels_list = torch.cat(labels_list, 0)

        for i, th in enumerate(thresholds):
            predictions = self.apply_threshold(preds=preds_list, threshold=th)
            results = self.compute_score(preds=predictions, labels=labels_list)
            results.update({'threshold': "{:.2f}".format(th)})
            res[i] = results

        res = pd.DataFrame.from_dict(res, orient='index', columns=res[0].keys())
        best_f1 = res['f1_score_micro'].max()
        # best_f1 = 'lol'
        if note:
            note += '-{}-{}-'.format(activation, pp)
        save_name = save_dir + note + 'evaluation_f1={}'.format(best_f1) + '.csv'
        res.to_csv(save_name)

    def prepare_model(self):
        if self.ckpt:
            checkpoint = torch.load(self.ckpt)
            self.model.load_state_dict(checkpoint['model'])
        self.model = self.model.to(self.device)
        self.model.eval()

    def get_valid_data(self):
        valid_dataset = IRMASDataset(meta_path=self.valid_meta_path,
                                     wav_dir=self.valid_data_dir,
                                     normalize_amp=True,
                                     )
        valid_loader = DataLoader(dataset=valid_dataset,
                                  batch_size=512,  # fix the valid batch size
                                  shuffle=False,
                                  num_workers=8,
                                  pin_memory=False)
        return valid_loader

    def get_test_data(self):
        test_dataset = IRMASDataset(meta_path=self.test_meta_path,
                                    wav_dir=self.test_data_dir,
                                    is_test=True,
                                    normalize_amp=True,
                                    )
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=8
                                 )
        return test_loader

    def compute_score(self, preds, labels):
        metrics = {}
        for avg in ['micro', 'macro']:
            f1 = f1_score(labels, preds, average=avg)
            p = precision_score(labels, preds, average=avg)
            r = recall_score(labels, preds, average=avg)
            metrics.update({
                'precision_{}'.format(avg): round(p, 3),
                'recall_{}'.format(avg): round(r, 3),
                'f1_score_{}'.format(avg): round(f1, 3)
            })

        return metrics

    def apply_threshold(self, preds, threshold):
        if isinstance(threshold, np.floating):
            preds = torch.where(preds >= threshold, 1, 0)
        else:
            print(type(threshold))
            raise NotImplementedError
        return preds

    def load_or_create(self, res_dir):
        res_path = os.path.join(res_dir, 'results.csv')
        if not os.path.exists(res_path):
            df = pd.DataFrame()
        else:
            df = pd.read_csv(res_path)

        return df, res_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--ckpt_dir', type=str)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--mode', type=str, default='eval')
    parser.add_argument('--act', type=str, default='sigmoid')
    parser.add_argument('--pp', type=str, default='average')
    parser.add_argument('--tem', type=float, default=1.0)
    parser.add_argument('--seg-dur', type=int, default=1)

    parser.add_argument('--add_up', action='store_true')
    parser.add_argument('--no-add_up', dest='add_up', action='store_false')
    parser.set_defaults(add_up=False)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()

    config = {
        'model': '',
        'ckpt': '',
        'valid_meta_path': 'metadata/irmas_valid_full_shuffle_slice.json',
        'valid_data_dir': 'irmas_data/IRMAS-TrainingData/',
        'test_meta_path': 'metadata/irmas_test.json',
        'test_data_dir': 'irmas_data/',
    }
    if args.model:
        config['model'] = args.model

    if args.ckpt_dir:
        ckpts = glob.glob(os.path.join(args.ckpt_dir, '**/*.pth'), recursive=True)
        for ckpt in ckpts:
            # if not ('best' in ckpt or 'last' in ckpt):
            #     continue
            print('currently evaluating: {}'.format(ckpt))
            config['ckpt'] = ckpt
            e = Evaluator(**config)
            if args.mode == 'eval':
                e.evaluate_model()
            elif args.mode == 'plot':
                e.plot()
            elif args.mode == 'test':
                e.test(activation=args.act,
                       pp=args.pp,
                       tem=args.tem,
                       seg_dur=args.seg_dur,
                       add_up=args.add_up,
                       )
            else:
                raise NotImplementedError

        exit()

    elif args.ckpt:
        config['ckpt'] = args.ckpt

    e = Evaluator(**config)
    if args.mode == 'eval':
        e.evaluate_model()
    elif args.mode == 'plot':
        e.plot()
    elif args.mode == 'test':
        e.test(activation=args.act,
               pp=args.pp,
               tem=args.tem,
               seg_dur=args.seg_dur,
               add_up=args.add_up,
               )
    elif args.mode == 'plot_pred':
        e.plot_pred_map()
    else:
        raise NotImplementedError
