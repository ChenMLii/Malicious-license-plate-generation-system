import numpy as np
import torch
from data.load_data import CHARS


class DataUtils:

    @staticmethod
    def collate_fn(batch):
        imgs = []
        labels = []
        lengths = []
        for _, sample in enumerate(batch):
            img, label, length = sample
            imgs.append(torch.from_numpy(img))
            labels.extend(label)
            lengths.append(length)
        labels = np.asarray(labels).flatten().astype(np.float32)

        return (torch.stack(imgs, 0), torch.from_numpy(labels), lengths)

    @staticmethod
    def greedy_decoder(prebs):
        # greedy decode
        prebs = prebs.cpu().detach().numpy()
        preb_labels = list()
        scores = dict()
        for i in range(prebs.shape[0]):
            preb = prebs[i, :, :]
            preb_label = list()
            for j in range(preb.shape[1]):
                preb_label.append(np.argmax(preb[:, j], axis=0))
                scores[np.argmax(preb[:, j], axis=0)] = preb[:, j].tolist()
            no_repeat_blank_label = list()
            score = list()
            pre_c = preb_label[0]
            if pre_c != len(CHARS) - 1:
                no_repeat_blank_label.append(pre_c)
                score.append(scores[pre_c])
            for c in preb_label:  # dropout repeate label and blank label
                if (pre_c == c) or (c == len(CHARS) - 1):
                    if c == len(CHARS) - 1:
                        pre_c = c
                    continue
                no_repeat_blank_label.append(c)
                score.append(scores[c])
                pre_c = c
            preb_labels.append(no_repeat_blank_label)
            
        return no_repeat_blank_label
    
    @staticmethod
    def greedy_decoder_get_scores(prebs):
        # greedy decode
        prebs = prebs.cpu().detach().numpy()
        preb_labels = list()
        scores = dict()
        for i in range(prebs.shape[0]):
            preb = prebs[i, :, :]
            preb_label = list()
            for j in range(preb.shape[1]):
                preb_label.append(np.argmax(preb[:, j], axis=0))
                scores[np.argmax(preb[:, j], axis=0)] = preb[:, j].tolist()
            no_repeat_blank_label = list()
            score = list()
            pre_c = preb_label[0]
            if pre_c != len(CHARS) - 1:
                no_repeat_blank_label.append(pre_c)
                score.append(scores[pre_c])
            for c in preb_label:  # dropout repeate label and blank label
                if (pre_c == c) or (c == len(CHARS) - 1):
                    if c == len(CHARS) - 1:
                        pre_c = c
                    continue
                no_repeat_blank_label.append(c)
                score.append(scores[c])
                pre_c = c
            preb_labels.append(no_repeat_blank_label)
            
        return score
    
    @staticmethod
    def save_to_file(path, filename):
        pass
