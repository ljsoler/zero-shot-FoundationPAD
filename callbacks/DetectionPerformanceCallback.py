import pytorch_lightning as pl
import torch
import os
import numpy as np
from pyeer.eer_info import get_eer_stats
from collections import defaultdict

class DetectionPerformanceCallback(pl.Callback):
    def __init__(self, data_module, output_folder):
        super().__init__()
        self.data_module = data_module
        self.output_folder = output_folder

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.sanity_checking:
            gen, imp = [], []
            for batch in self.data_module:
                x, _, l = batch
                x = x.to(device=pl_module.device)
                l = l.to(device=pl_module.device)
                prob = pl_module(x)
                prob = torch.squeeze(prob, dim=1)
                prob = prob.detach().cpu().numpy()
                label = l.detach().cpu().numpy()
                for i in range(label.shape[0]):
                    if label[i] == 1.0:
                        gen.append(prob[i])
                    else:
                        imp.append(prob[i])
            
            stat = get_eer_stats(gen, imp)
            eer = torch.tensor(stat.eer*100, dtype=torch.float)
            fmr10 = torch.tensor(stat.fmr10*100, dtype=torch.float)
            fmr20 = torch.tensor(stat.fmr20*100, dtype=torch.float)
            fmr100 = torch.tensor(stat.fmr100*100, dtype=torch.float)

            pl_module.log_dict({'D-EER': eer, 'BPCER10': fmr10, 'BPCER20': fmr20, 'BPCER100': fmr100})

    
    def on_test_epoch_end(self, trainer, pl_module):
        gen, imp = [], []
        video_predictions = defaultdict(list)
        labels = {}
        
        for batch in self.data_module:
            x, videos_indices, l = batch
            x = x.to(device=pl_module.device)
            l = l.to(device=pl_module.device)
            prob = pl_module(x)
            prob = torch.squeeze(prob, dim=1)
            prediction = prob.detach().cpu().numpy()
            label = l.detach().cpu().numpy()
            videos_indices = videos_indices.detach().cpu().numpy()

            # Distribute predictions to corresponding videos
            for pred, video_idx, lidx in zip(prediction, videos_indices, label):
                video_predictions[video_idx].append(pred)
                if not video_idx in labels:
                    labels[video_idx] = lidx

            for i in range(label.shape[0]):
                if label[i] == 1.0:
                    gen.append(prediction[i])
                else:
                    imp.append(prediction[i])
        
        stat = get_eer_stats(gen, imp)
        eer = torch.tensor(stat.eer*100, dtype=torch.float)
        fmr10 = torch.tensor(stat.fmr10*100, dtype=torch.float)
        fmr20 = torch.tensor(stat.fmr20*100, dtype=torch.float)
        fmr100 = torch.tensor(stat.fmr100*100, dtype=torch.float)

        pl_module.log_dict({'D-EER_Frames': eer, 'BPCER10_Frames': fmr10, 'BPCER20_Frames': fmr20, 'BPCER100_Frames': fmr100})

        # save the results to a file
        score_path = os.path.join(self.output_folder, 'comparisons_frames.npz')
        np.savez(score_path, gen=gen, imp=imp)

        # Average predictions for each video
        avg_predictions = {k:np.mean(pred) for k, pred in video_predictions.items()}
        ap_scores, bp_scores = [], []

        mean, std = np.mean(list(avg_predictions.values())), np.std(list(avg_predictions.values()))

        for k, s in avg_predictions.items():
            label = labels[k]
            score = (s - mean)/std
            if label == 0:
                ap_scores.append(score)
            else:
                bp_scores.append(score) 

        stat = get_eer_stats(bp_scores, ap_scores)
        eer = torch.tensor(stat.eer*100, dtype=torch.float)
        fmr10 = torch.tensor(stat.fmr10*100, dtype=torch.float)
        fmr20 = torch.tensor(stat.fmr20*100, dtype=torch.float)
        fmr100 = torch.tensor(stat.fmr100*100, dtype=torch.float)

        pl_module.log_dict({'D-EER_Videos': eer, 'BPCER10_Videos': fmr10, 'BPCER20_Videos': fmr20, 'BPCER100_Videos': fmr100})

        # save the results to a file
        score_path = os.path.join(self.output_folder, 'comparisons_videos.npz')
        np.savez(score_path, gen=bp_scores, imp=ap_scores)
