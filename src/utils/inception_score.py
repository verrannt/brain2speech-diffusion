"""
The function `calculate_inception_score()` is taken from:
https://machinelearningmastery.com/how-to-implement-the-inception-score-from-scratch-for-evaluating-generated-images/
"""

import librosa
import pickle
from tqdm import tqdm
import numpy as np


class InceptionScore:
    def __init__(self, clf_path, audio_load_workers=8, verbose=False):
        with open(clf_path, "rb") as f:
            self.clf = pickle.load(f)
        self.audio_load_workers = audio_load_workers
        self.verbose = verbose

    def load_audio_files_from_list(self, file_list):
        res_list = [self.get_mfcc(fn) for fn in tqdm(file_list, disable=not self.verbose)]
        return res_list

    @classmethod
    def get_mfcc(self, audio_file):
        samples, sample_rate = librosa.load(audio_file, sr=None)
        mfcc = librosa.feature.mfcc(y=samples, sr=sample_rate, hop_length=1024, n_mfcc=40)
        return mfcc

    @classmethod
    def calculate_inception_score(self, p_yx, eps=1e-16):
        """Calculate the Inception Score for p(y|x)"""
        # calculate p(y)
        p_y = np.expand_dims(p_yx.mean(axis=0), 0)
        # kl divergence for each image
        kl_d = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))
        # sum over classes
        sum_kl_d = kl_d.sum(axis=1)
        # average over images
        avg_kl_d = np.mean(sum_kl_d)
        # undo the logs
        score = np.exp(avg_kl_d)
        return score

    def __call__(self, files):
        mfccs = self.load_audio_files_from_list(files)
        mfccs = np.array(mfccs).reshape(len(mfccs), -1)
        probas = self.clf.predict_proba(mfccs)
        inception_score = self.calculate_inception_score(probas)

        return inception_score
