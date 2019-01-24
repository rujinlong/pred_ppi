#!/usr/bin/env python3
# Group 5: Jinlong, Julian, Reza, Chris
# USAGE: python3 pred_ppi.py -i test.fasta -o results.txt

import os
from Bio import SeqIO
import gensim
import pickle
import numpy as np
from argparse import ArgumentParser


def aa_represent_strs(seq, swindow):
    """
    Represent a single amino acid using it and it's neighbours.
    AA location is 0 based.
    """
    sub_strs = []
    for i in range(len(seq)-swindow+1):
        sub_str = seq[i:i + swindow]
        if len(sub_str) == swindow:
            sub_strs.append(sub_str)
    return sub_strs


def aa_represent_vecs(sub_strs, model, n_gram):
    """Turn 'ABCDEFG' to vectors of shape (1,100)
    """
    sub_vecs = []
    for v in sub_strs:
        tokens = [v[i:i+n_gram]
                  for i in range(len(v)) if len(v[i:i+n_gram]) == n_gram]
        sub_vecs.append(np.sum([np.array(model.wv[v])
                                for v in tokens], axis=0))
    return np.array(sub_vecs)


def predict(seq_id, seq, pred_scores, swindow, threshold):
    z = [">" + seq_id]
    for idx, x in enumerate(pred_scores):
        idx += int((swindow + 1)/2 - 1)  # 0-based
        x = round(x, 2)
        if x > threshold:
            pred = [seq[idx], "+", str(x)]
        else:
            pred = [seq[idx], "-", str(x)]
        z.append("\t".join(pred))
    return z


def main(file_fa, file_w2v, file_clf, file_out, n_gram, swindow=7, threshold=0.5):

    clf = pickle.load(open(file_clf, "rb"))
    w2v_model = gensim.models.KeyedVectors.load(file_w2v)
    for fa in SeqIO.parse(file_fa, "fasta"):
        seq_id = fa.description
        seq = str(fa.seq)
        # print(seq_id)
        sub_strs = aa_represent_strs(seq, swindow=swindow)
        sub_vecs = aa_represent_vecs(sub_strs, w2v_model, n_gram=n_gram)
        y_proba = clf.best_estimator_.predict_proba(sub_vecs)
        y_scores = y_proba[:, 1]
        result = predict(seq_id, seq, y_scores, swindow, threshold)

        with open(file_out, "a") as fh:
            fh.write("\n".join(result))
            fh.write("\n")


if __name__ == "__main__":
    arg_parser = ArgumentParser(
        description='Predict protein-protein interaction sites in protein sequence.\n example: python3 pred_ppi.py -i test.fasta')

    arg_parser.add_argument(
        '-i',
        '--in',
        dest='fasta_file',
        required=True,
        help='FASTA file with one or multiple protein sequences.')
    arg_parser.add_argument(
        '-w',
        '--w2v-model',
        dest='w2v_model',
        default='uniprot_sprot_80_n3_m2_k5_s100_w5.w2v',
        help='Pre-trained word2vec model.')
    arg_parser.add_argument(
        '-c',
        '--classifier',
        dest='clf_model',
        default='clf.pkl',
        help='The classifier model.')
    arg_parser.add_argument(
        '-o',
        '--out',
        dest='out_file',
        default='results.txt',
        help='Save predict results to this file.')
    arg_parser.add_argument(
        "-n",
        "--n-gram",
        dest="n_gram",
        default=3,
        type=int,
        help='ngram_size when training the word2vec model.'
    )
    arg_parser.add_argument(
        "-s",
        "--sliding-window",
        dest="swindow",
        default=7,
        type=int,
        help='The sliding window size when representing each amino acid using \
        the sliding window.'
    )
    arg_parser.add_argument(
        "-t",
        "--threshold",
        dest="threshold",
        default=0.37,
        type=float,
        help='Decision threshold to predict a sample as positive or negative.\
        Default = 0.37'
    )

    args = arg_parser.parse_args()

    file_fa = args.fasta_file
    file_w2v = args.w2v_model
    file_clf = args.clf_model
    file_out = args.out_file
    n_gram = args.n_gram
    swindow = args.swindow
    threshold = args.threshold

    dir_out = os.path.abspath(os.path.dirname(file_out))
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    if os.path.exists(file_out):
        print("File {} exists, please set another name.".format(file_out))
    else:
        main(file_fa, file_w2v, file_clf, file_out, n_gram, swindow, threshold)
