#!/usr/bin/env python3

from copy import copy
import fire
from tqdm import tqdm
import json
import os
import numpy as np
import tensorflow as tf

import model, sample, encoder

def extract_encoding(
    sentences_file,
    out_file,
    model_name='117M',
    seed=None,
    length=None,
):
    batch_size = 1
    np.random.seed(seed)
    tf.set_random_seed(seed)

    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    with open(sentences_file, "r") as sentences_f:
        sentences = [line.strip() for line in sentences_f]

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)
    print("length", length)

    encodings = []

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        output = model.model(hparams=hparams, X=context,
                             past=None, reuse=tf.AUTO_REUSE)["present"]

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
        saver.restore(sess, ckpt)

        for sentence in tqdm(sentences):
            batch_tokens = [enc.encode(sentence)]
            out = sess.run(output, feed_dict={context: batch_tokens})

            # Extract last layer, averaging word embeddings.
            # shape is `batch * n_layers * 2 * n_heads * seq_length * embedding_dim`
            out = out[0, -1].mean(axis=(0, 2)).flatten()
            encodings.append(out)

    encodings = np.concatenate(encodings)
    np.save(out_file, encodings)


if __name__ == '__main__':
    fire.Fire(extract_encoding)

