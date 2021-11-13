import argparse
import json
import os
import numpy as np
import time
import json
import tensorflow as tf

from encoder import get_encoder

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='aMLP-base-ja')
parser.add_argument('--context', type=str, required=True)
parser.add_argument('--output', type=str, default='')

def main():
    args = parser.parse_args()

    assert os.path.exists(os.path.join(args.model, 'hparams.json')), f"hparams.json not found in {args.model}"
    with open(os.path.join(args.model, 'hparams.json')) as f:
        conf_dict = json.load(f)
    vocab_size = conf_dict["num_vocab"]
    EOT_TOKEN = vocab_size - 1
    MASK_TOKEN = vocab_size - 2
    PAD_TOKEN = vocab_size - 3
    CLS_TOKEN = vocab_size - 4
    max_seq_length = conf_dict["num_ctx"]

    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        print("Running on TPU:", tpu.master())
    except ValueError:
        strategy = tf.distribute.get_strategy()
    print(f"Running on {strategy.num_replicas_in_sync} replicas")

    with strategy.scope():
        assert os.path.exists(args.model), f"Model file not found [{args.model}]."
        bpe_path = os.path.join(args.model, "vocabulary.txt")
        assert os.path.exists(bpe_path), f"vocabulary file not found in {bpe_path}."
        assert os.path.exists('emoji.json'), f"emoji file not found."

        with open(bpe_path) as f:
            ww = np.sum([1 if ('##' in l) else 0 for l in f.readlines()]) > 0
        enc = get_encoder(bpe_path, 'emoji.json', ww)

        print(f"Load Model from {args.model}")
        model = tf.keras.models.load_model(args.model, custom_objects={'loss': tf.keras.losses.Loss()})

        contexts = [args.context]
        _input_ids = []
        _input_weights = []
        for context in contexts:
            context_tokens = [enc.encode(c)+[MASK_TOKEN] for c in context.split('[MASK]')]
            context_tokens = sum(context_tokens, [])
            if len(context_tokens) > 1:
                context_tokens = context_tokens[:-1]
            context_tokens = context_tokens[:max_seq_length-2]
            inputs = []
            weights = []
            inputs.append(CLS_TOKEN)
            inputs.extend(context_tokens)
            inputs.append(EOT_TOKEN)
            weights.extend([1.0]*len(inputs))
            while len(inputs) < max_seq_length:
                inputs.append(PAD_TOKEN)
                weights.append(0.0)
            _input_ids.append(inputs)
            _input_weights.append(weights)

        _input_ids = np.array(_input_ids, dtype=np.int32)
        _input_weights = np.array(_input_weights, dtype=np.float32)
        out, _ = model(inputs=[_input_ids, _input_weights])
        if len(args.output) > 0:
            with open(args.output, "w") as wf:
                wf.write(','.join(list(map(str,out[0][0].numpy().tolist()))))
        else:
            print(out[0][0].numpy().tolist())



if __name__ == '__main__':
    main()
