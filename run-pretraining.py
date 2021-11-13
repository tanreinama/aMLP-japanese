import argparse
import json
import os
import numpy as np
import time
import shutil
from copy import copy
import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam, SGD

from modeling import model as build_model

CHECKPOINT_DIR = 'checkpoint'

parser = argparse.ArgumentParser(
    description='Pretraining MLP on your custom dataset.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dataset', metavar='PATH', type=str, required=True, help='Input npz file')
parser.add_argument('--vocabulary', metavar='PATH', type=str, required=True, help='Encode vocabulary file')
parser.add_argument('--mlm_whole_word', action='store_true', help='whole word mode')
parser.add_argument('--restore_from', type=str, default='', help='Partial training weight directory. Need includes model.npz and opt.npz.')
parser.add_argument('--restore_train_chunks', action='store_true', help='Restore partial training data. Need same dataset to resume training.')
parser.add_argument('--model_size', type=str, default='base', help='Model size (pico/tiny/small/medium/base/large/xlarge).')
parser.add_argument('--model_type', type=str, default='mlp', help='Model type (mlp/transformer).')
parser.add_argument('--max_context', type=int, default=512, help='maximin context size.')

parser.add_argument('--batch_size', metavar='SIZE', type=int, default=4, help='Batch size')
parser.add_argument('--log_dir', type=str, default='', help='Directory to save log.')
parser.add_argument('--run_name', type=str, default='aMLP-base-ja', help='Run id. Name of subdirectory in checkpoint/')
parser.add_argument('--save_every', metavar='N', type=int, default=335000, help='Write a checkpoint every N steps')
parser.add_argument('--num_epochs', metavar='N', type=float, default=-1, help='Maximum training epochs.')
parser.add_argument('--learning_rate', type=float, default=7e-4, help='Initial learning rate for Adam.')
parser.add_argument('--lr_dense_each', type=int, default=500000, help='learning rate dense schedul.')
parser.add_argument('--lr_dense_rate', type=float, default=2, help='learning rate dense rage.')
parser.add_argument('--min_lr', type=float, default=2e-6, help='learning rate dense minimum value.')
parser.add_argument('--num_warmup_step', type=int, default=10000, help='warming up steps.')
parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer name(adam / sgd).')

def main():
    args = parser.parse_args()
    hid, att = (768,128) if "large" in args.model_size else (512,64)
    lyr = [3,9,18,24,36,72,96][["pico","tiny","small","medium","base","large","xlarge"].index(args.model_size)]
    typ = args.model_type
    if typ=="transformer":
        lyr //= 3
    assert os.path.isfile(args.vocabulary), f"Vocabulary file not found in {args.vocabulary}."
    with open(args.vocabulary) as f:
        vocaburalys = f.read().split('\n')

    conf_dict = {
        "num_ctx":args.max_context,
        "num_vocab":len(vocaburalys),
        "num_hidden":hid,
        "num_soft_att":att,
        "num_layer":lyr,
        "dropout_prob":0.02,
        "mlp_dropout_prob":0.02,
        "type":typ
    }

    vocab_size = conf_dict["num_vocab"]
    EOT_TOKEN = vocab_size - 1
    MASK_TOKEN = vocab_size - 2
    PAD_TOKEN = vocab_size - 3
    CLS_TOKEN = vocab_size - 4
    batch_size = args.batch_size
    max_seq_length = conf_dict["num_ctx"]
    max_predictions = int(np.round(max_seq_length*0.015)+1)*10
    log_dir = args.log_dir

    if log_dir != '':
        os.makedirs(log_dir, exist_ok=True)
        if args.restore_train_chunks and os.path.isfile(os.path.join(log_dir,"log.csv")) and os.path.isfile(os.path.join(log_dir,"log.txt")):
            log_csv = open(os.path.join(log_dir,"log.csv"), "a")
            log_text = open(os.path.join(log_dir,"log.txt"), "a")
        else:
            log_csv = open(os.path.join(log_dir,"log.csv"), "w")
            log_text = open(os.path.join(log_dir,"log.txt"), "w")
            log_csv.write("counter,epoch,time,loss,avg\n")
            log_text.write(f"args:{vars(args)}\n")
    os.makedirs(os.path.join(CHECKPOINT_DIR,args.run_name), exist_ok=True)
    with open(os.path.join(CHECKPOINT_DIR,args.run_name,"hparams.json"), "w") as wf:
        wf.write(json.dumps(conf_dict))
    shutil.copy(args.vocabulary, os.path.join(CHECKPOINT_DIR,args.run_name,"vocabulary.txt"))

    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        print("Running on TPU:", tpu.master())
    except ValueError:
        strategy = tf.distribute.get_strategy()
    print(f"Running on {strategy.num_replicas_in_sync} replicas")

    def masked_crossentropy(labels, logits):
        num_vocabrary = logits.shape.as_list()[-1]
        flat_labels = tf.reshape(labels, [-1])
        flat_logits = tf.reshape(logits, [-1, num_vocabrary])
        mask = flat_labels+1
        mask = tf.cast(tf.cast(mask, tf.bool), tf.int32)
        flat_labels = tf.cast(flat_labels + (1 - mask), tf.int32)
        one_hot_labels = tf.one_hot(flat_labels, depth=num_vocabrary, dtype=tf.float32)
        log_probs = tf.nn.log_softmax(flat_logits)
        loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
        mask = tf.cast(mask, tf.float32)
        loss = tf.multiply(loss, mask)
        loss = tf.reduce_sum(loss) / (tf.reduce_sum(mask) + 1e-12)
        return loss

    class train_model(tf.keras.Model):
        def __init__(self, model):
            super(train_model, self).__init__(name='train_model')
            self.model = model
        def call(self, inputs):
            input_ids, input_weights, masked_lm_positions = inputs
            _, lm_output = self.model(inputs=[input_ids, input_weights])
            return tf.gather(lm_output, masked_lm_positions, axis=1, batch_dims=1)

    with strategy.scope():
        counter = 1

        model = build_model(**conf_dict)
        lossmodel = train_model(model)
        if args.optimizer == 'adam':
            opt = Adam(learning_rate=args.learning_rate,
                       beta_1=0.9,
                       beta_2=0.98,
                       epsilon=1e-6,
                       amsgrad=True)
        else:
            opt = SGD(learning_rate=args.learning_rate)
        lossmodel.compile(optimizer=opt, loss=masked_crossentropy)
        lossmodel.optimizer.apply_gradients(zip([tf.zeros_like(x) for x in model.weights], model.weights))

        if args.restore_from!='':
            print(f"Load Model from {args.restore_from}")
            # tf.keras.save_weights not work insted tf.gather so loas weights as numpy array
            fn = os.path.join(args.restore_from, 'model.npz')
            w = np.load(fn)
            lossmodel.set_weights([w[f] for f in w.files])
            fn = os.path.join(args.restore_from, 'opt.npz')
            w = np.load(fn)
            lossmodel.optimizer.set_weights([w[f] for f in w.files])

        print('Loading dataset...')
        global_chunks = np.load(args.dataset)
        global_chunk_index = copy(global_chunks.files)
        global_chunk_step = 0
        global_epochs = 0
        np.random.shuffle(global_chunk_index)

        if args.restore_from!='' and args.restore_train_chunks:
            with open(os.path.join(args.restore_from, 'partial.json')) as f:
                w = json.loads(f.read())
            global_chunk_index = w['global_chunk_index']
            global_chunk_step = w['global_chunk_step']
            global_epochs = w['global_epochs']
            counter = w['counter']
            args.learning_rate = w['learning_rate']
            backend.set_value(opt.lr, args.learning_rate)

        def get_epoch():
            return global_epochs + (1 - len(global_chunk_index) / len(global_chunks.files))

        def pop_feature(sample_seq_length):
            nonlocal global_chunks,global_chunk_index,global_chunk_step, global_epochs
            token = [np.uint16(CLS_TOKEN)]
            weight = [1.0]
            chunk = global_chunks[global_chunk_index[-1]].astype(np.uint16)
            while len(token) < sample_seq_length and global_chunk_step < len(chunk):
                if chunk[global_chunk_step] < CLS_TOKEN:
                    token.append(chunk[global_chunk_step])
                    weight.append(1.0)
                global_chunk_step += 1
            if global_chunk_step >= len(chunk):
                index = global_chunk_index.pop()
                global_chunk_step = 0
                if len(global_chunk_index) == 0:
                    global_chunk_index = copy(global_chunks.files)
                    np.random.shuffle(global_chunk_index)
                    global_epochs += 1
            if len(token) < sample_seq_length and token[-1] != EOT_TOKEN:
                token.append(np.uint16(EOT_TOKEN))
                weight.append(1.0)
            while len(token) < sample_seq_length:
                token.append(np.uint16(PAD_TOKEN))
                weight.append(0.0)
            return token, weight

        def sample_feature_full():
            nonlocal global_chunks,global_chunk_index,global_chunk_step
            # Use dynamic mask
            p_input_ids = []
            p_input_weights = []
            p_masked_lm_positions = []
            p_masked_lm_ids = []

            for b in range(batch_size):
                while True:
                    # Make Sequence
                    ids, weights = pop_feature(max_seq_length)
                    # Make Masks
                    mask_indexs = [i for i in range(max_seq_length) if ids[i] < CLS_TOKEN]
                    np.random.shuffle(mask_indexs)
                    mask_indexs = mask_indexs[:max_predictions]

                    lm_positions = []
                    lm_ids = []
                    for i in sorted(mask_indexs):
                        masked_token = None
                        # 80% of the time, replace with [MASK]
                        if np.random.random() < 0.8:
                            masked_token = MASK_TOKEN # [MASK]
                        else:
                            # 10% of the time, keep original
                            if np.random.random() < 0.5:
                                masked_token = ids[i]
                            # 10% of the time, replace with random word
                            else:
                                masked_token = np.random.randint(CLS_TOKEN-1)

                        target = int(ids[i])
                        lm_positions.append(i)
                        lm_ids.append([target])
                        # apply mask
                        ids[i] = masked_token

                    if len(lm_positions) == 0:
                        continue

                    while len(lm_positions) < max_predictions:
                        lm_positions.append(0)
                        lm_ids.append([-1])
                    lm_positions = lm_positions[:max_predictions]
                    lm_ids = lm_ids[:max_predictions]

                    break

                p_input_ids.append(ids)
                p_input_weights.append(weights)
                p_masked_lm_positions.append(lm_positions)
                p_masked_lm_ids.append(lm_ids)

            p_input_ids = np.array(p_input_ids, dtype=np.int32)
            p_input_weights = np.array(p_input_weights, dtype=np.float32)
            p_masked_lm_positions = np.array(p_masked_lm_positions, dtype=np.int32)
            p_masked_lm_ids = np.array(p_masked_lm_ids, dtype=np.int32)
            return [p_input_ids,p_input_weights,p_masked_lm_positions], p_masked_lm_ids

        def sample_feature_word():
            nonlocal global_chunks,global_chunk_index,global_chunk_step
            # Use dynamic mask
            p_input_ids = []
            p_input_weights = []
            p_masked_lm_positions = []
            p_masked_lm_ids = []

            for b in range(batch_size):
                while True:
                    # Make Sequence
                    ids, weights = pop_feature(max_seq_length)
                    # Start Words
                    swd = [not (len(vocaburalys[i])>2 and vocaburalys[i][0]=='#' and vocaburalys[i][1]=='#') for i,w in zip(ids,weights) if w!=0]
                    iwd = [i for i in range(len(swd)) if swd[i]] + [len(swd)]
                    iwd = [(iwd[i],iwd[i+1]) for i in range(len(iwd)-1)]
                    # Make Masks
                    mask_word_index = np.random.permutation(len(iwd))
                    lm_positions = []
                    lm_ids = []
                    for p in range(len(iwd)):
                        mi = iwd[mask_word_index[p]]
                        for i in range(mi[0], mi[1], 1):
                            masked_token = None
                            # 80% of the time, replace with [MASK]
                            if np.random.random() < 0.8:
                                masked_token = MASK_TOKEN # [MASK]
                            else:
                                # 10% of the time, keep original
                                if np.random.random() < 0.5:
                                    masked_token = ids[i]
                                # 10% of the time, replace with random word
                                else:
                                    masked_token = np.random.randint(CLS_TOKEN-1)

                            target = int(ids[i])
                            lm_positions.append(i)
                            lm_ids.append([target])
                            # apply mask
                            ids[i] = masked_token

                        if p < len(iwd)-1:
                            ni = iwd[mask_word_index[p+1]]
                            if len(lm_positions)+(ni[1]-ni[0]) > max_predictions:
                                break

                    if len(lm_positions) == 0:
                        continue

                    while len(lm_positions) < max_predictions:
                        lm_positions.append(0)
                        lm_ids.append([-1])
                    lm_positions = lm_positions[:max_predictions]
                    lm_ids = lm_ids[:max_predictions]

                    break

                p_input_ids.append(ids)
                p_input_weights.append(weights)
                p_masked_lm_positions.append(lm_positions)
                p_masked_lm_ids.append(lm_ids)

            p_input_ids = np.array(p_input_ids, dtype=np.int32)
            p_input_weights = np.array(p_input_weights, dtype=np.float32)
            p_masked_lm_positions = np.array(p_masked_lm_positions, dtype=np.int32)
            p_masked_lm_ids = np.array(p_masked_lm_ids, dtype=np.int32)
            return [p_input_ids,p_input_weights,p_masked_lm_positions], p_masked_lm_ids

        def sample_feature():
            return sample_feature_word() if args.mlm_whole_word else sample_feature_full()

        def save(last=False):
            if last:
                fn = os.path.join(CHECKPOINT_DIR, args.run_name)
                print('Saving model in',fn)
                model.save(fn, save_format="tf")
            fn = os.path.join(CHECKPOINT_DIR, args.run_name, 'restore-training', 'checkpoint-{}').format(counter)
            os.makedirs(fn, exist_ok=True)
            print('Saving partial weights in',fn)
            # tf.keras.save_weights not work insted tf.gather so save weights as numpy array
            fn = os.path.join(CHECKPOINT_DIR, args.run_name, 'restore-training', 'checkpoint-{}', 'model').format(counter)
            np.savez(fn, *lossmodel.get_weights())
            fn = os.path.join(CHECKPOINT_DIR, args.run_name, 'restore-training', 'checkpoint-{}', 'opt').format(counter)
            np.savez(fn, *opt.get_weights())
            fn = os.path.join(CHECKPOINT_DIR, args.run_name, 'restore-training', 'checkpoint-{}', 'partial.json').format(counter)
            with open(fn, 'w') as wf:
                wf.write(json.dumps({'global_chunk_index':global_chunk_index,'global_chunk_step':global_chunk_step,'global_epochs':global_epochs,'counter':counter,'learning_rate':args.learning_rate}))

        print('Training...')

        avg_loss = (0.0, 0.0)
        start_time = time.time()

        try:
            while True:
                if args.num_epochs > 0 and get_epoch() >= args.num_epochs:
                    save(True)
                    break

                X, y = sample_feature()

                v_loss = lossmodel.train_on_batch(X, y)

                avg_loss = (avg_loss[0] * 0.99 + v_loss,
                            avg_loss[1] * 0.99 + 1.0)
                log = '[{counter}step | {epoch:2.2f}epoch | {time:2.2f}sec] loss={loss:2.2f} avg={avg:2.2f}'.format(
                            counter=counter,
                            epoch=get_epoch(),
                            time=time.time() - start_time,
                            loss=v_loss,
                            avg=avg_loss[0] / avg_loss[1])
                print(log)
                if log_dir != '':
                    log_text.write(log+'\n')
                    log_text.flush()
                    log = '{counter},{epoch},{time},{loss},{avg}'.format(
                            counter=counter,
                            epoch=get_epoch(),
                            time=time.time() - start_time,
                            loss=v_loss,
                            avg=avg_loss[0] / avg_loss[1])
                    log_csv.write(log+'\n')
                    log_csv.flush()

                counter = counter+1
                if counter % args.save_every == 0:
                    save()
                if counter<=args.num_warmup_step:
                    backend.set_value(opt.lr, args.learning_rate * (counter / args.num_warmup_step))
                elif counter%args.lr_dense_each == 0 and args.learning_rate/args.lr_dense_rate>args.min_lr:
                    args.learning_rate /= args.lr_dense_rate
                    backend.set_value(opt.lr, args.learning_rate)
        except KeyboardInterrupt:
            print('interrupted')
            save(True)


if __name__ == '__main__':
    main()
