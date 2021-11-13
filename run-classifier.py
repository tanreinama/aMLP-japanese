import argparse
import json
import os
import numpy as np
import time
import shutil
from copy import copy
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam

from modeling import model as build_model
from modeling import fc

from encoder import get_encoder

CHECKPOINT_DIR = 'checkpoint'

parser = argparse.ArgumentParser(
    description='Clasification task.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dataset', metavar='PATH', type=str, default="", help='Training json file')
parser.add_argument('--val_size', metavar='PATH', type=int, default=1000, help='data size to calucrate training score')
parser.add_argument('--val_dataset', metavar='PATH', type=str, default="", help='Validation json file')
parser.add_argument('--pred_dataset', metavar='PATH', type=str, default="", help='Prediction json file')
parser.add_argument('--base_model', type=str, default='aMLP-base-ja', help='Base Model Name.')
parser.add_argument('--clean_text', action='store_true', help='clean text.')
parser.add_argument('--output', metavar='PATH', type=str, default="classifier-predicted.csv", help='Prediction csv file')

parser.add_argument('--batch_size', metavar='SIZE', type=int, default=4, help='Batch size')
parser.add_argument('--log_dir', type=str, default='', help='Directory to save log.')
parser.add_argument('--run_name', type=str, default='aMLP-classifier-ja', help='Run id. Name of subdirectory in checkpoint/')
parser.add_argument('--save_every', metavar='N', type=int, default=10000, help='Write a checkpoint every N steps')
parser.add_argument('--val_every', metavar='N', type=int, default=1000, help='Validate every N steps')
parser.add_argument('--num_epochs', metavar='N', type=float, default=5, help='Maximum training epochs.')
parser.add_argument('--learning_rate', type=float, default=1e-5, help='Training learning rate for Adam.')
parser.add_argument('--restore_from', type=str, default='', help='checkpoint name for restore training.')

def main():
    args = parser.parse_args()
    if len(args.restore_from)==0:
        assert os.path.isdir(args.base_model), f"model not found: {args.base_model}"
        bpe_path = os.path.join(args.base_model, "vocabulary.txt")
        hpm_path = os.path.join(args.base_model,"hparams.json")
        map_dict = None
    else:
        bpe_path = os.path.join(args.restore_from, "vocabulary.txt")
        hpm_path = os.path.join(args.restore_from,"hparams.json")
        map_path = os.path.join(args.restore_from,"idmap.json")
        assert os.path.exists(map_path), f"idmap file not found."
        with open(map_path) as f:
            map_dict = json.loads(f.read())
    assert len(args.restore_from)==0 or os.path.isdir(args.restore_from), f"checkpoint not found: {args.restore_from}"
    assert os.path.isfile(bpe_path), f"vocabulary.txt not found in {bpe_path}."
    assert os.path.isfile(hpm_path), f"hparams.json not found in {hpm_path}."
    assert os.path.exists('emoji.json'), f"emoji file not found."
    with open(hpm_path) as f:
        conf_dict = json.loads(f.read())

    assert len(args.dataset)==0 or os.path.isdir(args.dataset), f"training data not found: {args.dataset}"
    assert len(args.val_dataset)==0 or os.path.isdir(args.val_dataset), f"validation data not found: {args.val_dataset}"
    assert len(args.pred_dataset)==0 or os.path.isdir(args.pred_dataset), f"prediction data not found: {args.pred_dataset}"
    do_training = len(args.dataset) > 0 and os.path.isdir(args.dataset)
    do_validation = len(args.val_dataset) > 0 and os.path.isdir(args.val_dataset)
    do_prediction = len(args.pred_dataset) > 0 and os.path.isdir(args.pred_dataset)
    assert (do_training or do_prediction), "must run with training or prediction task"
    assert (do_training or map_dict is not None), "not found restored model on prediction task"

    vocab_size = conf_dict["num_vocab"]
    EOT_TOKEN = vocab_size - 1
    MASK_TOKEN = vocab_size - 2
    SEP_TOKEN = vocab_size - 3
    CLS_TOKEN = vocab_size - 4
    batch_size = args.batch_size
    max_seq_length = conf_dict["num_ctx"]
    max_predictions = 1
    log_dir = args.log_dir

    with open(bpe_path) as f:
        ww = np.sum([1 if ('##' in l) else 0 for l in f.readlines()]) > 0
    enc = get_encoder(bpe_path, 'emoji.json', ww)

    if log_dir != '':
        os.makedirs(log_dir, exist_ok=True)
        log_csv = open(os.path.join(log_dir,"log.csv"), "w")
        log_text = open(os.path.join(log_dir,"log.txt"), "w")
        log_csv.write("counter,epoch,time,loss,avg\n")
        log_text.write(f"args:{vars(args)}\n")
        if do_validation:
            val_csv = open(os.path.join(log_dir,"val.csv"), "w")
            val_csv.write("counter,epoch,validation_score\n")
    os.makedirs(os.path.join(CHECKPOINT_DIR,args.run_name), exist_ok=True)

    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        print("Running on TPU:", tpu.master())
    except ValueError:
        strategy = tf.distribute.get_strategy()
    print(f"Running on {strategy.num_replicas_in_sync} replicas")

    class classifier_model(tf.keras.Model):
        def __init__(self, model, class_num, conf_dict):
            super(classifier_model, self).__init__(name='classifier_model')
            self.model = model
            self.fc = fc(conf_dict["num_hidden"], class_num, name='output')
        def call(self, inputs):
            input_ids, input_weights = inputs
            lm_output, _ = self.model(inputs=[input_ids, input_weights])
            feat_output = lm_output[:,0,:]
            logits = self.fc(feat_output)
            return logits

    def crossentropy(labels, logits):
        num_vocabrary = logits.shape.as_list()[-1]
        flat_labels = tf.reshape(labels, [-1])
        flat_labels = tf.cast(flat_labels, tf.int32)
        flat_logits = tf.reshape(logits, [-1, num_vocabrary])
        one_hot_labels = tf.one_hot(flat_labels, depth=num_vocabrary, dtype=tf.float32)
        log_probs = tf.nn.log_softmax(flat_logits)
        loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
        loss = tf.reduce_mean(loss)
        return loss

    with strategy.scope():
        counter = 1

        print('Loading dataset...')
        def encode_data(input_dir, map_dct=None):
            if map_dct is None:
                M = {}
                datasets = []
                for ind,dir in enumerate(os.listdir(input_dir)):
                    if not os.path.isdir(os.path.join(input_dir,dir)):
                        continue
                    print(f"{dir} mapped index#{ind}")
                    M[dir] = ind
                    for file in os.listdir(os.path.join(input_dir,dir)):
                        with open(os.path.join(input_dir,dir,file)) as fp:
                            txt = fp.read()
                        context = enc.encode(txt, clean=args.clean_text)
                        tokens = [CLS_TOKEN] + context + [EOT_TOKEN]
                        tokens_weights = [1.0] * len(tokens)
                        while len(tokens) < max_seq_length:
                            tokens.append(EOT_TOKEN)
                            tokens_weights.append(0.0)
                        tokens = tokens[:max_seq_length]
                        tokens_weights = tokens_weights[:max_seq_length]
                        datasets.append({"tokens":tokens,"tokens_weights":tokens_weights,"target":ind,"filename":str(os.path.join(input_dir,dir,file))})
                return M, datasets
            else:
                datasets = []
                for dir in os.listdir(input_dir):
                    if not os.path.isdir(os.path.join(input_dir,dir)):
                        continue
                    if dir not in map_dct:
                        continue
                    ind = map_dct[dir]
                    for file in os.listdir(os.path.join(input_dir,dir)):
                        with open(os.path.join(input_dir,dir,file)) as fp:
                            txt = fp.read()
                        context = enc.encode(txt, clean=args.clean_text)
                        tokens = [CLS_TOKEN] + context + [EOT_TOKEN]
                        tokens_weights = [1.0] * len(tokens)
                        while len(tokens) < max_seq_length:
                            tokens.append(EOT_TOKEN)
                            tokens_weights.append(0.0)
                        tokens = tokens[:max_seq_length]
                        tokens_weights = tokens_weights[:max_seq_length]
                        datasets.append({"tokens":tokens,"tokens_weights":tokens_weights,"target":ind,"filename":str(os.path.join(input_dir,dir,file))})
                return datasets

        if map_dict is None:
            map_dict, global_chunks = encode_data(args.dataset) if do_training else (None, None)
        else:
            global_chunks = encode_data(args.dataset, map_dict) if do_training else None

        global_chunk_index = 0
        global_epochs = 0

        if do_training:
            np.random.shuffle(global_chunks)

        validation_chunks = encode_data(args.val_dataset, map_dict) if do_validation else None
        prediction_chunks = encode_data(args.pred_dataset, map_dict) if do_prediction else None

        if len(args.restore_from) > 0:
            lossmodel = tf.keras.models.load_model(args.restore_from, \
                    custom_objects={'crossentropy': crossentropy})
        else:
            model = tf.keras.models.load_model(args.base_model, \
                    custom_objects={'loss': tf.keras.losses.Loss()})
            lossmodel = classifier_model(model, len(map_dict), conf_dict)
            opt = Adam(learning_rate=args.learning_rate)
            lossmodel.compile(optimizer=opt, loss=crossentropy)
            lossmodel.optimizer.apply_gradients(zip([tf.zeros_like(x) for x in model.weights], model.weights))


        def get_epoch():
            return global_epochs + (global_chunk_index / len(global_chunks))

        def sample_feature():
            nonlocal global_chunks, global_chunk_index, global_epochs
            tokens,tokens_weights,targets = [], [], []
            for b in range(batch_size):
                chunk = global_chunks[global_chunk_index]
                global_chunk_index += 1
                if global_chunk_index >= len(global_chunks):
                    global_epochs += 1
                    global_chunk_index = 0
                    np.random.shuffle(global_chunks)
                tokens.append(chunk["tokens"])
                tokens_weights.append(chunk["tokens_weights"])
                targets.append(chunk["target"])
            tokens = np.array(tokens, dtype=np.int32)
            tokens_weights = np.array(tokens_weights, dtype=np.float32)
            targets = np.array(targets, dtype=np.int32)
            return [tokens, tokens_weights], targets

        def save():
            print('Saving model.')
            fn = os.path.join(CHECKPOINT_DIR, args.run_name, 'checkpoint-{}').format(counter)
            lossmodel.save(fn, save_format="tf")
            shutil.copy(os.path.join(args.base_model,"hparams.json"), os.path.join(fn,"hparams.json"))
            shutil.copy(os.path.join(args.base_model,"vocabulary.txt"), os.path.join(fn,"vocabulary.txt"))
            with open(os.path.join(fn,"idmap.json"), "w") as wf:
                wf.write(json.dumps(map_dict))

        def valid(chunks):
            _, pred, target = run_predict(chunks)
            return np.mean(np.array(pred) == np.array(target))

        def pred(fn, chunks):
            with open(fn, "w", encoding="utf-8") as wf:
                rev_map = {v:k for k,v in map_dict.items()}
                wf.write('filename,pred,true\n')
                for fn, pred, tgt in zip(*run_predict(chunks)):
                    wf.write(','.join([fn, rev_map[pred], rev_map[tgt]]) + '\n')

        def run_predict(input_chunks):
            tokens,tokens_weights,filenames,targets = [], [], [], []
            for chunk in input_chunks:
                tokens.append(chunk["tokens"])
                tokens_weights.append(chunk["tokens_weights"])
                filenames.append(chunk["filename"])
                targets.append(chunk["target"])
            tokens = np.array(tokens, dtype=np.int32)
            tokens_weights = np.array(tokens_weights, dtype=np.float32)
            preds = lossmodel.predict([tokens,tokens_weights], batch_size=batch_size)
            preds = np.argmax(preds, axis=-1).tolist()
            return filenames, preds, targets

        if do_training:
            print('Training...')
        elif do_prediction:
            pred(args.output, prediction_chunks)
            return

        avg_loss = (0.0, 0.0)
        best_score = [0.0, 0.0]
        start_time = time.time()

        try:
            while do_training:
                if args.num_epochs > 0 and get_epoch() >= args.num_epochs:
                    save()
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
                    if do_prediction:
                        fn = os.path.join(CHECKPOINT_DIR, args.run_name, 'checkpoint-{}', args.output).format(counter)
                        pred(fn, prediction_chunks)
                if counter % args.val_every == 0:
                    val = valid(global_chunks[:args.val_size])
                    log = '[{counter}step | {epoch:2.2f}epoch | training score: {val}'.format(
                                counter=counter,
                                epoch=get_epoch(),
                                val=val)
                    if val > best_score[0]:
                        best_score[0] = val
                    if do_validation:
                        val = valid(validation_chunks)
                        log += '\n[{counter}step | {epoch:2.2f}epoch | validation score: {val}'.format(
                                    counter=counter,
                                    epoch=get_epoch(),
                                    val=val)
                        if val > best_score[1]:
                            best_score[1] = val
                    print("########")
                    print(log)
                    print(f'best training score: {best_score[0]} validation score: {best_score[1]}')
                    print("########")
                    log = '{counter},{epoch},{val}'.format(
                                counter=counter,
                                epoch=get_epoch(),
                                val=val)
                    if log_dir != '':
                        val_csv.write(log+'\n')
                        val_csv.flush()
        except KeyboardInterrupt:
            print('interrupted')
            save()
            if do_prediction:
                pred(args.output, prediction_chunks)


if __name__ == '__main__':
    main()
