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
from modeling import projection

from encoder import get_encoder

CHECKPOINT_DIR = 'checkpoint'

parser = argparse.ArgumentParser(
    description='Question Answering for SQuAD task.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dataset', metavar='PATH', type=str, default="", help='Training json file')
parser.add_argument('--val_size', metavar='PATH', type=int, default=1000, help='data size to calucrate training score')
parser.add_argument('--val_dataset', metavar='PATH', type=str, default="", help='Validation json file')
parser.add_argument('--pred_dataset', metavar='PATH', type=str, default="", help='Prediction json file')
parser.add_argument('--base_model', type=str, default='aMLP-base-ja', help='Base Model Name.')
parser.add_argument('--max_answer_length', metavar='SIZE', type=int, default=50, help='Max answer size.')
parser.add_argument('--num_best_indexes', metavar='SIZE', type=int, default=20, help='Select answer size from outputs.')

parser.add_argument('--batch_size', metavar='SIZE', type=int, default=4, help='Batch size')
parser.add_argument('--log_dir', type=str, default='', help='Directory to save log.')
parser.add_argument('--run_name', type=str, default='aMLP-squad-ja', help='Run id. Name of subdirectory in checkpoint/')
parser.add_argument('--save_every', metavar='N', type=int, default=10000, help='Write a checkpoint every N steps')
parser.add_argument('--val_every', metavar='N', type=int, default=1000, help='Validate every N steps')
parser.add_argument('--num_epochs', metavar='N', type=float, default=2, help='Maximum training epochs.')
parser.add_argument('--learning_rate', type=float, default=1e-5, help='Training learning rate for Adam.')
parser.add_argument('--restore_from', type=str, default='', help='checkpoint name for restore training.')

def read_squad_json(filename, to_val=False):
    with open(filename) as f:
        squad = json.loads(f.read())
    context, question, answer_start, answer_end, question_id, answer = [], [], [], [], [], []
    num_quest = 0
    for data in squad["data"]:
        for p in data["paragraphs"]:
            c = p["context"]
            for q in p["qas"]:
                if "is_impossible" not in q or not q["is_impossible"]:
                    for a in (q["answers"][:1] if to_val else q["answers"]):
                        answer.append(a["text"])
                        context.append(c)
                        question.append(q["question"])
                        if "id" in q:
                            question_id.append(q["id"])
                        else:
                            question_id.append(str(num_quest))
                        answer_start.append(a["answer_start"])
                        answer_end.append(a["answer_start"]+len(a["text"]))
                        num_quest += 1
                elif not to_val:
                    answer.append("")
                    context.append(c)
                    question.append(q["question"])
                    if "id" in q:
                        question_id.append(q["id"])
                    else:
                        question_id.append(str(num_quest))
                    answer_start.append(-1)
                    answer_end.append(-1)
                    num_quest += 1
    print(f'read {len(context)} contexts from {filename}.')
    return context, question, answer_start, answer_end, question_id, answer

def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def jaccard_wd(str1, str2):
    a = set(str1)
    b = set(str2)
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def get_best_indexes(logits, n_best_size):
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)
    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes

def main():
    args = parser.parse_args()
    if len(args.restore_from)==0:
        assert os.path.isdir(args.base_model), f"model not found: {args.base_model}"
        bpe_path = os.path.join(args.base_model, "vocabulary.txt")
        hpm_path = os.path.join(args.base_model,"hparams.json")
    else:
        bpe_path = os.path.join(args.restore_from, "vocabulary.txt")
        hpm_path = os.path.join(args.restore_from,"hparams.json")
    assert len(args.restore_from)==0 or os.path.isdir(args.restore_from), f"checkpoint not found: {args.restore_from}"
    assert os.path.isfile(bpe_path), f"vocabulary.txt not found in {bpe_path}."
    assert os.path.isfile(hpm_path), f"hparams.json not found in {hpm_path}."
    assert os.path.exists('emoji.json'), f"emoji file not found."
    with open(hpm_path) as f:
        conf_dict = json.loads(f.read())

    assert len(args.dataset)==0 or os.path.isfile(args.dataset), f"training file not found: {args.dataset}"
    assert len(args.val_dataset)==0 or os.path.isfile(args.val_dataset), f"validation file not found: {args.val_dataset}"
    assert len(args.pred_dataset)==0 or os.path.isfile(args.pred_dataset), f"prediction file not found: {args.pred_dataset}"
    do_training = len(args.dataset) > 0 and os.path.isfile(args.dataset)
    do_validation = len(args.val_dataset) > 0 and os.path.isfile(args.val_dataset)
    do_prediction = len(args.pred_dataset) > 0 and os.path.isfile(args.pred_dataset)
    assert (do_training or do_prediction), "must run with training or prediction task"

    vocab_size = conf_dict["num_vocab"]
    EOT_TOKEN = vocab_size - 1
    MASK_TOKEN = vocab_size - 2
    SEP_TOKEN = vocab_size - 3
    CLS_TOKEN = vocab_size - 4
    batch_size = args.batch_size
    max_seq_length = conf_dict["num_ctx"]
    max_predictions = 1
    log_dir = args.log_dir
    max_answer_length = args.max_answer_length

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

    class squad_model(tf.keras.Model):
        def __init__(self, model, conf_dict):
            super(squad_model, self).__init__(name='squad_model')
            self.model = model
            self.projection = projection(conf_dict["num_hidden"], 2, name='squad_output')
        def call(self, inputs):
            input_ids, input_weights = inputs
            lm_output, _ = self.model(inputs=[input_ids, input_weights])
            logits = self.projection(lm_output)
            logits = tf.transpose(logits, [2, 0, 1])
            unstacked_logits = tf.unstack(logits, axis=0)
            start_logits, end_logits = unstacked_logits[0], unstacked_logits[1]
            return [start_logits, end_logits]

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

        if len(args.restore_from) > 0:
            lossmodel = tf.keras.models.load_model(args.restore_from, \
                    custom_objects={'crossentropy': crossentropy})
        else:
            model = tf.keras.models.load_model(args.base_model, \
                    custom_objects={'loss': tf.keras.losses.Loss()})
            lossmodel = squad_model(model, conf_dict)
            opt = Adam(learning_rate=args.learning_rate)
            lossmodel.compile(optimizer=opt, loss=[crossentropy,crossentropy])
            lossmodel.optimizer.apply_gradients(zip([tf.zeros_like(x) for x in model.weights], model.weights))

        print('Loading dataset...')
        def encode_json(filename):
            result_chunks = []
            for context, question, answer_start, answer_end, question_id, answer in zip(*read_squad_json(filename)):
                if '？' not in question:
                    question = question.reulace('?', '？')
                    if '？' not in question:
                        question = question + '？'
                enc_context, ctx_posisions = enc.encode(context, clean=False, position=True)
                enc_question = enc.encode(question, clean=False, position=False)
                token_start = -1 if answer_start<0 else np.argmax(np.array(ctx_posisions+[1000000]) >= answer_start)
                token_end = 0 if answer_end<=0 else np.argmax(np.array(ctx_posisions+[1000000]) >= answer_end)
                ctx_offset = 1 + len(enc_question) + 2
                tokens = [CLS_TOKEN] + enc_question + [SEP_TOKEN, CLS_TOKEN] + enc_context + [EOT_TOKEN]
                tokens_weights = [1.0] * len(tokens)
                token_start = min(len(tokens)-2, token_start + ctx_offset)
                token_end = max(token_start, token_end + ctx_offset - 1)
                while len(tokens) < max_seq_length:
                    tokens.append(EOT_TOKEN)
                    tokens_weights.append(0.0)
                tokens = tokens[:max_seq_length]
                tokens_weights = tokens_weights[:max_seq_length]
                if token_start >= max_seq_length:
                    token_start = ctx_offset-1
                    token_end = ctx_offset-1
                elif token_end >= max_seq_length:
                    token_end = max_seq_length-1
                answer = context[answer_start:answer_end]
                result_chunks.append({"tokens":tokens,"tokens_weights":tokens_weights,"token_start":token_start,"token_end":token_end,"question":question,
                                      "ctx_offset":ctx_offset,"ctx_posisions":ctx_posisions,"context":context,"answer":answer,"question_id":question_id})
            return result_chunks

        global_chunks = encode_json(args.dataset) if do_training else None
        global_chunk_index = 0
        global_epochs = 0

        if do_training:
            np.random.shuffle(global_chunks)

        validation_chunks = encode_json(args.val_dataset) if do_validation else None
        prediction_chunks = encode_json(args.pred_dataset) if do_prediction else None

        def get_epoch():
            return global_epochs + (global_chunk_index / len(global_chunks))

        def sample_feature():
            nonlocal global_chunks, global_chunk_index, global_epochs
            tokens,tokens_weights,token_start,token_end = [], [], [], []
            for b in range(batch_size):
                chunk = global_chunks[global_chunk_index]
                global_chunk_index += 1
                if global_chunk_index >= len(global_chunks):
                    global_epochs += 1
                    global_chunk_index = 0
                    np.random.shuffle(global_chunks)
                tokens.append(chunk["tokens"])
                tokens_weights.append(chunk["tokens_weights"])
                token_start.append([chunk["token_start"]])
                token_end.append([chunk["token_end"]])
            tokens = np.array(tokens, dtype=np.int32)
            tokens_weights = np.array(tokens_weights, dtype=np.float32)
            token_start = np.array(token_start, dtype=np.int32)
            token_end = np.array(token_end, dtype=np.int32)
            return [tokens, tokens_weights], [token_start, token_end]

        def save():
            print('Saving model.')
            fn = os.path.join(CHECKPOINT_DIR, args.run_name, 'checkpoint-{}').format(counter)
            lossmodel.save(fn, save_format="tf")
            shutil.copy(os.path.join(args.base_model,"hparams.json"), os.path.join(fn,"hparams.json"))
            shutil.copy(os.path.join(args.base_model,"vocabulary.txt"), os.path.join(fn,"vocabulary.txt"))

        def valid(chunks):
            scores = []
            for preds in run_predict(chunks):
                pred = preds["predictionstrings"]
                label = preds["answer"]
                if len(pred) > 0:
                    added = False
                    for text in pred:
                        if len(text) > 0:
                            if not ww:
                                scores.append(jaccard_wd(text, label))
                            else:
                                scores.append(jaccard(text, label))
                            added = True
                            break
                    if not added:
                        scores.append(0.0)
                else:
                    scores.append(0.0)
            return np.mean(scores)

        def pred(fn, chunks):
            data = []
            for preds in run_predict(chunks):
                answers = []
                context = preds["context"]
                question = preds["question"]
                for pred, pred_pos in zip(preds["predictionstrings"],preds["predictionpositions"]):
                    if len(pred) > 0:
                        answers.append({"text":pred,"answer_start":pred_pos})
                qas = {"id":preds["question_id"],"question":question,"is_impossible":preds["impossible"],"answers":answers}
                data.append({"paragraphs":[{"context":context,"qas":[qas]}]})
            with open(fn, "w", encoding="utf-8") as wf:
                wf.write(json.dumps({"data":data}, ensure_ascii=False , indent=2))

        def run_predict(input_chunks):
            tokens,tokens_weights,ctx_offset,ctx_posisions,context,question_id,answer,question = [], [], [], [], [], [], [], []
            pp=[]
            for chunk in input_chunks:
                tokens.append(chunk["tokens"])
                tokens_weights.append(chunk["tokens_weights"])
                ctx_offset.append(chunk["ctx_offset"])
                ctx_posisions.append(chunk["ctx_posisions"])
                context.append(chunk["context"])
                question_id.append(chunk["question_id"])
                answer.append(chunk["answer"])
                question.append(chunk["question"])
                pp.append("true_y: %d %d"%(chunk["token_start"],chunk["token_end"]))
            tokens = np.array(tokens, dtype=np.int32)
            tokens_weights = np.array(tokens_weights, dtype=np.float32)
            pred = lossmodel.predict([tokens,tokens_weights], batch_size=batch_size)
            result = []
            pi = 0
            for starts, ends, off, pos, ctx, qid, ans, qes in zip(pred[0], pred[1], ctx_offset, ctx_posisions, context, question_id, answer, question):
                selected = []
                impossible = False
                p_starts = get_best_indexes(starts, args.num_best_indexes)
                p_ends = get_best_indexes(ends, args.num_best_indexes)
                pi += 1
                for start_index in p_starts:
                    for end_index in p_ends:
                        if start_index==off-1 and end_index==off-1 and len(selected)==0:
                            impossible = True
                        if start_index-off >= len(pos) or start_index<off:
                            continue
                        if end_index-off >= len(pos) or end_index<off:
                            continue
                        if end_index < start_index:
                            continue
                        length = end_index - start_index + 1
                        if length > max_answer_length:
                            continue
                        selected.append((start_index, end_index))
                predictionstrings = []
                predictionpositions = []
                for p_start,p_end in selected:
                    start_token = p_start-off
                    end_token = p_end-off
                    start_pos = pos[start_token]
                    end_pos = pos[end_token+1] if end_token+1<len(pos) else len(ctx)
                    predictionstrings.append(ctx[start_pos:end_pos])
                    predictionpositions.append(start_pos)
                result.append({"predictionstrings":predictionstrings, "predictionpositions":predictionpositions,
                               "impossible":impossible, "answer":ans, "question_id":qid, "context":ctx, "question":qes})
            return result

        if do_training:
            print('Training...')
        elif do_prediction:
            pred('squad-predicted.json', prediction_chunks)
            result = encode_json('squad-predicted.json')
            question_id = np.array([res["question_id"] for res in result])
            question = [res["question"] for res in result]
            answer = [res["answer"] for res in result]
            index = np.arange(len(result))
            print('Question\tAnswer')
            for qid in np.unique(question_id):
                i = sorted(index[np.where(question_id == qid)])[0]
                print(question[i]+'\t'+answer[i])
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
                v_loss, _, _ = lossmodel.train_on_batch(X, y)

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
                        fn = os.path.join(CHECKPOINT_DIR, args.run_name, 'checkpoint-{}', 'squad-predicted.json').format(counter)
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
                pred('squad-predicted.json', prediction_chunks)


if __name__ == '__main__':
    main()
