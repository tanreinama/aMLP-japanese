import argparse
import json
import os
import numpy as np
import time
import shutil
from copy import copy
import torch

import modeling_pt

from encoder import get_encoder

parser = argparse.ArgumentParser(
    description='Question Answering for SQuAD task.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--pred_dataset', metavar='PATH', type=str, required=True, help='Prediction json file')
parser.add_argument('--model', type=str, default='aMLP-SQuAD-base-ja.pt', help='PyTorch Model Name.')
parser.add_argument('--max_answer_length', metavar='SIZE', type=int, default=50, help='Max answer size.')
parser.add_argument('--num_best_indexes', metavar='SIZE', type=int, default=20, help='Select answer size from outputs.')
parser.add_argument('--gpu', type=int, default=0, help='GPU number to use.')
parser.add_argument("--verbose", action='store_true' )

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
    assert os.path.isfile(args.model), f"model file not found: {args.model}"
    bpe_path = os.path.splitext(args.model)[0]+".txt"
    hpm_path = os.path.splitext(args.model)[0]+".json"
    assert os.path.isfile(bpe_path), f"vocabulary file not found in {bpe_path}."
    assert os.path.isfile(hpm_path), f"hparams file not found in {hpm_path}."
    assert os.path.exists('emoji.json'), f"emoji file not found."
    with open(hpm_path) as f:
        conf_dict = json.loads(f.read())

    assert len(args.pred_dataset)==0 or os.path.isfile(args.pred_dataset), f"prediction file not found: {args.pred_dataset}"

    vocab_size = conf_dict["num_vocab"]
    EOT_TOKEN = vocab_size - 1
    MASK_TOKEN = vocab_size - 2
    SEP_TOKEN = vocab_size - 3
    CLS_TOKEN = vocab_size - 4
    max_seq_length = conf_dict["num_ctx"]
    max_answer_length = args.max_answer_length

    with open(bpe_path) as f:
        ww = np.sum([1 if ('##' in l) else 0 for l in f.readlines()]) > 0
    enc = get_encoder(bpe_path, 'emoji.json', ww)

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

    prediction_chunks = encode_json(args.pred_dataset)

    pt_model = modeling_pt.model(num_ctx=conf_dict["num_ctx"],
                     num_vocab=conf_dict["num_vocab"],
                     num_hidden=conf_dict["num_hidden"],
                     num_soft_att=conf_dict["num_soft_att"],
                     num_layer=conf_dict["num_layer"],
                     has_voc=True,
                     classifier_n=-1,
                     is_squad=True)
    pt_model.load_state_dict(torch.load(args.model))
    device = "cuda:%d"%args.gpu if args.gpu>=0 else "cpu"
    pt_model.to(device)
    pt_model.eval()

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
        tokens = np.array(tokens, dtype=np.int64)
        tokens_weights = np.array(tokens_weights, dtype=np.float32)
        pred0, pred1 = [], []
        for i in range(len(tokens)):
            t,w = tokens[i:i+1,:],tokens_weights[i:i+1,:]
            t,w = torch.tensor(t),torch.tensor(w)
            t,w = t.to(device),w.to(device)
            _, _, pred = pt_model(t,w)
            pred = pred.detach().cpu().numpy()[0].transpose((1,0))
            pred0.append(pred[0])
            pred1.append(pred[1])
        result = []
        pi = 0
        for starts, ends, off, pos, ctx, qid, ans, qes in zip(pred0, pred1, ctx_offset, ctx_posisions, context, question_id, answer, question):
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

    pred('squad-predicted.json', prediction_chunks)
    result = encode_json('squad-predicted.json')
    question_id = np.array([res["question_id"] for res in result])
    question = [res["question"] for res in result]
    answer = [res["answer"] for res in result]
    index = np.arange(len(result))
    if not args.verbose:
        print('Question\tAnswer')
    for qid in np.unique(question_id):
        i = sorted(index[np.where(question_id == qid)])[0]
        if args.verbose:
            print("[Context]")
            print(result[i]["context"])
            if len(question[i]) > 0:
                print("[Question]")
                print(question[i])
            print("[Answer]")
            print(answer[i])
        else:
            print(question[i]+'\t'+answer[i])

if __name__ == '__main__':
    main()
