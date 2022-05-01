import tensorflow as tf
import torch
import argparse
import json
import os
from collections import OrderedDict
import shutil
import modeling_pt

parser = argparse.ArgumentParser(
    description='model converter.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--tf_model_dir', metavar='PATH', type=str, required=True, help='import model')
parser.add_argument('--output', metavar='PATH', type=str, required=True, help='output model')

def main():
    args = parser.parse_args()
    params = json.loads(open(os.path.join(args.tf_model_dir,"hparams.json")).read())
    assert "type" in params and params["type"] == "mlp", "model type must 'mlp'."
    if not args.output.endswith(".pt"):
        args.output = args.output+".pt"
    model = tf.keras.models.load_model(args.tf_model_dir, \
            custom_objects={'loss': tf.keras.losses.Loss(),'crossentropy': tf.keras.losses.Loss()})
    names = [weight.name for layer in model.layers for weight in layer.weights]
    weights = model.get_weights()
    state = {name:weight for name, weight in zip(names, weights)}
    has_voc = "vocabrary_embed:0" in state
    is_squad = "squad_output_w:0" in state
    is_classifier = os.path.isfile(os.path.join(args.tf_model_dir,"idmap.json"))
    assert (not is_classifier) or ("output_w:0" in state), "not classifier model but idmap.json in dir."
    assert int(is_squad)+int(is_classifier) <= 1, "squad model but idmap.json in dir."
    classifier_n = state["output_w:0"].shape[1] if is_classifier else -1
    new_state = OrderedDict()
    pt_model = modeling_pt.model(num_ctx=params["num_ctx"],
                     num_vocab=params["num_vocab"],
                     num_hidden=params["num_hidden"],
                     num_soft_att=params["num_soft_att"],
                     num_layer=params["num_layer"],
                     has_voc=has_voc,
                     classifier_n=classifier_n,
                     is_squad=is_squad)
    for k,v in pt_model.state_dict().items():
        kn = k.replace(".","_") + ":0"
        if kn=="vocabrary_embed:0" and kn not in state:
            continue
        assert kn in state and state[kn].shape == v.shape, "key name not match in %s"%k
        new_state[k] = torch.tensor(state[kn])
    assert len(new_state) == len(pt_model.state_dict()), "loaded parameter mismach."
    pt_model.load_state_dict(new_state)
    print("has_voc:",has_voc,"is_classifier:",is_classifier,"is_squad:",is_squad)
    torch.save(new_state, args.output)
    shutil.copy(os.path.join(args.tf_model_dir,"vocabulary.txt"), os.path.splitext(args.output)[0]+".txt")
    shutil.copy(os.path.join(args.tf_model_dir,"hparams.json"), os.path.splitext(args.output)[0]+".json")
    if is_classifier:
        shutil.copy(os.path.join(args.tf_model_dir,"idmap.json"), os.path.splitext(args.output)[0]+"-idmap.json")

if __name__ == '__main__':
    main()
