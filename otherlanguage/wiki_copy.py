import os
import uuid
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--src_dir", help="source dir", required=True )
parser.add_argument("--dst_dir", help="destnation dir", required=True )
parser.add_argument("--max_articles", help="max num", default=-1, type=int )
args = parser.parse_args()

if not os.path.isdir(args.dst_dir):
    os.mkdir(args.dst_dir)
    for a in '0123456789abcdef':
        for b in '0123456789abcdef':
            os.mkdir(args.dst_dir+'/'+a+b)

n_out = 0
def one_copy(src_file):
    global n_out

    with open(src_file) as f:
        ln = f.readlines()

    df = None
    for text in ln:
        text = text.strip()
        if not (text.startswith('<doc ') or text.startswith('</doc>')):
            if len(text) > 0:
                if df is None:
                    ui = str(uuid.uuid4())
                    dst_file = args.dst_dir + '/' + ui[0] + ui[1] + '/' + ui + '.txt'
                    df = open(dst_file, "w")
                df.write(text+'\n')
                df.flush()
        else:
            if df is not None:
                df.close()
                n_out = n_out + 1
                if args.max_articles > 0 and n_out >= args.max_articles:
                    exit(0)
            df = None
    if df is not None:
        df.close()

for d in os.listdir(args.src_dir):
    for f in os.listdir(args.src_dir+'/'+d):
        one_copy(args.src_dir+'/'+d+'/'+f)

print("end")
