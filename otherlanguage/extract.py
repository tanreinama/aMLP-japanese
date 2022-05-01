import os,json,uuid,gzip,random,pickle
import numpy as np
from multiprocessing import Pool
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--src_dir", help="source dir", required=True )
parser.add_argument("--C4_max_data", help="data number from C4 dataset", type=int, default=3000000000 )
parser.add_argument("--C100_max_data", help="data number from C100 dataset", type=int, default=4000000000 )
args = parser.parse_args()

# make subset of C4/C100 data
lang = {args.src_dir:(args.C4_max_data,args.C100_max_data)}
for l,p in lang.items():
    if not os.path.isdir(l+"/content"):
        os.mkdir(l+"/content")
    for a in "abcdef0123456789":
        for b in "abcdef0123456789":
            if not os.path.isdir(l+"/content/"+a+b):
                    os.mkdir(l+"/content/"+a+b)

def check_hindi(x):
    e = x.encode()
    if len(x) == 1 and len(e)==3:
        c = (int(e[0])<<16)+(int(e[1])<<8)+int(e[2])
        if c >= 0xE0A480 and c <= 0xE0A5BF:
            return True
    return False
def check_tamil(x):
    e = x.encode()
    if len(x) == 1 and len(e)==3:
        c = (int(e[0])<<16)+(int(e[1])<<8)+int(e[2])
        if c >= 0xE0A484 and c <= 0xE0AFBA:
            return True
    return False
def count_char(l, t):
    if l=="Tamil":
        f = check_tamil
    elif l=="Hindi":
        f = check_hindi
    else:
        return 0
    return np.sum([f(x) for x in t])

def add_file(d, t):
    fn = str(uuid.uuid4())+'.txt'
    with open(f"{d}/content/{fn[0]}{fn[1]}/{fn}", "w") as wf:
        wf.write(t)
    print(f"{d}/content/{fn[0]}{fn[1]}/{fn}")

def count_cont(fn):
    print("count...",fn)
    l = fn.split("/")[0]
    p = []
    if fn.endswith(".txt"):
        current = []
        with open(fn) as f:
            line = f.readline()
            while line:
                line = line.strip()
                if len(line) == 0:
                    p.append(count_char(l,'\n'.join(current)))
                    current = []
                else:
                    current.append(line)
                line = f.readline()
    elif fn.endswith(".json.gz"):
        with gzip.open(fn) as f:
            line = f.readline()
            while line:
                line = line.strip()
                p.append(count_char(l,json.loads(line)['text']))
                line = f.readline()
    with open(f'{fn}.pickle', mode='wb') as f:
        pickle.dump(p, f)

content_size1 = {args.src_dir:[]}
content_size2 = {args.src_dir:[]}

for l,p in lang.items():
    files = [l+"/"+fn for fn in os.listdir(l) if os.path.isfile(l+"/"+fn) and not fn.endswith(".pickle")]
    with Pool(8) as p:
        p.map(count_cont, [fn for fn in files if not os.path.isfile(f'{fn}.pickle')])
    for fn in files:
        with open(f'{fn}.pickle', mode='rb') as f:
            r = pickle.load(f)
        lg = fn.split("/")[0]
        if fn.endswith(".txt"):
            content_size1[lg].extend(r)
        elif fn.endswith(".json.gz"):
            content_size2[lg].extend(r)

# cleaning
content_threash1 = {l:np.sum(content_size1[l]) for l,p in lang.items()}
content_threash2 = {l:np.sum(content_size2[l]) for l,p in lang.items()}
print("max_size:")
print(content_threash1, content_threash2)

content_size1 = {l:sorted(content_size1[l])[::-1] for l,p in lang.items()}
content_size2 = {l:sorted(content_size2[l])[::-1] for l,p in lang.items()}
content_sizex1 = {l:np.cumsum(content_size1[l]) for l,p in lang.items()}
content_sizex2 = {l:np.cumsum(content_size2[l]) for l,p in lang.items()}
content_threash1 = {l:content_size1[l][np.where(content_sizex1[l]>p[0])[0][0]] for l,p in lang.items()}
content_threash2 = {l:content_size2[l][np.where(content_sizex2[l]>p[1])[0][0]] for l,p in lang.items()}

lang = {l:(content_threash1[l],content_threash2[l]) for l,p in lang.items()}
print("threasholds:")
print(lang)

def write_cont(fn):
    print("write...",fn)
    l = fn.split("/")[0]
    p = lang[l]
    p_a = lang_a[l]
    if fn.endswith(".txt"):
        current = []
        with open(fn) as f:
            line = f.readline()
            while line:
                line = line.strip()
                if len(line) == 0:
                    txt = '\n'.join(current)
                    cc = count_char(l, txt)
                    if cc > p[0]:
                        add_file(l, txt)
                    current = []
                else:
                    current.append(line)
                line = f.readline()
    elif fn.endswith(".json.gz"):
        with gzip.open(fn) as f:
            line = f.readline()
            while line:
                line = line.strip()
                txt = json.loads(line)['text']
                cc = count_char(l, txt)
                if cc > p[1]:
                    add_file(l, txt)
                line = f.readline()

for l,p in lang.items():
    files = [l+"/"+fn for fn in os.listdir(l) if os.path.isfile(l+"/"+fn) and not fn.endswith(".pickle")]
    with Pool(8) as p:
        res = p.map(write_cont, files)

print('end')
