import os, glob
import numpy as np
from multiprocessing import Pool
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--src_dir", help="source dir", required=True )
parser.add_argument("--language", help="use language (hi/ta/other)", required=True )
args = parser.parse_args()

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
        if c >= 0xE0AE82 and c <= 0xE0AFBA:
            return True
    return False
def check_other(x):
    return True
def check_cansep(x):
    e = x.encode()
    if len(x) == 1 and len(e)==3:
        c = (int(e[0])<<16)+(int(e[1])<<8)+int(e[2])
        # Hindi
        if c >= 0xE0A484 and c <= 0xE0A4B9:
            return True
        elif c == 0xE0A590:
            return True
        elif c >= 0xE0A593 and c <= 0xE0A594:
            return True
        elif c >= 0xE0A598 and c <= 0xE0A5A1:
            return True
        elif c >= 0xE0A5A4 and c <= 0xE0A5BF:
            return True
        # Tamil
        elif c == 0xE0AE83:
            return True
        elif c >= 0xE0AE85 and c <= 0xE0AEB9:
            return True
        elif c == 0xE0AF90:
            return True
        elif c >= 0xE0AFA6 and c <= 0xE0AFBA:
            return True
    return not (check_hindi(x) or check_tamil(x))

lang_func = check_tamil if args.language=="ta" else (check_hindi if args.language=="hi" check_other)
lang = {args.src_dir:check_tamil}
max_bpe = 24000
base_words = """!
"
#
$
%
&
'
(
)
*
+
,
-
.
/
0
1
2
3
4
5
6
7
8
9
:
;
<
=
?
@
A
B
C
D
E
F
G
H
I
J
K
L
M
N
O
P
Q
R
S
T
U
V
W
X
Y
Z
[
\
]
^
_
`
a
b
c
d
e
f
g
h
i
j
k
l
m
n
o
p
q
r
s
t
u
v
w
x
y
z
{
|
}
~
、
。
・
‥
…
→
←
↑
↓
<BR>
<SP>
<TAB>
<URL>
<EMAIL>
<BLOCK>
<KIGOU>
<U2000U2BFF>
<|emoji1|>
<|emoji2|>
<|emoji3|>
<|emoji4|>
<|emoji5|>
<|emoji6|>
<|emoji7|>
<|emoji8|>
<|emoji9|>
<|emoji10|>
<|emoji11|>
<|emoji12|>
<|byte0|>
<|byte1|>
<|byte2|>
<|byte3|>
<|byte4|>
<|byte5|>
<|byte6|>
<|byte7|>
<|byte8|>
<|byte9|>
<|byte10|>
<|byte11|>
<|byte12|>
<|byte13|>
<|byte14|>
<|byte15|>
<|byte16|>
<|byte17|>
<|byte18|>
<|byte19|>
<|byte20|>
<|byte21|>
<|byte22|>
<|byte23|>
<|byte24|>
<|byte25|>
<|byte26|>
<|byte27|>
<|byte28|>
<|byte29|>
<|byte30|>
<|byte31|>
<|byte32|>
<|byte33|>
<|byte34|>
<|byte35|>
<|byte36|>
<|byte37|>
<|byte38|>
<|byte39|>
<|byte40|>
<|byte41|>
<|byte42|>
<|byte43|>
<|byte44|>
<|byte45|>
<|byte46|>
<|byte47|>
<|byte48|>
<|byte49|>
<|byte50|>
<|byte51|>
<|byte52|>
<|byte53|>
<|byte54|>
<|byte55|>
<|byte56|>
<|byte57|>
<|byte58|>
<|byte59|>
<|byte60|>
<|byte61|>
<|byte62|>
<|byte63|>
<|byte64|>
<|byte65|>
<|byte66|>
<|byte67|>
<|byte68|>
<|byte69|>
<|byte70|>
<|byte71|>
<|byte72|>
<|byte73|>
<|byte74|>
<|byte75|>
<|byte76|>
<|byte77|>
<|byte78|>
<|byte79|>
<|byte80|>
<|byte81|>
<|byte82|>
<|byte83|>
<|byte84|>
<|byte85|>
<|byte86|>
<|byte87|>
<|byte88|>
<|byte89|>
<|byte90|>
<|byte91|>
<|byte92|>
<|byte93|>
<|byte94|>
<|byte95|>
<|byte96|>
<|byte97|>
<|byte98|>
<|byte99|>
<|byte100|>
<|byte101|>
<|byte102|>
<|byte103|>
<|byte104|>
<|byte105|>
<|byte106|>
<|byte107|>
<|byte108|>
<|byte109|>
<|byte110|>
<|byte111|>
<|byte112|>
<|byte113|>
<|byte114|>
<|byte115|>
<|byte116|>
<|byte117|>
<|byte118|>
<|byte119|>
<|byte120|>
<|byte121|>
<|byte122|>
<|byte123|>
<|byte124|>
<|byte125|>
<|byte126|>
<|byte127|>
<|byte128|>
<|byte129|>
<|byte130|>
<|byte131|>
<|byte132|>
<|byte133|>
<|byte134|>
<|byte135|>
<|byte136|>
<|byte137|>
<|byte138|>
<|byte139|>
<|byte140|>
<|byte141|>
<|byte142|>
<|byte143|>
<|byte144|>
<|byte145|>
<|byte146|>
<|byte147|>
<|byte148|>
<|byte149|>
<|byte150|>
<|byte151|>
<|byte152|>
<|byte153|>
<|byte154|>
<|byte155|>
<|byte156|>
<|byte157|>
<|byte158|>
<|byte159|>
<|byte160|>
<|byte161|>
<|byte162|>
<|byte163|>
<|byte164|>
<|byte165|>
<|byte166|>
<|byte167|>
<|byte168|>
<|byte169|>
<|byte170|>
<|byte171|>
<|byte172|>
<|byte173|>
<|byte174|>
<|byte175|>
<|byte176|>
<|byte177|>
<|byte178|>
<|byte179|>
<|byte180|>
<|byte181|>
<|byte182|>
<|byte183|>
<|byte184|>
<|byte185|>
<|byte186|>
<|byte187|>
<|byte188|>
<|byte189|>
<|byte190|>
<|byte191|>
<|byte192|>
<|byte193|>
<|byte194|>
<|byte195|>
<|byte196|>
<|byte197|>
<|byte198|>
<|byte199|>
<|byte200|>
<|byte201|>
<|byte202|>
<|byte203|>
<|byte204|>
<|byte205|>
<|byte206|>
<|byte207|>
<|byte208|>
<|byte209|>
<|byte210|>
<|byte211|>
<|byte212|>
<|byte213|>
<|byte214|>
<|byte215|>
<|byte216|>
<|byte217|>
<|byte218|>
<|byte219|>
<|byte220|>
<|byte221|>
<|byte222|>
<|byte223|>
<|byte224|>
<|byte225|>
<|byte226|>
<|byte227|>
<|byte228|>
<|byte229|>
<|byte230|>
<|byte231|>
<|byte232|>
<|byte233|>
<|byte234|>
<|byte235|>
<|byte236|>
<|byte237|>
<|byte238|>
<|byte239|>
<|byte240|>
<|byte241|>
<|byte242|>
<|byte243|>
<|byte244|>
<|byte245|>
<|byte246|>
<|byte247|>
<|byte248|>
<|byte249|>
<|byte250|>
<|byte251|>
<|byte252|>
<|byte253|>
<|byte254|>
<|cls|>
<|sep|>
<|pad|>
<|endoftext|>"""

def count_bpe(fn_list):
    print("start_batch",fn_list[0])
    l = fn_list[0].split('/')[0]
    check_word = lang[l]
    chars = set()
    bpe_dict = {}
    for fn in fn_list:
        with open(fn) as f:
            words = f.read()
        words = words.replace("।",".")
        words = words.replace("॥",".")
        words = words.replace("१","1")
        words = words.replace("२","2")
        words = words.replace("३","3")
        words = words.replace("४","4")
        words = words.replace("५","5")
        words = words.replace("६","6")
        words = words.replace("७","7")
        words = words.replace("८","8")
        words = words.replace("९","9")
        words = words.replace("०","0")
        words = words.replace("शून्य","0")
        words = words.replace("सिफ़र","0")
        words = words.replace("एक","1")
        words = words.replace("दो","2")
        words = words.replace("तीन","3")
        words = words.replace("चार","4")
        words = words.replace("पाँच","5")
        words = words.replace("छह","6")
        words = words.replace("सात","7")
        words = words.replace("आठ","8")
        words = words.replace("नौ","9")
        words = words.replace("दस","10")
        words = words.replace("ग्यारह","11")
        words = words.replace("बारह","12")
        words = words.replace("तेरह","13")
        words = words.replace("चौदह","14")
        words = words.replace("पंद्रह","15")
        words = words.replace("सोलह","16")
        words = words.replace("सत्रह","17")
        words = words.replace("अठारह","18")
        words = words.replace("उन्नीस","19")
        words = words.replace("बीस","20")
        if len(words) > 4:
            for i in range(len(words)-4):
                b0 = False if i==0 else check_word(words[i-1])
                s5 = True if i+4==len(words)-4 else check_cansep(words[i+4])
                c1 = words[i]
                c2 = words[i+1]
                c3 = words[i+2]
                c4 = words[i+3]
                b1 = check_word(c1)
                b2 = check_word(c2)
                b3 = check_word(c3)
                b4 = check_word(c4)
                s1 = check_cansep(c1)
                s2 = check_cansep(c2)
                s3 = check_cansep(c3)
                s4 = check_cansep(c4)
                if b0:
                    p = '##'
                else:
                    p = ''
                if b1:
                    chars.add(c1)
                if s1 and b1 and b2:
                    if s3:
                        q = p+c1+c2
                        if q not in bpe_dict:
                            bpe_dict[q] = 0
                        bpe_dict[q] += 1
                    if b1 and b2 and b3:
                        if s4:
                            q = p+c1+c2+c3
                            if q not in bpe_dict:
                                bpe_dict[q] = 0
                            bpe_dict[q] += 1
                        if b1 and b2 and b3 and b4:
                            if s5:
                                q = p+c1+c2+c3+c4
                                if q not in bpe_dict:
                                    bpe_dict[q] = 0
                                bpe_dict[q] += 1
    bpe_order = sorted(bpe_dict.items(), key=lambda x:x[1])[-max_bpe:]
    return [chars, bpe_order]

BATCH_SIZE = 12000
for l, check_word in lang.items():
    fns = list(glob.glob(f'{l}/content/*/*.txt'))
    spl = []
    for i in range(0,len(fns),BATCH_SIZE):
        j = min(i+BATCH_SIZE, len(fns))
        spl.append(fns[i:j])
    with Pool(8) as p:
        res = p.map(count_bpe, spl)
    chars = set()
    bpe_dict = {}
    for r in res:
        chars |= r[0]
        for k,v in r[1]:
            if k not in bpe_dict:
                bpe_dict[k] = 0
            bpe_dict[k] += v
    bpe_count = max_bpe - len(chars)*2 - len(base_words.split('\n'))
    bpe_order = sorted(bpe_dict.items(), key=lambda x:x[1])[-bpe_count:]
    bpe_order = bpe_order[::-1]
    all_words = [w[0] for w in bpe_order] + list(chars) + ["##"+s for s in chars] + base_words.split('\n')
    with open(f"{l}/vocabulary.txt", "w") as wf:
        wf.write('\n'.join(all_words))

print("end")
