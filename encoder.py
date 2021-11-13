import numpy as np
import re
import json
import os

class SWEEncoder_wholeword:
    def __init__(self, bpe, emoji):
        self.bpe = bpe
        self.swe = {}
        for idx, wd in enumerate(self.bpe):
            self.swe[wd] = idx
        self.emoji = emoji
        self.maxlen = np.max([len(w) for w in self.swe.keys()])
        self.content_repatter1 = re.compile(r"(https?|ftp)(:\/\/[-_\.!~*\'()a-zA-Z0-9;\/?:\@&=\+$,%#]+)")
        self.content_repatter2 = re.compile(r"[A-Za-z0-9\._+]*@[\-_0-9A-Za-z]+(\.[A-Za-z]+)*")
        self.content_repatter3 = re.compile(r'[\(]{0,1}[0-9]{2,4}[\)\-\(]{0,1}[0-9]{2,4}[\)\-]{0,1}[0-9]{3,4}')
        self.content_repatter4 = re.compile(r"([12]\d{3}[/\-年])*(0?[1-9]|1[0-2])[/\-月]((0?[1-9]|[12][0-9]|3[01])日?)*(\d{1,2}|:|\d{1,2}時|\d{1,2}分|\(日\)|\(月\)|\(火\)|\(水\)|\(木\)|\(金\)|\(土\)|㈰|㈪|㈫|㈬|㈭|㈮|㈯)*")
        self.content_repatter5 = re.compile(r"(明治|大正|昭和|平成|令和|㍾|㍽|㍼|㍻|\u32ff)\d{1,2}年(0?[1-9]|1[0-2])月(0?[1-9]|[12][0-9]|3[01])日(\d{1,2}|:|\d{1,2}時|\d{1,2}分|\(日\)|\(月\)|\(火\)|\(水\)|\(木\)|\(金\)|\(土\)|㈰|㈪|㈫|㈬|㈭|㈮|㈯)*")
        self.content_repatter6 = re.compile(r'((0|[1-9]\d*|[1-9]\d{0,2}(,\d{3})+)*億)*((0|[1-9]\d*|[1-9]\d{0,2}(,\d{3})+)*万)*((0|[1-9]\d*|[1-9]\d{0,2}(,\d{3})+)*千)*(0|[1-9]\d*|[1-9]\d{0,2}(,\d{3})+)*(千円|万円|千万円|円|千ドル|万ドル|千万ドル|ドル|千ユーロ|万ユーロ|千万ユーロ|ユーロ)+(\(税込\)|\(税抜\)|\+tax)*')
        keisen = "─━│┃┄┅┆┇┈┉┊┋┌┍┎┏┐┑┒┓└┕┖┗┘┙┚┛├┝┞┟┠┡┢┣┤┥┦┧┨┩┪┫┬┭┮┯┰┱┲┳┴┵┶┷┸┹┺┻┼┽┾┿╀╁╂╃╄╅╆╇╈╉╊╋╌╍╎╏═║╒╓╔╕╖╗╘╙╚╛╜╝╞╟╠╡╢╣╤╥╦╧╨╩╪╫╬╭╮╯╰╱╲╳╴╵╶╷╸╹╺╻╼╽╾╿"
        blocks = "▀▁▂▃▄▅▆▇█▉▊▋▌▍▎▏▐░▒▓▔▕▖▗▘▙▚▛▜▝▞▟"
        self.content_trans1 = str.maketrans({k:'<BLOCK>' for k in keisen+blocks})

    def __len__(self):
        return len(self.bpe)

    def clean_text(self, content):
        content = self.content_repatter1.sub("<URL>" ,content)
        content = self.content_repatter2.sub("<EMAIL>" ,content)
        content = self.content_repatter3.sub("<TEL>" ,content)
        content = self.content_repatter4.sub("<DATE>" ,content)
        content = self.content_repatter5.sub("<DATE>" ,content)
        content = self.content_repatter6.sub("<PRICE>" ,content)
        content = content.translate(self.content_trans1)
        while '<BLOCK><BLOCK>' in content:
            content = content.replace('<BLOCK><BLOCK>', '<BLOCK>')
        return content

    def encode(self, words, clean=False, position=False):
        replace_words = {}
        def add_replace_words(org, rep):
            if org in words:
                replace_words[org] = rep
        add_replace_words(' ', '<SP>')
        add_replace_words('　', '<SP>')
        add_replace_words('\r\n', '<BR>')
        add_replace_words('\n', '<BR>')
        add_replace_words('\r', '<BR>')
        add_replace_words('\t', '<TAB>')
        add_replace_words('—', 'ー')
        add_replace_words('−', 'ー')
        for k,v in self.emoji['emoji'].items():
                add_replace_words(k, v)
        if clean:
            words = self.clean_text(words)
        def checkkigou(x):
            e = x.encode()
            if len(x) == 1 and len(e)==2:
                c = (int(e[0])<<8)+int(e[1])
                if (c >= 0xc2a1 and c <= 0xc2bf) or (c >= 0xc780 and c <= 0xc783) or (c >= 0xcab9 and c <= 0xcbbf) or (c >= 0xcc80 and c <= 0xcda2):
                    return True
            return False
        def checku2e(x):
            e = x.encode()
            if len(x) == 1 and len(e)==3:
                c = (int(e[0])<<16)+(int(e[1])<<8)+int(e[2])
                if c >= 0xe28080 and c <= 0xe2b07f:
                    return True
            return False

        pos = 0
        result = []
        result_position = []
        while pos < len(words):
            kouho = []
            for k in replace_words.keys():
                if words[pos:pos+len(k)] == k:
                    wd = replace_words[k]
                    kouho.append((self.swe[wd], wd, pos+len(k)))
            if len(kouho) == 0:
                end = min(len(words), pos+self.maxlen+1) if words[pos]=='<' else pos+4
                for e in range(end, pos, -1):
                    if pos>0:
                        p = "##"
                    else:
                        p = ""
                    if e>=len(words):
                        wd = p+words[pos:e]
                        if wd in self.swe:
                            if wd[0]=='<' and len(wd) > 2:
                                kouho = [(self.swe[wd], wd, e)]
                                break
                            else:
                                kouho.append((self.swe[wd], wd, e))

            if len(kouho) > 0:
                wp,wd,e = sorted(kouho, key=lambda x:x[0])[0]
                if len(result)>0 and self.bpe[result[-1]]=='<SP>':
                    result.pop()
                    result_position.pop()
                result.append(wp)
                result_position.append(pos)
                pos = e
            else:
                end = pos+1
                wd = words[pos:end]
                if checkkigou(wd):
                    result.append(self.swe['<KIGOU>'])
                    result_position.append(pos)
                elif checku2e(wd):
                    result.append(self.swe['<U2000U2BFF>'])
                    result_position.append(pos)
                else:
                    for i in wd.encode('utf-8'):
                        result.append(self.swe['<|byte%d|>'%i])
                        result_position.append(pos)
                pos = end
        if position:
            return result, result_position
        else:
            return result

    def decode(self, tokens, breakline='\n'):
        words = []
        byte_tokens = []
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
        for i in tokens:
            word = self.bpe[i]
            if word[:6] == '<|byte' and word[-2:] == '|>':
                byte_tokens.append(int(word[6:-2]))
            else:
                if len(byte_tokens) > 0:
                    words.append(bytearray(byte_tokens).decode('utf-8', errors='replace'))
                    byte_tokens = []
                if word[:7] == '<|emoji' and word[-2:] == '|>':
                    words.append(self.emoji['emoji_inv'][word])
                elif word == '<SP>':
                    words.append(' ')
                elif word == '<BR>':
                    words.append(breakline)
                elif word == '<TAB>':
                    words.append('\t')
                elif word == '<BLOCK>':
                    words.append('▀')
                elif word == '<KIGOU>':
                    words.append('§')
                elif word == '<U2000U2BFF>':
                    words.append('■')
                else:
                    if word.startswith("##"):
                        words.append(word[2:])
                    else:
                        if len(words)>0 and (check_hindi(word[0]) or check_tamil(word[0])):
                            words.append(' ')
                        words.append(word)
        if len(byte_tokens) > 0:
            words.append(bytearray(byte_tokens).decode('utf-8', errors='replace'))
        text = ''.join(words)
        return text

class SWEEncoder_ja:
    def __init__(self, bpe, emoji):
        self.bpe = [[b] if (b==',' or ',' not in b) else b.split(',') for b in bpe]
        self.swe = {}
        for idx, b in enumerate(self.bpe):
            for wd in b:
                self.swe[wd] = idx
        self.emoji = emoji
        self.maxlen = np.max([len(w) for w in self.swe.keys()])
        self.content_repatter1 = re.compile(r"(https?|ftp)(:\/\/[-_\.!~*\'()a-zA-Z0-9;\/?:\@&=\+$,%#]+)")
        self.content_repatter2 = re.compile(r"[A-Za-z0-9\._+]*@[\-_0-9A-Za-z]+(\.[A-Za-z]+)*")
        self.content_repatter3 = re.compile(r'[\(]{0,1}[0-9]{2,4}[\)\-\(]{0,1}[0-9]{2,4}[\)\-]{0,1}[0-9]{3,4}')
        self.content_repatter4 = re.compile(r"([12]\d{3}[/\-年])*(0?[1-9]|1[0-2])[/\-月]((0?[1-9]|[12][0-9]|3[01])日?)*(\d{1,2}|:|\d{1,2}時|\d{1,2}分|\(日\)|\(月\)|\(火\)|\(水\)|\(木\)|\(金\)|\(土\)|㈰|㈪|㈫|㈬|㈭|㈮|㈯)*")
        self.content_repatter5 = re.compile(r"(明治|大正|昭和|平成|令和|㍾|㍽|㍼|㍻|\u32ff)\d{1,2}年(0?[1-9]|1[0-2])月(0?[1-9]|[12][0-9]|3[01])日(\d{1,2}|:|\d{1,2}時|\d{1,2}分|\(日\)|\(月\)|\(火\)|\(水\)|\(木\)|\(金\)|\(土\)|㈰|㈪|㈫|㈬|㈭|㈮|㈯)*")
        self.content_repatter6 = re.compile(r'((0|[1-9]\d*|[1-9]\d{0,2}(,\d{3})+)*億)*((0|[1-9]\d*|[1-9]\d{0,2}(,\d{3})+)*万)*((0|[1-9]\d*|[1-9]\d{0,2}(,\d{3})+)*千)*(0|[1-9]\d*|[1-9]\d{0,2}(,\d{3})+)*(千円|万円|千万円|円|千ドル|万ドル|千万ドル|ドル|千ユーロ|万ユーロ|千万ユーロ|ユーロ)+(\(税込\)|\(税抜\)|\+tax)*')
        keisen = "─━│┃┄┅┆┇┈┉┊┋┌┍┎┏┐┑┒┓└┕┖┗┘┙┚┛├┝┞┟┠┡┢┣┤┥┦┧┨┩┪┫┬┭┮┯┰┱┲┳┴┵┶┷┸┹┺┻┼┽┾┿╀╁╂╃╄╅╆╇╈╉╊╋╌╍╎╏═║╒╓╔╕╖╗╘╙╚╛╜╝╞╟╠╡╢╣╤╥╦╧╨╩╪╫╬╭╮╯╰╱╲╳╴╵╶╷╸╹╺╻╼╽╾╿"
        blocks = "▀▁▂▃▄▅▆▇█▉▊▋▌▍▎▏▐░▒▓▔▕▖▗▘▙▚▛▜▝▞▟"
        self.content_trans1 = str.maketrans({k:'<BLOCK>' for k in keisen+blocks})

    def __len__(self):
        return len(self.bpe)

    def clean_text(self, content):
        content = self.content_repatter1.sub("<URL>" ,content)
        content = self.content_repatter2.sub("<EMAIL>" ,content)
        content = self.content_repatter3.sub("<TEL>" ,content)
        content = self.content_repatter4.sub("<DATE>" ,content)
        content = self.content_repatter5.sub("<DATE>" ,content)
        content = self.content_repatter6.sub("<PRICE>" ,content)
        content = content.translate(self.content_trans1)
        while '<BLOCK><BLOCK>' in content:
            content = content.replace('<BLOCK><BLOCK>', '<BLOCK>')
        return content

    def encode(self, words, clean=False, position=False):
        replace_words = {}
        def add_replace_words(org, rep):
            if org in words:
                replace_words[org] = rep
        add_replace_words(' ', '<SP>')
        add_replace_words('　', '<SP>')
        add_replace_words('\r\n', '<BR>')
        add_replace_words('\n', '<BR>')
        add_replace_words('\r', '<BR>')
        add_replace_words('\t', '<TAB>')
        add_replace_words('—', 'ー')
        add_replace_words('−', 'ー')
        for k,v in self.emoji['emoji'].items():
                add_replace_words(k, v)
        if clean:
            words = self.clean_text(words)
        def checkkigou(x):
            e = x.encode()
            if len(x) == 1 and len(e)==2:
                c = (int(e[0])<<8)+int(e[1])
                if (c >= 0xc2a1 and c <= 0xc2bf) or (c >= 0xc780 and c <= 0xc783) or (c >= 0xcab9 and c <= 0xcbbf) or (c >= 0xcc80 and c <= 0xcda2):
                    return True
            return False
        def checku2e(x):
            e = x.encode()
            if len(x) == 1 and len(e)==3:
                c = (int(e[0])<<16)+(int(e[1])<<8)+int(e[2])
                if c >= 0xe28080 and c <= 0xe2b07f:
                    return True
            return False
        pos = 0
        result = []
        result_position = []
        while pos < len(words):
            kouho = []
            for k in replace_words.keys():
                if words[pos:pos+len(k)] == k:
                    wd = replace_words[k]
                    kouho.append((self.swe[wd], pos+len(k)))
            if len(kouho) == 0:
                end = min(len(words), pos+self.maxlen+1) if words[pos]=='<' else pos+3
                for e in range(end, pos, -1):
                    wd = words[pos:e]
                    if wd in self.swe:
                        if wd[0]=='<' and len(wd) > 2:
                            kouho = [(self.swe[wd], e)]
                            break
                        else:
                            kouho.append((self.swe[wd], e))
            if len(kouho) > 0:
                wp,e = sorted(kouho, key=lambda x:x[0])[0]
                result.append(wp)
                result_position.append(pos)
                pos = e
            else:
                end = pos+1
                wd = words[pos:end]
                if checkkigou(wd):
                    result.append(self.swe['<KIGOU>'])
                    result_position.append(pos)
                elif checku2e(wd):
                    result.append(self.swe['<U2000U2BFF>'])
                    result_position.append(pos)
                else:
                    for i in wd.encode('utf-8'):
                        result.append(self.swe['<|byte%d|>'%i])
                        result_position.append(pos)
                pos = end
        if position:
            return result, result_position
        else:
            return result

    def decode(self, tokens, breakline='\n'):
        words = []
        byte_tokens = []
        for i in tokens:
            word = self.bpe[i][0]
            if word[:6] == '<|byte' and word[-2:] == '|>':
                byte_tokens.append(int(word[6:-2]))
            else:
                if len(byte_tokens) > 0:
                    words.append(bytearray(byte_tokens).decode('utf-8', errors='replace'))
                    byte_tokens = []
                if word[:7] == '<|emoji' and word[-2:] == '|>':
                    words.append(self.emoji['emoji_inv'][word])
                elif word == '<SP>':
                    words.append(' ')
                elif word == '<BR>':
                    words.append(breakline)
                elif word == '<TAB>':
                    words.append('\t')
                elif word == '<BLOCK>':
                    words.append('▀')
                elif word == '<KIGOU>':
                    words.append('ǀ')
                elif word == '<U2000U2BFF>':
                    words.append('‖')
                else:
                    words.append(word)
        if len(byte_tokens) > 0:
            words.append(bytearray(byte_tokens).decode('utf-8', errors='replace'))
        text = ''.join(words)
        return text

def get_encoder(voc_file, emoji_file, wholeword=False):
    assert os.path.exists(voc_file), f"vocabulary file not found in {voc_file}"
    assert os.path.exists(emoji_file), f"emoji file not found in {emoji_file}"
    with open(voc_file, encoding='utf-8') as f:
        bpe = f.read().split('\n')
    with open('emoji.json', encoding='utf-8') as f:
        emoji = json.loads(f.read())
    if not wholeword:
        return SWEEncoder_ja(bpe, emoji)
    else:
        return SWEEncoder_wholeword(bpe, emoji)
