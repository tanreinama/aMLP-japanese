import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

def activation(x):
    return x * torch.sigmoid(x)

def softmax(x, axis=-1):
    return torch.nn.functional.softmax(x, dim=axis)

class embed(Module):
    def __init__(self, num_vocab, num_hidden, w_init_stdev=0.02):
        super(embed, self).__init__()
        wb = torch.normal(mean=torch.zeros([num_vocab, num_hidden]), std=w_init_stdev)
        self.w = Parameter(wb, requires_grad=True)

    def forward(self, x, num_ctx):
        num_vocab, num_hidden = self.w.shape[-2:]
        if num_ctx > 0:
            x_flat = torch.reshape(x, [-1, num_hidden])
            w = torch.transpose(self.w, 0, 1)
            logits = torch.matmul(x_flat, w)
            return torch.reshape(logits, [-1, num_ctx, num_vocab])
        else:
            bs = x.shape[0]
            x_flat = torch.reshape(x, [-1])
            oh = torch.nn.functional.one_hot(x_flat, num_classes=num_vocab)
            logits = torch.matmul(oh.float(), self.w)
            return torch.reshape(logits, [bs, -1, num_hidden])

class fc(Module):
    def __init__(self, n_inputs, n_feats, w_init_stdev=0.02):
        super(fc, self).__init__()
        nx = n_inputs
        nf = n_feats
        wb = torch.normal(mean=torch.zeros([nx, nf]), std=w_init_stdev)
        self.w = Parameter(wb)
        bb = torch.zeros([nf])
        self.b = Parameter(bb)

    def forward(self, x):
        w = self.w
        b = self.b
        c = torch.matmul(x,w)+b
        return c

class projection(Module):
    def __init__(self, nx, nf, w_init_stdev=0.02):
        super(projection, self).__init__()
        wb = torch.normal(mean=torch.zeros([nx, nf]), std=w_init_stdev)
        self.w = Parameter(wb)
        bb = torch.zeros([nf])
        self.b = Parameter(bb)

    def forward(self, x):
        w = self.w
        b = self.b
        shape = list(x.shape)
        bs = -1 if shape[0] is None else shape[0]
        nx = shape[-1]
        nf = b.shape[0]
        x = torch.reshape(x, [-1, nx])
        c = torch.matmul(x, w)+b
        c = torch.reshape(c, (bs, shape[1], nf))
        return c

class normalization(Module):
    def __init__(self, nx):
        super(normalization, self).__init__()
        gb = torch.ones([nx])
        self.g = Parameter(gb)
        bb = torch.zeros([nx])
        self.b = Parameter(bb)

    def forward(self, x, axis=-1, epsilon=1e-5):
        g = self.g
        b = self.b
        u = torch.mean(x, axis=axis, keepdim=True)
        s = torch.mean(torch.square(x-u), axis=axis, keepdim=True)
        x = (x - u) * torch.rsqrt(s + epsilon)
        x = x*g + b
        return x

class softattention(Module):
    def __init__(self, num_hidden, num_hidden_att, num_projection):
        super(softattention, self).__init__()
        self.c = projection(num_hidden, num_hidden_att*3)
        self.p = projection(num_hidden_att, num_projection)

    def forward(self, x, score_weights):
        # [batch, sequence, features]
        c = self.c(x)
        nx = c.shape[-1] // 3
        query, key, value = torch.split(c, nx, dim=2)
        # Self-Attention
        tkey = torch.transpose(key, 1, 2)
        scores = torch.matmul(query, tkey)
        scores = torch.multiply(scores, torch.rsqrt(torch.tensor(nx).float()))
        if score_weights is not None:
            scores += score_weights
        probs = softmax(scores)
        context = torch.matmul(probs, value)
        context = self.p(context)
        return context

class gmlp(Module):
    def __init__(self, num_hidden, num_ctx, num_projection):
        super(gmlp, self).__init__()
        self.norm = normalization(num_hidden)
        self.proj = projection(num_hidden, num_ctx*2)
        self.sgu_norm = normalization(num_ctx)
        self.proj_c = projection(num_ctx, num_projection)
        wb = torch.normal(mean=torch.zeros([1, num_ctx, num_ctx]), std=1e-5)
        self.sgu_w = Parameter(wb)
        bb = torch.ones([num_ctx])
        self.sgu_b = Parameter(bb)

    def sgu(self, x, weight):
        x = self.sgu_norm(x)
        w = self.sgu_w
        b = self.sgu_b
        shape = x.shape
        bs = -1 if shape[0] is None else shape[0]
        nx = shape[-1]
        weight = torch.reshape(weight, [-1, nx, 1])
        x = torch.multiply(x, weight)
        x = torch.transpose(x, 1, 2)
        weight = torch.reshape(weight, [-1, 1, nx])
        w = torch.multiply(w, weight)
        w = torch.reshape(w, [1, -1, nx, nx])
        c = torch.matmul(x, w)+b
        c = torch.reshape(c, (bs, shape[1], nx))
        c = torch.transpose(c, 1, 2)
        return c

    def forward(self, x, weight=None, att=None):
        shape = x.shape
        nx = shape[-1]
        nq = shape[1]
        bs = -1 if shape[0] is None else shape[0]
        a = self.norm(x)
        a = self.proj(a)
        a = activation(a)
        # [batch, sequence, sequence]
        u, v = torch.split(a, nq, dim=2)
        v = self.sgu(v, weight)
        if att is not None:
            v = v + att
        proj = torch.multiply(u, v)
        # [batch, sequence, features]
        proj = self.proj_c(proj)
        return proj

class block(Module):
    def __init__(self, num_hidden, num_ctx, num_soft_att, use_att):
        super(block, self).__init__()
        self.b = gmlp(num_hidden=num_hidden, num_ctx=num_ctx, num_projection=num_hidden)
        self.s = softattention(num_hidden, num_soft_att, num_ctx) if use_att else None

    def forward(self, x, weight, att_weight):
        skip = x
        s = self.s(x, score_weights=att_weight)
        x = self.b(x, weight=weight, att=s)
        x = activation(x)
        x = x + skip
        return x

class position(Module):
    def __init__(self, num_context, num_hidden, w_init_stdev=0.01):
        super(position, self).__init__()
        wb = torch.normal(mean=torch.zeros([num_context, num_hidden]), std=w_init_stdev)
        self.v = Parameter(wb)

    def forward(self, x):
        return x + self.v

class model(Module):
    def __init__(self, num_ctx, num_vocab, num_hidden, num_soft_att, num_layer, has_voc, classifier_n, is_squad):
        super(model, self).__init__()
        self.nq = num_ctx
        self.embed = embed(num_vocab, num_hidden)
        self.num_layer = num_layer
        ml = []
        for i,_ in enumerate(range(num_layer)):
            x = block(num_hidden=num_hidden, num_ctx=num_ctx, num_soft_att=num_soft_att, use_att=num_soft_att!=0)
            ml.append(x)
        self.layer = torch.nn.ModuleList(ml)
        self.last_proj = projection(num_hidden, num_hidden)
        self.vocabrary_embed = Parameter(torch.zeros([num_vocab])) if has_voc else None
        self.output = fc(num_hidden, classifier_n) if classifier_n>0 else None
        self.squad_output = projection(num_hidden, 2) if is_squad else None

    def attention_mask(self, weight):
        shape = weight.shape
        bs = -1 if shape[0] is None else shape[0]
        nq = shape[1]
        mask = torch.reshape(weight, [bs, 1, nq])
        ones = torch.ones([nq, 1]).to(weight.device)
        mask = ones * mask
        return mask

    def forward(self, x, weight):
        att_mask = self.attention_mask(weight)
        adder = 1.0 - att_mask
        att_weight = adder * -1e7

        x = self.embed(x, 0)
        for i in range(self.num_layer):
            x = self.layer[i](x, weight, att_weight)
        o = x
        x = self.last_proj(o)
        x = self.embed(x, self.nq)
        if self.vocabrary_embed is not None:
            x = x + self.vocabrary_embed
        if self.output is None and self.squad_output is None:
            return o, x
        elif self.squad_output is None:
            s = self.output(o[:,0,:])
            return o, x, s
        else:
            s = self.squad_output(o)
            return o, x, s
