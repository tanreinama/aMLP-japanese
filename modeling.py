import tensorflow as tf

def activation(x):
    return x * tf.math.sigmoid(x)

def softmax(x, axis=-1):
    ex = tf.exp(x - tf.reduce_max(x, axis=axis, keepdims=True))
    return ex / tf.reduce_sum(ex, axis=axis, keepdims=True)

class embed(tf.keras.layers.Layer):
    def __init__(self, num_vocab, num_hidden, name, w_init_stdev=0.02):
        super(embed, self).__init__(name=name)
        wb = tf.random.normal([num_vocab, num_hidden], stddev=w_init_stdev)
        self.w = tf.Variable(wb, name=f'{name}_w')

    def call(self, x, num_ctx):
        if num_ctx > 0:
            num_vocab, num_hidden = self.w.shape.as_list()[-2:]
            x_flat = tf.reshape(x, [-1, num_hidden])
            logits = tf.matmul(x_flat, self.w, transpose_b=True)
            return tf.reshape(logits, [-1, num_ctx, num_vocab])
        else:
            return tf.gather(params=self.w, indices=x)

class fc(tf.keras.layers.Layer):
    def __init__(self, n_inputs, n_feats, name, w_init_stdev=0.02):
        super(fc, self).__init__(name=name)
        nx = n_inputs
        nf = n_feats
        wb = tf.random.normal([nx, nf], stddev=w_init_stdev)
        self.w = tf.Variable(wb, name=f'{name}_w')
        bb = tf.zeros([nf])
        self.b = tf.Variable(bb, name=f'{name}_b')

    def call(self, x):
        w = self.w
        b = self.b
        c = tf.matmul(x,w)+b
        return c

class projection(tf.keras.layers.Layer):
    def __init__(self, nx, nf, name, w_init_stdev=0.02):
        super(projection, self).__init__(name=name)
        wb = tf.random.normal([nx, nf], stddev=w_init_stdev)
        self.w = tf.Variable(wb, name=f'{name}_w')
        bb = tf.zeros([nf])
        self.b = tf.Variable(bb, name=f'{name}_b')

    def call(self, x):
        w = self.w
        b = self.b
        shape = x.shape.as_list()
        bs = -1 if shape[0] is None else shape[0]
        nx = shape[-1]
        nf = b.shape.as_list()[0]
        x = tf.reshape(x, [-1, nx])
        c = tf.matmul(x, w)+b
        c = tf.reshape(c, (bs, shape[1], nf))
        return c

class normalization(tf.keras.layers.Layer):
    def __init__(self, nx, name):
        super(normalization, self).__init__(name=name)
        gb = tf.ones([nx])
        self.g = tf.Variable(gb, name=f'{name}_g')
        bb = tf.zeros([nx])
        self.b = tf.Variable(bb, name=f'{name}_b')

    def call(self, x, axis=-1, epsilon=1e-5):
        g = self.g
        b = self.b
        u = tf.reduce_mean(x, axis=axis, keepdims=True)
        s = tf.reduce_mean(tf.square(x-u), axis=axis, keepdims=True)
        x = (x - u) * tf.math.rsqrt(s + epsilon)
        x = x*g + b
        return x

class softattention(tf.keras.layers.Layer):
    def __init__(self, num_hidden, num_hidden_att, num_projection, name):
        super(softattention, self).__init__(name=name)
        self.c = projection(num_hidden, num_hidden_att*3, f'{name}_c')
        self.p = projection(num_hidden_att, num_projection, f'{name}_p')

    def call(self, x, score_weights):
        # [batch, sequence, features]
        c = self.c(x)
        query, key, value = tf.split(c, 3, axis=2)
        nx = query.shape.as_list()[-1]
        # Self-Attention
        scores = tf.matmul(query, key, transpose_b=True)
        scores = tf.multiply(scores, tf.math.rsqrt(tf.cast(nx, tf.float32)))
        if score_weights is not None:
            scores += score_weights
        probs = softmax(scores)
        context = tf.matmul(probs, value)
        context = self.p(context)
        return context

class multiheadattention(tf.keras.layers.Layer):
    def __init__(self, n_inputs, n_feats, n_heads, name, dp_gmlp):
        super(multiheadattention, self).__init__(name=name)
        nx = n_inputs
        nf = n_feats
        self.query = projection(nx, nx, name+'_q')
        self.key = projection(nx, nx, name+'_k')
        self.value = projection(nx, nx, name+'_v')
        self.projection = projection(nx, nf, name+'_p')
        self.nf = n_feats
        self.nh = n_heads
        self.dp = tf.keras.layers.Dropout(dp_gmlp) if dp_gmlp>0 else None

    def call(self, x, score_weights=None):
        nf = self.nf
        nh = self.nh
        shape = x.shape.as_list()
        bs = -1 if shape[0] is None else shape[0]
        nq = shape[1]
        nx = shape[-1]
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        query = tf.reshape(query, (bs, nq, nh, nx//nh))
        key = tf.reshape(key, (bs, nq, nh, nx//nh))
        value = tf.reshape(value, (bs, nq, nh, nx//nh))
        query = tf.transpose(query, [0, 2, 1, 3])
        key = tf.transpose(key, [0, 2, 1, 3])
        value = tf.transpose(value, [0, 2, 1, 3])
        scores = tf.matmul(query, key, transpose_b=True)
        scores = tf.multiply(scores, tf.math.rsqrt(tf.cast(nx, tf.float32)))
        if score_weights is not None:
            scores += score_weights
        probs = tf.nn.softmax(scores)
        if self.dp is not None:
            probs = self.dp(probs)
        context = tf.matmul(probs, value)
        context = tf.transpose(context, [0, 2, 1, 3])
        context = tf.reshape(context, (bs, nq, nx))
        context = self.projection(context)
        return context

class transformer(tf.keras.layers.Layer):
    def __init__(self, num_hidden, num_heads, name, dp, dp_gmlp):
        super(transformer, self).__init__(name=name)
        self.a = multiheadattention(num_hidden, num_hidden, num_heads, f'{name}_att', dp_gmlp)
        self.na = normalization(num_hidden, f'{name}_norm_a')
        self.p1 = projection(num_hidden, num_hidden, f'{name}_proj1')
        self.p2 = projection(num_hidden, num_hidden, f'{name}_proj2')
        self.nv = normalization(num_hidden, f'{name}_norm_v')
        self.dp = tf.keras.layers.Dropout(dp) if dp>0 else None

    def call(self, x, att_weight):
        skip1 = x
        x = self.a(x, att_weight)
        x = x + skip1
        x = self.na(x)
        skip2 = x
        x = self.p1(x)
        x = activation(x)
        x = self.p2(x) + skip2
        x = self.nv(x)
        if self.dp is not None:
            x = self.dp(x)
        return x

class gmlp(tf.keras.layers.Layer):
    def __init__(self, num_hidden, num_ctx, num_projection, name, dp_gmlp):
        super(gmlp, self).__init__(name=name)
        self.n = normalization(num_hidden, f'{name}_norm')
        self.p = projection(num_hidden, num_ctx*2, f'{name}_proj')
        self.nv = normalization(num_ctx, f'{name}_sgu_norm')
        self.c = projection(num_ctx, num_projection, f'{name}_proj_c')
        self.dp = tf.keras.layers.Dropout(dp_gmlp) if dp_gmlp>0 else None
        wb = tf.random.normal([1, num_ctx, num_ctx], stddev=1e-5)
        self.w = tf.Variable(wb, name=f'{name}_sgu_w')
        bb = tf.ones([num_ctx])
        self.b = tf.Variable(bb, name=f'{name}_sgu_b')

    def sgu(self, x, weight):
        x = self.nv(x)
        w = self.w
        b = self.b
        shape = x.shape.as_list()
        bs = -1 if shape[0] is None else shape[0]
        nx = shape[-1]
        weight = tf.reshape(weight, [-1, nx, 1])
        x = tf.multiply(x, weight)
        x = tf.transpose(x, [0,2,1])
        weight = tf.reshape(weight, [-1, 1, nx])
        w = tf.multiply(w, weight)
        w = tf.reshape(w, [1, -1, nx, nx])
        if self.dp is not None:
            w = self.dp(w)
        c = tf.matmul(x, w)+b
        c = tf.reshape(c, (bs, shape[1], nx))
        c = tf.transpose(c, [0,2,1])
        return c

    def call(self, x, weight=None, att=None):
        shape = x.shape.as_list()
        nx = shape[-1]
        nq = shape[1]
        bs = -1 if shape[0] is None else shape[0]
        a = self.n(x)
        a = self.p(a)
        a = activation(a)
        # [batch, sequence, sequence]
        u, v = tf.split(a, 2, axis=2)
        v = self.sgu(v, weight)
        if att is not None:
            v = v + att
        proj = tf.multiply(u, v)
        # [batch, sequence, features]
        proj = self.c(proj)
        return proj

class block(tf.keras.layers.Layer):
    def __init__(self, num_hidden, num_ctx, num_soft_att, name, dp, dp_gmlp, use_att):
        super(block, self).__init__(name=name)
        self.m = gmlp(num_hidden=num_hidden, num_ctx=num_ctx, num_projection=num_hidden, name=f'{name}_b', dp_gmlp=dp_gmlp)
        self.s = softattention(num_hidden, num_soft_att, num_ctx, f'{name}_s') if use_att else None
        self.dp = tf.keras.layers.Dropout(dp) if dp>0 else None

    def call(self, x, weight, att_weight):
        skip = x
        s = self.s(x, score_weights=att_weight)
        x = self.m(x, weight=weight, att=s)
        x = activation(x)
        if self.dp is not None:
            x = self.dp(x)
        x = x + skip
        return x

class position(tf.keras.layers.Layer):
    def __init__(self, num_context, num_hidden, name, w_init_stdev=0.01):
        super(position, self).__init__(name=name)
        wb = tf.random.normal([num_context, num_hidden], stddev=w_init_stdev)
        self.w = tf.Variable(wb, name=name+'_v')

    def call(self, x):
        return x + self.w

class model(tf.keras.Model):
    def __init__(self, num_ctx, num_vocab, num_hidden, num_soft_att, num_layer, dropout_prob, mlp_dropout_prob, type="mlp"):
        super(model, self).__init__(name='amlp')
        self.nq = num_ctx
        self.e = embed(num_vocab, num_hidden, name='embed')
        input_ids = tf.keras.Input(dtype=tf.float32, shape=[num_ctx, num_hidden])
        input_weights = tf.keras.Input(dtype=tf.float32, shape=[num_ctx])
        att_mask = self.attention_mask(input_weights)
        adder = 1.0 - tf.cast(att_mask, tf.float32)
        att_weight = adder * -1e7
        x = input_ids
        if type=="mlp":
            for i,_ in enumerate(range(num_layer)):
                x = block(num_hidden=num_hidden, num_ctx=num_ctx, num_soft_att=num_soft_att, name=f'layer_{i}',
                    dp=dropout_prob, dp_gmlp=mlp_dropout_prob, use_att=num_soft_att!=0)(x, input_weights, att_weight)
        else:
            self.wb = position(num_ctx, num_hidden, name='position_embed')
            x = self.wb(x)
            att_weight = tf.expand_dims(att_weight, axis=[1])
            num_heads = num_hidden // num_soft_att
            for i,_ in enumerate(range(num_layer)):
                x = transformer(num_hidden=num_hidden, num_heads=num_heads, name=f'layer_{i}',
                    dp=dropout_prob, dp_gmlp=mlp_dropout_prob)(x, att_weight)
        self.m = tf.keras.models.Model(inputs=[input_ids, input_weights], outputs=x)
        self.l = projection(num_hidden, num_hidden, name='last_proj')
        self.v = tf.Variable(tf.zeros([num_vocab], dtype=tf.float32), name='vocabrary_embed')

    def attention_mask(self, weight):
        shape = weight.shape.as_list()
        bs = -1 if shape[0] is None else shape[0]
        nq = shape[1]
        mask = tf.cast(tf.reshape(weight, [bs, 1, nq]), tf.float32)
        ones = tf.ones(shape=[nq, 1], dtype=tf.float32)
        mask = ones * mask
        return mask

    def call(self, inputs):
        x, weight = inputs
        x = self.e(x, 0)
        o = self.m([x, weight])
        x = self.l(o)
        x = self.e(x, self.nq)
        x = x + self.v
        return [o, x]
