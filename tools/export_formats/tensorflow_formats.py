"""TensorFlow family, from the same weights rebuilt in Keras / TF1:

* Keras 3 `.keras`, legacy HDF5 full model `.h5` (model_config + weights), weights-only `.weights.h5`
* TF2 object checkpoint `<name>.ckpt.index` + `.ckpt.data-00000-of-00001`
* SavedModel directory (`saved_model.pb` graph + variables/ in the same tensor-bundle checkpoint format)
* TFLite: float32, float16 weights, dynamic-range int8 weights
* TF1 checkpoint with graph: `<name>_tf1.ckpt.{index,data-*,meta}`, variables named like OpenAI's original GPT-2 release

The Keras model uses built-in layers except two small registered ones: GPT2Embeddings (token + position tables; called a
second time with reverse=True it is the tied LM head) and GeluTanh (GPT-2's tanh GELU). HF Conv1D weights are [in, out],
which is Keras' Dense kernel layout, so nothing is transposed; the fused c_attn is split into MultiHeadAttention's q/k/v."""
import math
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import keras  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2  # noqa: E402

from .common import Manifest, compare, hf_state_dict, load_source, parse_args, reference_logits  # noqa: E402

args = parse_args(__doc__)
model_pt, ids_pt = load_source(args.model)
ref = reference_logits(model_pt, ids_pt)
W = {k: v.numpy() for k, v in hf_state_dict(args.model, model_pt).items()}
c, ids = model_pt.config, ids_pt.numpy().astype(np.int32)
out, m, name = args.out / "tensorflow", Manifest(args, "tensorflow"), args.model
out.mkdir(exist_ok=True)
D, H, Dh, L, eps = c.hidden_size, c.num_heads, c.head_dim, c.num_layers, c.layer_norm_epsilon


@keras.saving.register_keras_serializable(package="tensormorph")
class GPT2Embeddings(keras.layers.Layer):
    def __init__(self, vocab_size, max_positions, hidden_size, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size, self.max_positions, self.hidden_size = vocab_size, max_positions, hidden_size

    def build(self, input_shape):
        self.wte = self.add_weight(name="wte", shape=(self.vocab_size, self.hidden_size))
        self.wpe = self.add_weight(name="wpe", shape=(self.max_positions, self.hidden_size))

    def call(self, inputs, reverse=False):
        if reverse:  # tied LM head: hidden [B,T,D] · wteᵀ -> [B,T,V]
            return keras.ops.matmul(inputs, keras.ops.transpose(self.wte))
        positions = keras.ops.arange(keras.ops.shape(inputs)[1])
        return keras.ops.take(self.wte, inputs, axis=0) + keras.ops.take(self.wpe, positions, axis=0)[None]

    def get_config(self):
        return super().get_config() | {"vocab_size": self.vocab_size, "max_positions": self.max_positions, "hidden_size": self.hidden_size}


@keras.saving.register_keras_serializable(package="tensormorph")
class GeluTanh(keras.layers.Layer):
    def call(self, x):
        return keras.activations.gelu(x, approximate=True)


def build_keras():
    ids_in = keras.Input(shape=(None,), dtype="int32", name="input_ids")
    embeddings = GPT2Embeddings(c.vocab_size, c.max_position_embeddings, D, name="embeddings")
    x = embeddings(ids_in)
    for i in range(L):
        h = keras.layers.LayerNormalization(epsilon=eps, name=f"h{i}_ln_1")(x)
        a = keras.layers.MultiHeadAttention(H, Dh, name=f"h{i}_attn")(h, h, use_causal_mask=True)
        x = keras.layers.Add(name=f"h{i}_attn_residual")([x, a])
        h = keras.layers.LayerNormalization(epsilon=eps, name=f"h{i}_ln_2")(x)
        h = keras.layers.Dense(c.intermediate_size, name=f"h{i}_mlp_c_fc")(h)
        h = GeluTanh(name=f"h{i}_mlp_gelu")(h)
        h = keras.layers.Dense(D, name=f"h{i}_mlp_c_proj")(h)
        x = keras.layers.Add(name=f"h{i}_mlp_residual")([x, h])
    x = keras.layers.LayerNormalization(epsilon=eps, name="ln_f")(x)
    return keras.Model(ids_in, embeddings(x, reverse=True), name=name)


def load_weights(km):
    km.get_layer("embeddings").wte.assign(W["transformer.wte.weight"])
    km.get_layer("embeddings").wpe.assign(W["transformer.wpe.weight"])
    for i in range(L):
        p = f"transformer.h.{i}."
        for ln in ("ln_1", "ln_2"):
            km.get_layer(f"h{i}_{ln}").set_weights([W[p + ln + ".weight"], W[p + ln + ".bias"]])
        qkv_w, qkv_b = W[p + "attn.c_attn.weight"], W[p + "attn.c_attn.bias"]  # [D, 3D] = [in, q|k|v]
        attn = km.get_layer(f"h{i}_attn")
        for j, dense in enumerate((attn._query_dense, attn._key_dense, attn._value_dense)):
            dense.kernel.assign(qkv_w[:, j * D:(j + 1) * D].reshape(D, H, Dh))
            dense.bias.assign(qkv_b[j * D:(j + 1) * D].reshape(H, Dh))
        attn._output_dense.kernel.assign(W[p + "attn.c_proj.weight"].reshape(H, Dh, D))
        attn._output_dense.bias.assign(W[p + "attn.c_proj.bias"])
        for dense in ("c_fc", "c_proj"):
            km.get_layer(f"h{i}_mlp_{dense}").set_weights([W[p + f"mlp.{dense}.weight"], W[p + f"mlp.{dense}.bias"]])
    km.get_layer("ln_f").set_weights([W["transformer.ln_f.weight"], W["transformer.ln_f.bias"]])
    return km


km = load_weights(build_keras())
saved_model_dir = out / f"{name}_saved_model"


def keras_in_memory():
    return [], compare(ref, km(ids))  # the rebuilt model itself; every file below is checked against PyTorch too


def keras_v3():
    p = out / f"{name}.keras"
    km.save(p)
    return [p], compare(ref, keras.saving.load_model(p)(ids))


def keras_h5_full():
    p = out / f"{name}.h5"
    km.save(p)  # legacy HDF5: root attrs model_config + model_weights group
    return [p], compare(ref, keras.saving.load_model(p)(ids))


def keras_weights_h5():
    p = out / f"{name}.weights.h5"
    km.save_weights(p)
    fresh = build_keras()
    fresh.load_weights(p)
    return [p], compare(ref, fresh(ids))


def tf2_checkpoint():
    prefix = out / f"{name}.ckpt"
    tf.train.Checkpoint(model=km).write(str(prefix))
    fresh = build_keras()
    tf.train.Checkpoint(model=fresh).read(str(prefix)).assert_existing_objects_matched()
    return [out / f"{name}.ckpt.index", out / f"{name}.ckpt.data-00000-of-00001"], compare(ref, fresh(ids))


def saved_model():
    km.export(str(saved_model_dir), verbose=False)
    loaded = tf.saved_model.load(str(saved_model_dir))  # keep a reference: serve() needs its variables alive
    return [saved_model_dir], compare(ref, loaded.serve(ids))


def tflite(suffix, configure):
    def fn():
        # static [2, 8]: with a dynamic sequence the causal-mask slicing in MultiHeadAttention needs Flex (TF Select) ops
        # and weights frozen into constants (otherwise they stay uninitialized resource variables: READ_VARIABLE ops)
        fixed = tf.function(lambda x: km(x)).get_concrete_function(tf.TensorSpec(ids.shape, tf.int32, name="input_ids"))
        converter = tf.lite.TFLiteConverter.from_concrete_functions([convert_variables_to_constants_v2(fixed)])
        configure(converter)
        p = out / f"{name}.{suffix}.tflite"
        p.write_bytes(converter.convert())
        interpreter = tf.lite.Interpreter(model_path=str(p))
        index = interpreter.get_input_details()[0]["index"]
        interpreter.resize_tensor_input(index, ids.shape)
        interpreter.allocate_tensors()
        interpreter.set_tensor(index, ids)
        interpreter.invoke()
        return [p], compare(ref, interpreter.get_tensor(interpreter.get_output_details()[0]["index"]))
    return fn


def fp16(converter):
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]


def dynamic_int8(converter):
    converter.optimizations = [tf.lite.Optimize.DEFAULT]


def tf1_checkpoint_with_meta():
    """TF1 graph mode: placeholders, variables, Saver -> .index + .data + .meta (MetaGraphDef with the full GraphDef)."""
    prefix = out / f"{name}_tf1.ckpt"
    v1 = tf.compat.v1
    graph = tf.Graph()
    with graph.as_default():
        feeds = {}

        def var(scope_name, value):  # initial value fed at init time, so the .meta holds no weight constants
            ph = v1.placeholder(tf.float32, value.shape, name=f"init/{scope_name}")
            feeds[ph] = value
            return tf.Variable(ph, name=scope_name)

        def layer_norm(x, scope):
            g, b = var(f"{scope}/g", W[f"{hf_scope[scope]}.weight"]), var(f"{scope}/b", W[f"{hf_scope[scope]}.bias"])
            mean, variance = tf.nn.moments(x, axes=[-1], keepdims=True)
            return (x - mean) * tf.math.rsqrt(variance + eps) * g + b

        def dense(x2d, scope, hf):
            return tf.matmul(x2d, var(f"{scope}/w", W[f"{hf}.weight"])) + var(f"{scope}/b", W[f"{hf}.bias"])

        hf_scope = {"model/ln_f": "transformer.ln_f"} | {f"model/h{i}/ln_{j}": f"transformer.h.{i}.ln_{j}" for i in range(L) for j in (1, 2)}
        ids_ph = v1.placeholder(tf.int32, [None, None], name="input_ids")
        batch, seq = tf.shape(ids_ph)[0], tf.shape(ids_ph)[1]
        wte = var("model/wte", W["transformer.wte.weight"])
        x = tf.gather(wte, ids_ph) + tf.gather(var("model/wpe", W["transformer.wpe.weight"]), tf.range(seq))[None]
        visible = tf.cast(tf.linalg.band_part(tf.ones([seq, seq]), -1, 0), tf.bool)
        for i in range(L):
            s, hf = f"model/h{i}", f"transformer.h.{i}"
            h = tf.reshape(layer_norm(x, f"{s}/ln_1"), [-1, D])
            qkv = tf.reshape(dense(h, f"{s}/attn/c_attn", f"{hf}.attn.c_attn"), [batch, seq, 3, H, Dh])
            q, k, v = (tf.transpose(qkv[:, :, j], [0, 2, 1, 3]) for j in range(3))  # [B,H,T,Dh]
            scores = tf.matmul(q, k, transpose_b=True) / math.sqrt(Dh)
            probs = tf.nn.softmax(tf.where(visible, scores, tf.fill(tf.shape(scores), -np.inf)), axis=-1)
            context = tf.reshape(tf.transpose(tf.matmul(probs, v), [0, 2, 1, 3]), [-1, D])
            x = x + tf.reshape(dense(context, f"{s}/attn/c_proj", f"{hf}.attn.c_proj"), [batch, seq, D])
            h = dense(tf.reshape(layer_norm(x, f"{s}/ln_2"), [-1, D]), f"{s}/mlp/c_fc", f"{hf}.mlp.c_fc")
            h = 0.5 * h * (1.0 + tf.tanh(math.sqrt(2.0 / math.pi) * (h + 0.044715 * tf.pow(h, 3))))
            x = x + tf.reshape(dense(h, f"{s}/mlp/c_proj", f"{hf}.mlp.c_proj"), [batch, seq, D])
        final = tf.reshape(layer_norm(x, "model/ln_f"), [-1, D])
        tf.reshape(tf.matmul(final, wte, transpose_b=True), [batch, seq, c.vocab_size], name="logits")
        saver = v1.train.Saver()
        with v1.Session(graph=graph) as sess:
            sess.run(v1.global_variables_initializer(), feeds)
            saver.save(sess, str(prefix))
    with tf.Graph().as_default() as g2, v1.Session(graph=g2) as sess:  # restore from the files alone
        v1.train.import_meta_graph(str(prefix) + ".meta").restore(sess, str(prefix))
        logits = sess.run("logits:0", {"input_ids:0": ids})
    files = [out / f"{name}_tf1.ckpt.{ext}" for ext in ("index", "data-00000-of-00001", "meta")] + [out / "checkpoint"]
    return files, compare(ref, logits)


m.variant("keras", "rebuilt model (in memory)", "-", keras_in_memory, "reference for the files below; nothing written")
m.variant("keras", "Keras 3 .keras (zip)", "Keras config", keras_v3, "zip: config.json + metadata.json + model.weights.h5")
m.variant("hdf5", "Keras full model .h5", "Keras config (model_config attr)", keras_h5_full, "legacy HDF5 format")
m.variant("hdf5", "Keras weights only .weights.h5", "none", keras_weights_h5, "no architecture: needs the code or a template")
m.variant("tf-checkpoint", "TF2 object checkpoint .ckpt", "none (object graph only)", tf2_checkpoint)
m.variant("tf-checkpoint", "TF1 checkpoint .ckpt + .meta", "TF1 GraphDef in .meta", tf1_checkpoint_with_meta,
          "variable names like OpenAI's GPT-2 release (model/h0/attn/c_attn/w); weights are [in, out]")
m.variant("savedmodel", "SavedModel directory", "TF GraphDef (saved_model.pb)", saved_model)
m.variant("tflite", "float32", "TFLite subgraph", tflite("float32", lambda converter: None), "static input [2, 8]")
m.variant("tflite", "float16 weights", "TFLite subgraph", tflite("float16", fp16), "static input [2, 8]; weights fp16 + DEQUANTIZE ops", tol=1e-2)
m.variant("tflite", "dynamic-range int8 weights", "TFLite subgraph", tflite("int8_dynamic", dynamic_int8), "static input [2, 8]; int8 weights, float activations", tol=0.1)
m.write()
