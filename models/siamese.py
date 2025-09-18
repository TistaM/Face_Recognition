from tensorflow.keras import layers, Model, Input
from tensorflow.keras.applications import ResNet50
import tensorflow as tf

def abs_diff(tensors):
    return tf.abs(tensors[0] - tensors[1])

def build_embedding_model(input_shape=(160, 160, 1)):
    def simple_cnn(x):
        x = layers.Conv2D(64, (10, 10), activation='relu')(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(128, (7, 7), activation='relu')(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(128, (4, 4), activation='relu')(x)
        return layers.Flatten()(x)

    input_img = Input(shape=input_shape)
    cnn_out = simple_cnn(input_img)
    x3 = layers.Concatenate()([input_img]*3)
    resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(160, 160, 3), pooling='avg')
    resnet.trainable = False
    resnet_out = resnet(x3)
    face_out = layers.Conv2D(32, (3, 3), activation='relu')(input_img)
    face_out = layers.GlobalAveragePooling2D()(face_out)
    face_out = layers.Dense(128, activation='relu')(face_out)
    combined = layers.Concatenate()([cnn_out, resnet_out, face_out])
    output = layers.Dense(256, activation='relu')(combined)
    return Model(input_img, output)

def build_siamese_network(input_shape=(160, 160, 1)):
    embed_model = build_embedding_model(input_shape)
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)
    emb_a = embed_model(input_a)
    emb_b = embed_model(input_b)
    diff = layers.Lambda(abs_diff, output_shape=(256,))([emb_a, emb_b])
    out = layers.Dense(1, activation='sigmoid')(diff)
    return Model(inputs=[input_a, input_b], outputs=out)
