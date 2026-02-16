# MODEL REUSIT 1. 
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import (
#     Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense,
#     Dropout, Input, BatchNormalization
# )
# from tensorflow.keras.regularizers import l2

# def build_multi_writer_model(input_shape=(128, 256, 1)):
#     weight_decay = 1e-4  

#     inputs = Input(shape=input_shape)

#     x = Conv2D(32, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(weight_decay))(inputs)
#     x = BatchNormalization()(x)
#     x = MaxPooling2D((2, 2))(x)

#     x = Conv2D(64, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(weight_decay))(x)
#     x = BatchNormalization()(x)
#     x = MaxPooling2D((2, 2))(x)

#     x = Conv2D(128, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(weight_decay))(x)
#     x = BatchNormalization()(x)
#     x = MaxPooling2D((2, 2))(x)

#     x = GlobalAveragePooling2D()(x)
#     x = Dropout(0.5)(x)

#     x = Dense(128, activation="relu", kernel_regularizer=l2(weight_decay))(x)
#     x = Dropout(0.4)(x)

#     outputs = Dense(1, activation="sigmoid")(x)

#     model = Model(inputs=inputs, outputs=outputs)
#     return model

# model 2 si 3 (seturi de date diferit)
# import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import (
#     Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense,
#     Dropout, Input, BatchNormalization, Activation,
#     SeparableConv2D
# )
# from tensorflow.keras.regularizers import l2

# def build_multi_writer_model(input_shape=(256, 128, 1)):
#     """   
#     Args:
#         input_shape: Forma imaginii de intrare (height, width, channels)
        
#     Returns:
#         model: Model Keras compilat
        
#     Labels:
#         0 = Un singur autor (single-writer)
#         1 = Mai mulți autori (multi-writer)
#     """
#     weight_decay = 1e-3
    
#     inputs = Input(shape=input_shape, name="handwriting_input")
    
#     # === BLOCK 1: Detectarea caracteristicilor de bază ===
#     x = Conv2D(32, (3, 3), padding="same", kernel_regularizer=l2(weight_decay), 
#                name="conv1_1")(inputs)
#     x = BatchNormalization(name="bn1_1")(x)
#     x = Activation("relu", name="relu1_1")(x)
    
#     x = Conv2D(32, (3, 3), padding="same", kernel_regularizer=l2(weight_decay), 
#                name="conv1_2")(x)
#     x = BatchNormalization(name="bn1_2")(x)
#     x = Activation("relu", name="relu1_2")(x)
#     x = MaxPooling2D((2, 2), name="pool1")(x)
#     x = Dropout(0.45, name="dropout1")(x)
    
#     # === BLOCK 2: Caracteristici de nivel mediu ===
#     x = SeparableConv2D(64, (3, 3), padding="same", kernel_regularizer=l2(weight_decay), 
#                         name="sepconv2_1")(x)
#     x = BatchNormalization(name="bn2_1")(x)
#     x = Activation("relu", name="relu2_1")(x)
    
#     x = SeparableConv2D(64, (3, 3), padding="same", kernel_regularizer=l2(weight_decay), 
#                         name="sepconv2_2")(x)
#     x = BatchNormalization(name="bn2_2")(x)
#     x = Activation("relu", name="relu2_2")(x)
#     x = MaxPooling2D((2, 2), name="pool2")(x)
#     x = Dropout(0.4, name="dropout2")(x)
    
#     # === BLOCK 3: Caracteristici de nivel înalt ===
#     x = SeparableConv2D(128, (3, 3), padding="same", kernel_regularizer=l2(weight_decay), 
#                         name="sepconv3_1")(x)
#     x = BatchNormalization(name="bn3_1")(x)
#     x = Activation("relu", name="relu3_1")(x)
    
#     x = SeparableConv2D(128, (3, 3), padding="same", kernel_regularizer=l2(weight_decay), 
#                         name="sepconv3_2")(x)
#     x = BatchNormalization(name="bn3_2")(x)
#     x = Activation("relu", name="relu3_2")(x)
#     x = MaxPooling2D((2, 2), name="pool3")(x)
#     x = Dropout(0.35, name="dropout3")(x)
    
#     # === BLOCK 4: Caracteristici abstracte ===
#     x = Conv2D(256, (3, 3), padding="same", kernel_regularizer=l2(weight_decay), 
#                name="conv4")(x)
#     x = BatchNormalization(name="bn4")(x)
#     x = Activation("relu", name="relu4")(x)
    
#     # === GLOBAL POOLING ===
#     x = GlobalAveragePooling2D(name="global_avg_pool")(x)
    
#     # === CLASSIFIER ===
#     x = Dropout(0.5, name="dropout_classifier1")(x)
#     x = Dense(256, kernel_regularizer=l2(weight_decay), name="dense1")(x)
#     x = BatchNormalization(name="bn_dense1")(x)
#     x = Activation("relu", name="relu_dense1")(x)
    
#     x = Dropout(0.5, name="dropout_classifier2")(x)
#     x = Dense(128, kernel_regularizer=l2(weight_decay), name="dense2")(x)
#     x = BatchNormalization(name="bn_dense2")(x)
#     x = Activation("relu", name="relu_dense2")(x)
    
#     x = Dropout(0.4, name="dropout_final")(x)
    
#     # === OUTPUT LAYER ===
#     outputs = Dense(1, activation="sigmoid", name="multi_writer_prediction")(x)
    
#     model = Model(inputs=inputs, outputs=outputs, name="MultiWriterDetector")
    
#     return model

# def compile_model(model, learning_rate=0.001):
#     """
#     Compilează modelul cu optimizer și metrici optimizate
#     """
#     optimizer = tf.keras.optimizers.Adam(
#         learning_rate=learning_rate,
#         beta_1=0.9,
#         beta_2=0.999,
#         epsilon=1e-7
#     )
    
#     model.compile(
#         optimizer=optimizer,
#         loss='binary_crossentropy',
#         metrics=[
#             'accuracy',
#             tf.keras.metrics.Precision(name='precision'),
#             tf.keras.metrics.Recall(name='recall'),
#             tf.keras.metrics.AUC(name='auc')
#         ]
#     )
    
#     return model

# # Test function
# if __name__ == "__main__":
#     print("=== TESTING MODEL ARCHITECTURE ===")
#     model = build_multi_writer_model(input_shape=(256, 128, 1))
#     model = compile_model(model)
#     model.summary()
    
#     print(f"\nModel name: {model.name}")
#     print(f"Total parameters: {model.count_params():,}")
#     print(f"Input shape: {model.input_shape}")
#     print(f"Output shape: {model.output_shape}")

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense,
    Dropout, Input, BatchNormalization, Activation,
    SeparableConv2D, SpatialDropout2D
)
from tensorflow.keras.regularizers import l2

def build_multi_writer_model(input_shape=(256, 128, 1)):
    """
    Construiește un model CNN regularizat pentru detecția scrisului de mână multi-writer.
    """
    weight_decay = 1e-4
    inputs = Input(shape=input_shape, name="handwriting_input")

    # Block 1
    x = Conv2D(32, (3, 3), padding="same", kernel_regularizer=l2(weight_decay))(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((2, 2))(x)
    x = SpatialDropout2D(0.2)(x)

    # Block 2
    x = SeparableConv2D(
        64, (3, 3), padding="same",
        depthwise_regularizer=l2(weight_decay),
        pointwise_regularizer=l2(weight_decay)
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((2, 2))(x)
    x = SpatialDropout2D(0.25)(x)

    # Block 3
    x = SeparableConv2D(
        96, (3, 3), padding="same",
        depthwise_regularizer=l2(weight_decay),
        pointwise_regularizer=l2(weight_decay)
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((2, 2))(x)
    x = SpatialDropout2D(0.3)(x)

    # Block 4
    x = Conv2D(128, (3, 3), padding="same", kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # Global Pooling
    x = GlobalAveragePooling2D()(x)

    # Classifier
    x = Dropout(0.4)(x)
    x = Dense(128, kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Dropout(0.3)(x)
    x = Dense(64, kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Dropout(0.2)(x)

    # Output
    outputs = Dense(1, activation="sigmoid", name="multi_writer_prediction")(x)

    model = Model(inputs=inputs, outputs=outputs, name="MultiWriterDetector")
    return model



def compile_model(model, learning_rate=0.001):
    """
    Compilează modelul cu optimizer și metrici optimizate
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    return model
