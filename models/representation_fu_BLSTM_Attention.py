

from keras.models import load_model,Model
from parserTest import parameter_parser
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, LeakyReLU,Concatenate,Bidirectional
from keras.optimizers import Adamax
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical #转one-hot
from sklearn.metrics import confusion_matrix
from sklearn.utils import compute_class_weight
import tensorflow as tf
from keras import backend as K
from keras import initializers, regularizers, constraints
from keras.layers import Layer
import matplotlib.pyplot as plt
from keras.callbacks import ReduceLROnPlateau
import time

class LearningRateLogger(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self._supports_tf_logs = True

    def on_epoch_end(self, epoch, logs=None):
        if logs is None or "learning_rate" in logs:
            return
        # logs["learning_rate"] = args.lr
        logs["learning_rate"] = self.model.optimizer.lr

def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


class AttentionWithContext(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    follows these equations:
    (1) u_t = tanh(W h_t + b)
    (2) \alpha_t = \frac{exp(u^T u)}{\sum_t(exp(u_t^T u))}, this is the attention weight
    (3) v_t = \alpha_t * h_t, v in time t
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        3D tensor with shape: `(samples, steps, features)`.
    """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero and this results in NaN's.
        # Should add a small epsilon as the workaround
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a

        return weighted_input

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[2]


class Addition(Layer):
    """
    This layer is supposed to add of all activation weight.
    We split this from AttentionWithContext to help us getting the activation weights
    follows this equation:
    (1) v = \sum_t(\alpha_t * h_t)

    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    """

    def __init__(self, **kwargs):
        super(Addition, self).__init__(**kwargs)

    def build(self, input_shape):
        self.output_dim = input_shape[-1]
        super(Addition, self).build(input_shape)

    def call(self, x):
        return K.sum(x, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

args = parameter_parser()

class Fusion_Model_BLSTM_ATT:
    def __init__(self,data_ast,data_codeSlicing,name="", batch_size=args.batch_size, lr=args.lr, epochs=args.epochs, dropout=args.dropout):
        vectors_ast = np.stack(data_ast.iloc[:, 1].values)
        labels_ast = data_ast.iloc[:, 0].values
        positive_idxs_ast = np.where(labels_ast == 1)[0]
        negative_idxs_ast = np.where(labels_ast == 0)[0]
        undersampled_negative_idxs_ast = np.random.choice(negative_idxs_ast, len(positive_idxs_ast), replace=False)
        resampled_idxs_ast = np.concatenate([positive_idxs_ast, undersampled_negative_idxs_ast])
        # print(resampled_idxs.shape)
        x_train_ast,x_test_ast,y_train_ast,y_test_ast = train_test_split(vectors_ast[resampled_idxs_ast], labels_ast[resampled_idxs_ast], train_size=0.8,test_size=0.2,
                                 stratify=labels_ast[resampled_idxs_ast])

        vectors_slicing = np.stack(data_codeSlicing.iloc[:, 1].values)
        labels_slicing = data_codeSlicing.iloc[:, 0].values
        positive_idxs_slicing = np.where(labels_slicing == 1)[0]
        negative_idxs_slicing = np.where(labels_slicing == 0)[0]
        undersampled_negative_idxs_slicing = np.random.choice(negative_idxs_slicing, len(positive_idxs_slicing), replace=False)
        resampled_idxs_slicing = np.concatenate([positive_idxs_slicing, undersampled_negative_idxs_slicing])
        # print(resampled_idxs.shape)
        x_train_slicing,x_test_slicing,y_train_slicing,y_test_slicing = train_test_split(vectors_slicing[resampled_idxs_slicing], labels_slicing[resampled_idxs_slicing], train_size=0.8,test_size=0.2,
                                 stratify=labels_slicing[resampled_idxs_slicing])

        x_train = np.concatenate((x_train_ast, x_train_slicing))
        x_test = np.concatenate((x_test_ast, x_test_slicing))
        y_train = np.concatenate((y_train_ast, y_train_slicing))
        y_test = np.concatenate((y_test_ast, y_test_slicing))


        self.x_train = x_train
        self.x_test = x_test

        self.y_train = to_categorical(y_train)
        self.y_test = to_categorical(y_test)

        self.name = name
        self.batch_size = batch_size
        self.epochs = epochs
        self.class_weight = compute_class_weight(class_weight='balanced', classes=[0, 1],
                                                 y=labels_ast)



        model = Sequential()
        model.add(Bidirectional(LSTM(300, return_sequences=True), input_shape=(x_train.shape[1], x_train.shape[2])))
        model.add(AttentionWithContext())
        model.add(Addition())
        model.add(Dense(300))
        model.add(LeakyReLU())
        model.add(Dropout(dropout))
        model.add(Dense(300))
        model.add(LeakyReLU())
        model.add(Dropout(dropout))
        model.add(Dense(2, activation='softmax'))

        adamax = Adamax(lr)
        metric = self.get_lr_metric(adamax)
        model.compile(adamax, 'categorical_crossentropy', metrics=['accuracy',metric])
        self.model = model

    def get_lr_metric(self,optimizer):
        def lr(y_true, y_pred):
            return optimizer.lr
        return lr


    def train_test(self):
        train_start_time = time.time()
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.00001)
        callbacks = [reduce_lr,LearningRateLogger()]
        history = self.model.fit(self.x_train, self.y_train, batch_size=self.batch_size, epochs=self.epochs,
                             class_weight=self.class_weight, validation_data=(self.x_test, self.y_test),callbacks=callbacks)
        train_end_time = time.time()
        print('model train time：')
        print(train_end_time-train_start_time)
        startTime = time.time()
        values = self.model.evaluate(self.x_test, self.y_test, batch_size=self.batch_size)
        print("Accuracy:",values[1])
        endTime = time.time()
        print('model test time：')
        print(endTime-startTime)
        predictions = (self.model.predict(self.x_test, batch_size=self.batch_size)).round()
        tn,fp,fn,tp = confusion_matrix(np.argmax(self.y_test, axis=1), np.argmax(predictions, axis=1)).flatten()
        print("False positive rate(FP):", fp / (fp + tn))
        print("False negative rate(FN):", fn / (fn + tp))
        recall = tp / (tp + fn)
        print("Recall:", recall)
        precision = tp / (tp + fp)
        print("Precision: ",precision)
        print("F1 score: ",(2 * precision * recall)/(precision + recall))

        loss = history.history['loss']
        acc = history.history['acc']
        val_loss = history.history['val_loss']
        val_acc = history.history['val_acc']
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, 'b',linestyle='--', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.plot(epochs,acc,'r',linestyle='--', label='Training accuracy')
        plt.plot(epochs,val_acc,'r',label='Validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.ylabel('Acc')
        plt.legend()
        plt.show()