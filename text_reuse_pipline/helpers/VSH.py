import tensorflow as tf
import numpy as np
import os, zipfile
import shutil


def xavier_init(fan_in, fan_out, constant=1): 
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), 
                             minval=low, maxval=high, 
                             dtype=tf.float32)
class Dense(object):
    
    def __init__(self, output_dim, activation, bias=True):
        self.output_dim = output_dim
        self.activation = activation
        self.has_build = False
        self.bias = bias
        
    def build(self, input_shapes):
        input_dim = input_shapes[1]
        self.W = tf.Variable(xavier_init(input_dim, self.output_dim))
        self.b = tf.Variable(tf.zeros(self.output_dim))
        
    def __call__(self, x):
        if not self.has_build:
            shape = x.get_shape()
            shape = tuple([i.__int__() for i in shape])
            
            # Handle when the input is 1D
            if len(shape) == 1:
                self.build([0, shape[0]])
            else:
                self.build(shape)
            self.has_build = True
            
        if self.activation == 'softplus':
            transfer_fct = tf.nn.softplus
        elif self.activation == 'sigmoid':
            transfer_fct = tf.sigmoid
        elif self.activation == 'tanh':
            transfer_fct = tf.tanh
        elif self.activation == 'relu':
            transfer_fct = tf.nn.relu
        elif self.activation == 'relu6':
            transfer_fct = tf.nn.relu6
        elif self.activation == 'elu':
            transfer_fct = tf.nn.elu
        elif self.activation == 'linear':
            transfer_fct = None
        else:
            assert('Unknown activation function.')
            transfer_fct = None
        
        if self.bias == True:
            if transfer_fct is None:
                return tf.add(tf.matmul(x, self.W), self.b)
            else:
                return transfer_fct(tf.add(tf.matmul(x, self.W), self.b))
        else:
            if transfer_fct is None:
                return tf.matmul(x, self.W)
            else:
                return transfer_fct(tf.matmul(x, self.W))

####################################################################################################################
## Unsupervised Learning model
####################################################################################################################

class VDSH(object):
    def __init__(self, sess, latent_dim, n_feas, n_hidden_dim=500):
        self.sess = sess
        self.n_feas = n_feas
        
        self.latent_dim = latent_dim
        
        n_batches = 1
        self.n_batches = n_batches
        
        self.hidden_dim = n_hidden_dim
        self.build()
    
    def transform(self, docs):
        z_data = []
        for i in range(len(docs)):
            doc = docs[i]
            word_indice = np.where(doc > 0)[0]
            z = self.sess.run(self.z_mean, 
                           feed_dict={ self.input_bow: doc.reshape((-1, self.n_feas)),
                                       self.input_bow_idx: word_indice,
                                       self.keep_prob: 1.0})
            z_data.append(z[0])
        return z_data

    def bin_transform(self, X):
        b_x = self.transform(X)

        binary_code = np.zeros(X.shape)
        for i in range(self.latent_dim):
            binary_code[np.nonzero(b_x[:,i] < self.threshold[i]),i] = 0
            binary_code[np.nonzero(b_x[:,i] >= self.threshold[i]),i] = 1
        return binary_code.astype(int)

    def calc_reconstr_error(self):
        # Pick score for those visiable words
        p_x_i_scores0 = tf.gather(self.p_x_i, self.input_bow_idx)
        weight_scores0 = tf.gather(tf.squeeze(self.input_bow), self.input_bow_idx)
        return -tf.reduce_sum(tf.log(tf.maximum(p_x_i_scores0 * weight_scores0, 1e-10)))

    def calc_KL_loss(self):
        return -0.5 * tf.reduce_sum(tf.reduce_sum(1 + self.z_log_var - tf.square(self.z_mean) 
                                              - tf.exp(self.z_log_var), axis=1))
    
    def build(self):
        with tf.name_scope('input'):
            # BOW
            self.input_bow = tf.placeholder(tf.float32, [1, self.n_feas], name="Input_BOW")
            # indices
            self.input_bow_idx = tf.placeholder(tf.int32, [None], name="Input_bow_Idx")

            self.kl_weight = tf.placeholder(tf.float32, name="KL_Weight")
            self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        ## Inference network q(z|x)
        with tf.name_scope("dense_layer_1"):
            self.z_enc_1 = Dense(self.hidden_dim, activation='relu')(self.input_bow)
        with tf.name_scope("dense_layer_2"):
            self.z_enc_2 = Dense(self.hidden_dim, activation='relu')(self.z_enc_1)
        with tf.name_scope("drop_out"):
            self.z_enc_3 = tf.nn.dropout(self.z_enc_2, keep_prob=self.keep_prob)
        
        with tf.name_scope("z_mean"):
            self.z_mean = Dense(self.latent_dim, activation='linear')(self.z_enc_3)
        with tf.name_scope("z_log_var"):
            self.z_log_var = Dense(self.latent_dim, activation='sigmoid')(self.z_enc_3)
        
        # Sampling Layers X
        with tf.name_scope("sampling"):
            self.eps_z = tf.random_normal((self.n_batches, self.latent_dim), 0, 1, dtype=tf.float32)
            self.z_sample = self.z_mean + tf.sqrt(tf.exp(self.z_log_var)) * self.eps_z
        
        # Decoding Layers
        with tf.name_scope("decoding_layer"):
            self.R = tf.Variable(tf.random_normal([self.n_feas, self.latent_dim]), name="R_Mat")
            self.b = tf.Variable(tf.zeros([self.n_feas]), name="B_Mat")
            self.e = -tf.matmul(self.z_sample, self.R, transpose_b=True) + self.b
            self.p_x_i = tf.squeeze(tf.nn.softmax(self.e))
        
        self.reconstr_err = self.calc_reconstr_error()
        self.kl_loss = self.calc_KL_loss()
        
        with tf.name_scope("cost"):
            self.cost = self.reconstr_err + self.kl_weight * self.kl_loss

class VDSHLoader(object):
    def __init__(self, model_path, model_name, threshold, no_features, no_hidden_dim, latent_dim):
        self._vdsh = None
        self._model_path = model_path
        self._model_name = model_name
        self._no_features= no_features
        self._latent_dim = latent_dim
        self._hidden_dim = no_hidden_dim
        self._bin_threshold  = threshold

    def unzip(self, filename, name):
        try:
            zip = zipfile.ZipFile(filename)
            zip.extractall(path=name)
            return True
        except OSError as e:
            print('execption file exist : ' + filename)
            print(e)
            return False

    def _get_file_path(self):
        from pyspark import SparkFiles
        return SparkFiles.get(self._model_path)

    def _get_model(self):
        if self._vdsh is None:
            path = self._get_file_path()
            if not os.path.exists(path.replace('.zip', '')):
                print('try to unzip:' + path + ' to ' + path.replace('.zip', ''))
                self.unzip(path, path.replace('.zip', ''))
            
            tf.reset_default_graph()
            sess  = tf.Session()
            self._vdsh = VDSH(sess, self._latent_dim, self._no_features, self._hidden_dim)
            saver = tf.train.Saver(tf.global_variables(), reshape=True)
            saver.restore(sess, path.replace('.zip', '') + '/' + self._model_name)
            return self._vdsh
        else:
            return self._vdsh
 
    def binarize(self, vector):
        binary_code = np.zeros(vector.shape)
        for i in range(self._latent_dim):
            binary_code[np.nonzero(vector[:,i] < self._bin_threshold[i]),i] = 0
            binary_code[np.nonzero(vector[:,i] >= self._bin_threshold[i]),i] = 1
        return binary_code.astype(int)
    
    def transform(self, tfidf):
        model    = self._get_model()
        latent_v = model.transform(tfidf)
        latent_v = np.array(latent_v)
        bin_v = self.binarize(latent_v)
        return bin_v

    def cleanup(self):
        try:
            print('cleaning up....')
            path = self._get_file_path()
            shutil.rmtree(path)
            shutil.rmtree(path.replace('.zip', ''))
        except OSError as err:
            print(err)