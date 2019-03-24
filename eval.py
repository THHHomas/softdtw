# coding=utf-8
import os

import tensorflow as tf
from keras.layers import Lambda, Concatenate,Flatten,AveragePooling2D, GlobalAveragePooling2D
from keras import backend as K
from keras.engine import Model
from keras.models import load_model
from keras.preprocessing import image

from baseline.evaluate import train_predict, test_predict, grid_result_eval, market_result_eval
#from transfer.simple_rank_transfer import cross_entropy_loss


#
SN=4
PN= 18

def train_pair_predict(pair_model_path, target_train_path, pid_path, score_path):
    model = load_model(pair_model_path)
    model = Model(inputs=[model.get_layer('resnet50').get_input_at(0)],
                  outputs=[model.get_layer('resnet50').get_output_at(0)])
    train_predict(model, target_train_path, pid_path, score_path)


def triplet_loss(y_true, y_pred):
    #y_pred=Lambda(K.print_tensor)(y_pred)
    y_pred = K.l2_normalize(y_pred,axis=1)
    
    batch = 0
    ref1 = y_pred[0:batch,:]
    pos1 = y_pred[batch:batch+batch,:]
    neg1 = y_pred[batch+batch:3*batch,:]
    dis_pos = K.sum(K.square(ref1 - pos1), axis=1, keepdims=True)
    dis_neg = K.sum(K.square(ref1 - neg1), axis=1, keepdims=True)
    dis_pos = K.sqrt(dis_pos)
    dis_neg = K.sqrt(dis_neg)
    a1 = 0.8
    d1 = K.maximum(0.0,dis_pos-dis_neg+a1)
    return K.mean(d1)

def triplet_hard_loss(y_true, y_pred):
    global SN  # the number of images in a class
    global PN  # the number of class
    feat_num = SN*PN # images num
    y_pred = K.l2_normalize(y_pred,axis=1)
    feat1 = K.tile(K.expand_dims(y_pred,axis = 0),[feat_num,1,1])
    feat2 = K.tile(K.expand_dims(y_pred,axis = 1),[1,feat_num,1])
    delta = feat1 - feat2
    dis_mat = K.sum(K.square(delta),axis = 2) + K.epsilon() # Avoid gradients becoming NAN
    dis_mat = K.sqrt(dis_mat)
    positive = dis_mat[0:SN,0:SN]
    negetive = dis_mat[0:SN,SN:]
    for i in range(1,PN):
        positive = tf.concat([positive,dis_mat[i*SN:(i+1)*SN,i*SN:(i+1)*SN]],axis = 0)
        if i != PN-1:
            negs = tf.concat([dis_mat[i*SN:(i+1)*SN,0:i*SN],dis_mat[i*SN:(i+1)*SN, (i+1)*SN:]],axis = 1)
        else:
            negs = tf.concat(dis_mat[i*SN:(i+1)*SN, 0:i*SN],axis = 0)
        negetive = tf.concat([negetive,negs],axis = 0)
    positive = K.max(positive,axis=1)
    negetive = K.min(negetive,axis=1) 
    a1 = 0.6
    loss = K.mean(K.maximum(0.0,positive-negetive+a1))
    return loss


def test_pair_predict(pair_model_path, target_probe_path, target_gallery_path, pid_path, score_path):
    # todo
    print("before  load_model:----------------------------------------------------------------- ")
    model = load_model("./market_pair_pretrain.h5", custom_objects={'triplet_hard_loss': triplet_hard_loss})
    print("model finished:----------------------------------------------------------------- ")
    # model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    #model = Model(inputs=[model.layers[0].get_input_at(0)],
    #              outputs=[model.get_layer('ave_pooled_part').get_output_at(0)])
    for layer in model.layers:
        print("!!!!:",layer)
    print("@@@@@",model.layers[-5])
    #model = Model(inputs=[model.layers[0].get_input_at(0)],
    #              outputs=[model.layers[-4].get_output_at(0)] )

    global_feature = model.layers[-5].get_output_at(0)
    feature1 = GlobalAveragePooling2D(name='ave_pooled')(global_feature)

    model = Model(inputs=[model.input], outputs=[feature1])
    # model = Model(inputs=[model.input], outputs=[model.get_layer('avg_pool').output])
    print("before  test_predict:----------------------------------------------------------------- ")
    test_predict(model, target_probe_path, target_gallery_path, pid_path, score_path)



def tf_eucl_dist(inputs):
    x, y = inputs
    return K.square((x - y))

def avg_eucl_dist(inputs):
    x, y = inputs
    return K.mean(K.square((x - y)), axis=1)


def train_rank_predict(rank_model_path, target_train_path, pid_path, score_path):
    model = load_model(rank_model_path, custom_objects={'cross_entropy_loss': cross_entropy_loss})
    model = Model(inputs=[model.get_layer('resnet50').get_input_at(0)],
                  outputs=[model.get_layer('resnet50').get_output_at(0)])
    train_predict(model, target_train_path, pid_path, score_path)


def test_rank_predict(rank_model_path, target_probe_path, target_gallery_path, pid_path, score_path):
    model = load_model(rank_model_path, custom_objects={'cross_entropy_loss': cross_entropy_loss})
    model = Model(inputs=[model.get_layer('resnet50').get_input_at(0)],
                  outputs=[model.get_layer('resnet50').get_output_at(0)])
    test_predict(model, target_probe_path, target_gallery_path, pid_path, score_path)


def grid_eval(source, transform_dir):
    target = 'grid'
    for i in range(10):
        test_pair_predict(source + '_pair_pretrain.h5',
                          transform_dir + 'cross%d' % i + '/probe', transform_dir + 'cross%d' % i + '/test',
                          source + '_' + target + '_pid.log', source + '_' + target + '_score.log')
        grid_result_eval(source + '_' + target + '_pid.log', 'gan.log')


def market_eval(source, transform_dir):
    target = 'market'
    test_pair_predict(source + '_model.h5',
                           transform_dir + '/query', transform_dir + '/bounding_box_test',	
                          source + '_' + target + '_pid.log', source + '_' + target + '_score.log')


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    market_eval('market', '../dataset/Market-1501-v15.09.15')
    market_result_eval('market_market_pid.log',
                        TEST='../dataset/Market-1501-v15.09.15/bounding_box_test',
                        QUERY='../dataset/Market-1501-v15.09.15/query')
    # grid_eval('market', '/home/person/grid_train_probe_gallery/cross0')
    #grid_result_eval('/home/person/TrackViz/data/market_grid-cv0-test/cross_filter_pid.log')


