import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
import numpy as np
import pandas as pd
import math
from tqdm import tqdm

from train_utilities import load_x_data_for_cnn, split_train_test_dev, CNN_model, generate_label_from_dead_date, test_dev_auc
import HP


result_csv = pd.read_csv(HP.result_csv)

# split train test dev
train_index, test_index, dev_index = split_train_test_dev(len(result_csv))

# get the label for train test and dev
train_dead_date = result_csv['dead_after_disch_date'].iloc[train_index]
dev_dead_date = result_csv['dead_after_disch_date'].iloc[dev_index]
test_dead_date = result_csv['dead_after_disch_date'].iloc[test_index]

dev_patient_name = np.asarray(result_csv["patient_id"].iloc[dev_index])
test_patient_name = np.asarray(result_csv["patient_id"].iloc[test_index])  # patient0, patient1, ...
train_patient_name = np.asarray(result_csv["patient_id"].iloc[train_index])

y_dev_task = generate_label_from_dead_date(dev_dead_date)  # list of nparray
y_test_task = generate_label_from_dead_date(test_dead_date)
y_train_task = generate_label_from_dead_date(train_dead_date)
  
n_train = len(train_patient_name)
n_dev = len(dev_patient_name)
n_test = len(test_patient_name)

# train CNN model
num_train_batch = int(math.ceil(n_train / float(HP.n_batch)))
num_dev_batch = int(math.ceil(n_dev / float(HP.n_batch)))
num_test_batch = int(math.ceil(n_test / float(HP.n_batch)))



# define placeholders and model
input_ys = []
for i in range(HP.multi_size):
    input_ys.append(tf.placeholder(tf.int32, [None, HP.num_classes], name="input_y"+str(i)))

input_x = tf.placeholder(tf.float32,
                     [None, HP.max_document_length, HP.max_sentence_length, HP.embedding_size],
                     name="input_x")
sent_length = tf.placeholder(tf.int32, [None], name="sent_length")
# category placeholder
category_index = tf.placeholder(tf.int32, [None, HP.max_document_length], name='category_index')
dropout_keep_prob = tf.placeholder(tf.float32, [], name="dropout_keep_prob")
optimize, scores_soft_max_list, scores_sentence_soft_max_list, gradients_sentence_list, _ = CNN_model(input_x, input_ys, sent_length, category_index, dropout_keep_prob)

saver = tf.train.Saver()



# start tf
with tf.Session() as sess:
    if HP.restore:
        saver.restore(sess, HP.model_path)
    else:
        sess.run(tf.global_variables_initializer())
    
    max_auc = 0
    current_early_stop_times = 0
    current_early_stop_times_force = 0
    
    while True:
        y_total_task_label_train = []
        for i in range(len(y_train_task)):
            y_total_task_label_train.extend(np.argmax(y_train_task[i], axis=1).tolist())
        
        pre_sent_train, grad_sent_train, predictions_train, len_sent_train = [],[],[],[]
        # start train
        for i in tqdm(range(num_train_batch)):
            tmp_train_patient_name = train_patient_name[i*HP.n_batch:min((i+1)*HP.n_batch, n_train)]
            tmp_y_train = []
            for t in y_train_task:
                tmp_y_train.append(t[i*HP.n_batch:min((i+1)*HP.n_batch, n_train)])

            feed_dict = load_x_data_for_cnn(tmp_train_patient_name,
                                            HP.drop_out_train,
                                            input_x,
                                            sent_length,
                                            category_index,
                                            dropout_keep_prob)

            for (M, input_y) in enumerate(input_ys):
                feed_dict[input_y] = tmp_y_train[M]
                
            _, pre, pre_sent, grad_sent, len_sent = sess.run([optimize, scores_soft_max_list, scores_sentence_soft_max_list, gradients_sentence_list, sent_length], feed_dict=feed_dict)

            for p in pre_sent:
                for k in p:
                    pre_sent_train.append(k)
            for g in grad_sent:
                for k in g:
                    grad_sent_train.append(k)
            len_sent_train.extend(len_sent)
            
            pre = pre.reshape(-1, HP.num_classes)  # [3*n_batch,2]  in one batch: task1+task2+task3
            pre = pre[:, 1]  # get probability of positive class
            predictions_train.extend(pre.tolist())   # task1,2,3_batch1 + task1,2,3_batch2+ task1,2,3_batch3....
        

        # get validation result
        dev_auc, _, pre_sent_dev, grad_sent_dev, predictions_dev, y_total_task_label_dev, len_sent_dev = test_dev_auc(num_dev_batch, y_dev_task, dev_patient_name, n_dev, sess, input_x, input_ys, sent_length, category_index, dropout_keep_prob, scores_soft_max_list, scores_sentence_soft_max_list, gradients_sentence_list, test_output_flag=False)
        print("Dev AUC: {}".format(dev_auc))
        
        # early stop techniques
        if dev_auc > max_auc:
            save_path = saver.save(sess, HP.model_path)
            max_auc = dev_auc
            current_early_stop_times = 0
        else:
            current_early_stop_times += 1
        if current_early_stop_times >= HP.early_stop_times:
            break
            
        current_early_stop_times_force += 1
        if current_early_stop_times_force >10:
            break
    
#     report the performance on test data
    test_auc, test_auc_per_task, pre_sent_test, grad_sent_test, predictions_test, y_total_task_label_test, len_sent_test = test_dev_auc(num_test_batch, y_test_task, test_patient_name, n_test, sess, input_x, input_ys, sent_length, category_index, dropout_keep_prob, scores_soft_max_list, scores_sentence_soft_max_list, gradients_sentence_list, test_output_flag=True)
    print("Test total AUC: {}".format(test_auc))
    print("Test total AUC: {}".format(test_auc_per_task))



    
def get_score(type_data):

    if type_data =='train':
        return train_patient_name, pre_sent_train, grad_sent_train, len_sent_train, predictions_train, y_total_task_label_train
    if type_data == 'dev':
        return dev_patient_name, pre_sent_dev, grad_sent_dev, len_sent_dev, predictions_dev, y_total_task_label_dev
    if type_data == 'test':
        return test_patient_name, pre_sent_test, grad_sent_test, len_sent_test, predictions_test, y_total_task_label_test

for type_data in ['train', 'dev', 'test']:

    patient_name, pre_sent, grad_sent, len_sent, predictions, y_total_task_label = get_score(type_data)
    f = open('results/{}_scores.txt'.format(type_data), 'w')
    g = open('results/{}_scores_gradient.txt'.format(type_data), 'w')
    
    for p,y in zip(patient_name, y_total_task_label):
        f.write(p+', '+str(y)+'\n')

        idx = list(patient_name).index(p)
        scores = pre_sent[idx][:,1][:len_sent[idx]]  
        for s in scores:
            f.write(str(s) + '\n')
        f.write('\n')
    f.close()
    
    for p,y in zip(patient_name, y_total_task_label):
        g.write(p+', '+str(y)+'\n')

        idx = list(patient_name).index(p)
        scores = np.max(grad_sent[idx], axis=1)[:len_sent[idx]]
        for s in scores:
            g.write(str(s) + '\n')
        g.write('\n')
    g.close()