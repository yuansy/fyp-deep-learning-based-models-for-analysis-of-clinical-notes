import tensorflow as tf
import numpy as np
import random
import math
from sklearn.metrics import roc_auc_score
from multiprocessing import Pool

from preprocess_utilities import load_embedding
import HP


def split_train_test_dev(n_patient):
    """Split the data into three subsets.
       Args:
            n_patient: the number of sample
       return:
            the index for each subset
    """
    index_ls = list(range(n_patient))
    random.Random(2).shuffle(index_ls)
    train_index = index_ls[:int(math.ceil(n_patient*0.8))]
    test_index = index_ls[int(math.ceil(n_patient*0.8)):int(math.ceil(n_patient*0.9))]
    dev_index = index_ls[int(math.ceil(n_patient*0.9)):]

    return train_index, test_index, dev_index


def generate_label_from_dead_date(y_series):
    """Generate the classification label"""
    labels = []
    for dead_date in HP.tasks_dead_date:
        label = []
        for index, y in y_series.iteritems():
            if y < dead_date:
                label.append([0, 1])
            else:
                label.append([1, 0])
        label = np.asarray(label)
        labels.append(label)
    return labels


def CNN_model(input_x, input_ys, sent_length, category_index, dropout_keep_prob):
    """Two-level CNN architecture"""
    
    # category lookup
    target_embeddings = tf.get_variable(
                        name="target_embeddings",
                        dtype=tf.float32,
                        shape=[HP.n_category, HP.dim_category])
    embedded_category = tf.nn.embedding_lookup(target_embeddings,
                                               category_index,
                                               name="target_embeddings")  # [n_batch, n_doc,dim_category]

    # =============================== reshape to do word level CNN ============================================================== #
    x = tf.reshape(input_x, [-1, HP.max_sentence_length, HP.embedding_size])
    pooled_outputs = []
    for i, filter_size in enumerate(HP.filter_sizes):
        with tf.name_scope("conv-maxpool-%s" % filter_size):
            # Convolution Layer
            filter_shape = [filter_size, HP.embedding_size, HP.num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[HP.num_filters]), name="b")
            conv = tf.nn.conv1d(
                value=x,
                filters=W,
                stride=1,
                padding="VALID"
            )  # shape: (n_batch*n_doc) * (n_seq - filter_size) * num_filters
            # Apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")  # shape not change

            # Maxpooling over the outputs
            # another implementation of max-pool
            pooled = tf.reduce_max(h, axis=1)  # (n_batch*n_doc) * n_filter
            pooled_outputs.append(pooled)  # three list of pooled array
    # Combine all the pooled features
    num_filters_total = HP.num_filters * len(HP.filter_sizes)
    h_pool = tf.concat(pooled_outputs, 1)  # shape: (n_batch*n_doc) * num_filters_total
    # Add dropout
    with tf.name_scope("dropout"):
        h_drop = tf.nn.dropout(h_pool, dropout_keep_prob)  # (n_batch * n_doc) * num_filters_total

    first_cnn_output = tf.reshape(h_drop, [-1, HP.max_document_length, num_filters_total])  # [n_batch, n_doc, n_filter]
    first_cnn_output = tf.concat([first_cnn_output, embedded_category], axis=2)  # [n_batch, n_doc, n_filter + dim_category]
    h_drop = tf.reshape(first_cnn_output,[-1, (num_filters_total+HP.dim_category)])  # [(n_batch * n_doc), n_filter + dim_category]

# do sentence loss with the matrix of the concat result of category & h_drop
    total_loss = 0
    scores_sentence_soft_max_list =[]
    gradients_sentence_list = []
    
    for (M, input_y) in enumerate(input_ys):
        with tf.name_scope("task"+str(M)):
            W = tf.Variable(tf.truncated_normal(
                [(num_filters_total+HP.dim_category), HP.num_classes], stddev=0.1),
                name="W")
            b = tf.Variable(tf.constant(0.1, shape=[HP.num_classes]), name="b")

            scores_sentence = tf.nn.xw_plus_b(h_drop, W, b)
            # scores has shape: [(n_batch * n_doc), num_classes]

            # input_y: shape: [n_batch, num_classes]  have to transfer the same shape to scores
            y = tf.tile(input_y, [1, HP.max_document_length])  # y: shape [n_batch, (num_classes*n_doc)]
            y = tf.reshape(y, [-1, HP.num_classes])  # y: shape [(n_batch*n_doc), num_classes]

            
            scores_sentence_soft_max = tf.nn.softmax(scores_sentence) # [(n_batch * n_doc), num_classes]
            scores_sentence_soft_max = tf.reshape(scores_sentence_soft_max, [-1, HP.max_document_length, HP.num_classes]) # [n_batch, n_doc, num_classes]
#             mask = tf.sequence_mask(sent_length)
#             scores_sentence_soft_max = tf.boolean_mask(scores_sentence_soft_max, mask)
            scores_sentence_soft_max_list.append(scores_sentence_soft_max)  
            
            
            sentence_losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores_sentence, labels=y)
            # sentence losses has shape: [(n_batch * n_doc), ] it is a 1D vector.
            
            gradients_sentence = tf.abs(tf.gradients(sentence_losses, [h_drop])) # [(n_batch * n_doc), n_filter + dim_category]
#             gradients_sentence = tf.reduce_max(gradients_sentence, axis=0) # [(n_batch * n_doc)]
            gradients_sentence = tf.reshape(gradients_sentence, [-1, HP.max_document_length, num_filters_total+HP.dim_category]) # [n_batch, n_doc, n_filter + dim_category]
            gradients_sentence_list.append(gradients_sentence)  
            
            
            sentence_losses = tf.reshape(sentence_losses, [-1, HP.max_document_length]) # [n_batch, n_doc]
            mask = tf.sequence_mask(sent_length)
            sentence_losses = tf.boolean_mask(sentence_losses, mask)
            sentence_losses_avg = tf.reduce_mean(sentence_losses)
            
            
            total_loss += sentence_losses_avg * HP.lambda_regularizer_strength

# =========================================== sentence-level CNN ==================================================================
    filter_shape = [HP.document_filter_size, (num_filters_total + HP.dim_category), HP.document_num_filters]
    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[HP.document_num_filters]), name="b")
    conv = tf.nn.conv1d(
        value=first_cnn_output,
        filters=W,
        stride=1,
        padding="VALID"
    ) # n_batch * (n_max_doc - filter+1) * doc_num_filters
    # Apply nonlinearity
    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
    # Maxpooling over the outputs
    # another implementation of max-pool
    pooled_second = tf.reduce_max(h, axis=1)  # n_batch * document_num_filters
    patient_vector = pooled_second
    with tf.name_scope("dropout"):
        pooled_second_drop = tf.nn.dropout(pooled_second, dropout_keep_prob)

    scores_soft_max_list = []
    for (M,input_y) in enumerate(input_ys):
        with tf.name_scope("task"+str(M)):
            '''
            W_fully = tf.Variable(tf.truncated_normal([HP.document_num_filters, HP.document_num_filters], stddev=0.1), name="W_fully")
            b_fully = tf.Variable(tf.constant(0.1, shape=[HP.document_num_filters]), name="b_fully")
            scores_2 = tf.nn.xw_plus_b(pooled_second_drop, W_fully, b_fully) # n_batch * document_num_filters

            with tf.name_scope("dropout_second"):
                scores_drop = tf.nn.dropout(scores_2, dropout_keep_prob)
            '''
            W = tf.Variable(tf.truncated_normal([HP.document_num_filters, HP.num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[HP.num_classes]), name="b")

            scores = tf.nn.xw_plus_b(pooled_second_drop, W, b)
            # scores_2 has shape: [n_batch, num_classes]
            scores_soft_max = tf.nn.softmax(scores)
            scores_soft_max_list.append(scores_soft_max)  # scores_soft_max_list shape:[multi_size, n_batch, num_classes]
            # predictions = tf.argmax(scores, axis=1, name="predictions")
            # predictions has shape: [None, ]. A shape of [x, ] means a vector of size x
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=input_y)
            # losses has shape: [None, ]
            # include target replication
            # total_loss += losses
            loss_avg = tf.reduce_mean(losses)
            total_loss += loss_avg
    # avg_loss = tf.reduce_mean(total_loss)
    # optimize function
    optimizer = tf.train.AdamOptimizer(learning_rate=HP.learning_rate)
    optimize = optimizer.minimize(total_loss)
    scores_soft_max_list = tf.stack(scores_soft_max_list, axis=0)
    scores_sentence_soft_max_list = tf.stack(scores_sentence_soft_max_list, axis=0)
    # correct_predictions = tf.equal(predictions, tf.argmax(input_y, 1))
    # accuracy = tf.reduce_sum(tf.cast(correct_predictions, "float"), name="accuracy")

    return optimize, scores_soft_max_list, scores_sentence_soft_max_list, gradients_sentence_list, patient_vector

def load_x_data_for_cnn(patient_name, keep_prob, input_x, sent_length, category_index, dropout_keep_prob):
    """Multiprocessing to fast speed"""
    pool = Pool(processes=HP.read_data_thread_num)
    generate_token_embedding_results = pool.map(generate_token_embedding, patient_name)
    pool.close()
    pool.join()

    tmp_x = np.zeros([len(generate_token_embedding_results),
                            HP.n_max_sentence_num,
                            HP.n_max_word_num,
                            HP.embedding_size], dtype=np.float32)
    l = []
    tmp_cate = []
    for (M, r) in enumerate(generate_token_embedding_results):
        tmp_x[M] = r[0]
        l.append(r[1])
        tmp_cate.append(r[2])

    cate_id = np.stack(tmp_cate)
    l = np.asarray(l)
    feed_dict = {input_x: tmp_x,
                 sent_length: l,
                 category_index: cate_id,
                 dropout_keep_prob: keep_prob}
    return feed_dict


def generate_token_embedding(pid):
    """Preprocessing the note and generate embedding for each word in the sentences
    
    Args:
        pid: patient id
    return: 
        x_doc: 3d embedding
        number_of_sentences: seq length
        categories_id_per_file: category of note
    """
    mimic3_embedding = load_embedding()
    
    x_doc = np.zeros([HP.n_max_sentence_num,
                      HP.n_max_word_num,
                      HP.embedding_size], dtype=np.float32)
    current_sentence_ind = 0
    f = open(HP.data_directory + pid + '.txt')
    categories_id_per_file = []
    waiting_for_new_sentence_flag = True
    for line in f:
        strip_line = line.strip()
        if len(strip_line) == 0:
            waiting_for_new_sentence_flag = True
            if current_word_ind > 0:
                current_sentence_ind += 1
                if current_sentence_ind >= HP.n_max_sentence_num:
                    break
            continue
        if waiting_for_new_sentence_flag:  # is new category line
            categories_id_per_file.append(int(strip_line))
            waiting_for_new_sentence_flag = False
            current_word_ind = 0
        else:  # is new word line
            if current_word_ind < HP.n_max_word_num:
                x_doc[current_sentence_ind][current_word_ind] = mimic3_embedding[strip_line]
                current_word_ind += 1
    f.close()
    
    number_of_sentences = len(categories_id_per_file)
    categories_id_per_file = categories_id_per_file + [0]*(HP.n_max_sentence_num-number_of_sentences)
    return x_doc, number_of_sentences, categories_id_per_file


def test_dev_auc(num_batch, y_task, patient_name, n, sess, 
                 input_x, input_ys, sent_length, category_index, dropout_keep_prob, scores_soft_max_list, 
                 scores_sentence_soft_max_list, gradients_sentence_list, test_output_flag):
    """Calculate AUC for the trained model on test and dev datasets"""
    
    y_total_task_label = []
    predictions = []
    pre_sent = []
    grad_sent = []
    len_sent = []
    
    seperate_pre = {}
    y_seperate_task_label = {}
    auc_per_task = {}
    for m in range(HP.multi_size):
        seperate_pre[m] = []
        y_seperate_task_label[m] = []

    for i in range(num_batch):
        tmp_patient_name = patient_name[i*HP.n_batch:min((i+1)*HP.n_batch, n)]
        tmp_y_task_feed = []

        for (y_i,y) in enumerate(y_task):
            tmp_y_task = y[i*HP.n_batch:min((i+1)*HP.n_batch, n)]
            
            tmp_y_task_feed.append(tmp_y_task)
            
            # get the total true label
            y_total_task_label.extend(np.argmax(tmp_y_task, axis=1).tolist())  # order task1 task2 _batch1  task1 task2 _batch2....
            # get the seperate true label for each task
            y_seperate_task_label[y_i].extend(np.argmax(tmp_y_task, axis=1).tolist()) # for each task: order : num_batch....

        feed_dict = load_x_data_for_cnn(tmp_patient_name,
                                        1.0,
                                        input_x,
                                        sent_length,
                                        category_index,
                                        dropout_keep_prob)

        for (M, input_y) in enumerate(input_ys):
            feed_dict[input_y] = tmp_y_task_feed[M]
                
        pre, pre_sent_tp, grad_sent_tp, len_sent_tp = sess.run([scores_soft_max_list, scores_sentence_soft_max_list, gradients_sentence_list, sent_length], feed_dict=feed_dict)  # [3,n_batch,2]
        
        
        for p in pre_sent_tp:
            for k in p:
                pre_sent.append(k)
        for g in grad_sent_tp:
            for k in g:
                grad_sent.append(k)
#         pre_sent.extend(list(pre_sent_tp))
        len_sent.extend(len_sent_tp)
    
        # slice the 3D array to get each on the first dimensional
        # get the seperate predictions for each task
        for m in range(HP.multi_size):
            pre_slice = pre[m, :]
            pre_pos = pre_slice[:, 1]
            seperate_pre[m].extend(pre_pos.tolist())

        # get the total predictions for all
        pre = pre.reshape(-1, HP.num_classes)  # [3*n_batch,2]  in one batch: task1+task2+task3
        pre = pre[:, 1]  # get probability of positive class
        predictions.extend(pre.tolist())   # task1,2,3_batch1 + task1,2,3_batch2+ task1,2,3_batch3....
        
#         pre_sent = pre_sent.reshape(-1, HP.num_classes)  
#         pre_sent = pre_sent[:, 1]  
#         predictions_sent.extend(pre_sent.tolist())   
        
#         print(y_total_task_label)
#         print(predictions)
        
    try:
        auc = roc_auc_score(np.asarray(y_total_task_label), np.asarray(predictions))
    except:
        auc = 0.5

    if test_output_flag:
        test_output_file = open(HP.test_output, "w")
        for m in range(HP.multi_size):
            try:
                auc_per_task[m] = roc_auc_score(np.asarray(y_seperate_task_label[m]), np.asarray(seperate_pre[m]))
            except:
                auc_per_task[m] = 0.5
            test_output_file.write("----task----"+str(m)+"\n")
            test_output_file.write("patient_id,true,prediction\n")
            for p in range(len(patient_name)):
                test_output_file.write(str(patient_name[p]) + "," + str(y_seperate_task_label[m][p]) + "," + str(round(seperate_pre[m][p],4)) + "\n")

    return auc, auc_per_task, pre_sent, grad_sent, predictions, y_total_task_label, len_sent
