"""parameter used in the project"""

# mimic data
mimic_note_events = 'mimic_csv/NOTEEVENTS.csv'
mimic_admissions = 'mimic_csv/ADMISSIONS.csv'
mimic_patients = 'mimic_csv/PATIENTS.csv'

# processed data
subject_index = "processed/subject.csv" # patient index
result_csv = "processed/result_csv_dead_los.csv" # death label
data_directory = "processed/entire_file/" # notes - one patient per file

# models
embedding_file = 'model/word2vec.model' # word2vec
model_path = "model/model.ckpt" # cnn

# note category
category = ['pad', 'Respiratory', 'ECG','Radiology','Nursing/other','Rehab Services','Nutrition','Pharmacy','Social Work',
            'Case Management','Physician','General','Nursing','Echo','Consult']
category_id = {cate: idx for idx, cate in enumerate(category)}



# mortality task
tasks_dead_date = [0]


# CNN model hyperparameters
restore = False

n_batch = 64
multi_size = 1 #len(tasks_dead_date)
num_classes = 2

max_document_length = 1000
max_sentence_length = 25
embedding_size = 100

n_category = len(category)
dim_category = 10

filter_sizes = [3, 4, 5]
num_filters = 50

lambda_regularizer_strength = 5

document_filter_size = 3
document_num_filters = 50

learning_rate = 0.001

drop_out_train = 0.8

early_stop_times = 5


# load data
read_data_thread_num = 8

n_max_sentence_num = 1000 # truncated to 1000 sentences a document
n_max_word_num = 25 # truncated to 25 words a sentence


# test
test_output = 'results/test_output.txt'