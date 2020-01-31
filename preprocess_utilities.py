""" the functions used for extracting and preprocessing clinical notes from MIMIC-III """

import pandas as pd
import numpy as np
import re
from gensim.models import Word2Vec
from nltk.corpus import stopwords
import HP


def preprocess(df_note, admission, patient):
    """the utility to preprocess the mimic note
    Args:
        df_note: the note dataframe
        admission: admission csv file
        patient: patient csv file
        
        return: one dataframe with note as features and label of mortality
                patient subject id for later visualization
    """
    # prepare datetime
    df_note['CHARTDATE'] = pd.to_datetime(df_note['CHARTDATE'])
    df_note['CHARTTIME'] = pd.to_datetime(df_note['CHARTTIME'])
    admission['ADMITTIME'] = pd.to_datetime(admission['ADMITTIME'])
    admission['ADMITDATE'] = admission['ADMITTIME'].values.astype('<M8[D]')
    admission['DISCHTIME'] = pd.to_datetime(admission['DISCHTIME'])
    admission['DISCHDATE'] = admission['DISCHTIME'].values.astype('<M8[D]')
    admission['DEATHTIME'] = pd.to_datetime(admission['DEATHTIME'])
    patient['DOD'] = pd.to_datetime(patient['DOD'])
    patient['DOB'] = pd.to_datetime(patient['DOB'])

    # remove discharge summary
    df_note = df_note[df_note.CATEGORY != 'Discharge summary']
    # remove the note with error tag
    df_note = df_note[df_note['ISERROR'] != 1]
    # drop the column that are not used in the future
    df_note = df_note.drop(['ROW_ID', 'STORETIME', 'DESCRIPTION', 'CGID', 'ISERROR', 'HADM_ID'], axis=1)

    # caculate the admission time for each patient
    admission['admit_times'] = admission.groupby(['SUBJECT_ID'])['SUBJECT_ID'].transform('size')
    # remove the patient with multiple admissions
    admission = admission[admission['admit_times'] < 2]
    # drop the column that are not used in the future
    admission = admission.drop(['ROW_ID', 'HADM_ID', 'ADMISSION_TYPE', 'ADMISSION_LOCATION', 'DISCHARGE_LOCATION', 'INSURANCE',
                                'LANGUAGE', 'RELIGION', 'MARITAL_STATUS', 'ETHNICITY', 'EDREGTIME', 'EDOUTTIME', 'DIAGNOSIS',
                                'HAS_CHARTEVENTS_DATA', 'admit_times'], axis=1)

    # drop the column that are not used in the future
    patient = patient.drop(['ROW_ID', 'GENDER', 'DOD_SSN', 'EXPIRE_FLAG', 'DOD_HOSP'], axis=1)


    # merge patient and admission csv to constraint patient
    patient_filter = pd.merge(patient, admission, on='SUBJECT_ID', how='inner')
    # remove patient of age below 18 when admission
    patient_filter = patient_filter[(patient_filter['ADMITTIME'] - patient_filter['DOB']) > pd.Timedelta(pd.offsets.Day(18*365))]
    # remove patient of duration of stay < 1 day
    patient_filter = patient_filter[(patient_filter['DISCHTIME'] - patient_filter['ADMITTIME']) > pd.Timedelta(pd.offsets.Day(1))]

    # merge patient and note; might generate a lot of replicate records
    patient_note = pd.merge(patient_filter, df_note, on='SUBJECT_ID', how='inner')
    # keep only notes where recorded time is whithin 1 day of admission
    patient_note = patient_note[((patient_note['CHARTTIME'] > patient_note['ADMITTIME'])
                                & (patient_note['CHARTTIME'] - patient_note['ADMITTIME'] < pd.Timedelta(pd.offsets.Day(1)))) |
                                ((patient_note['CHARTDATE'] == patient_note['ADMITDATE']) & patient_note['CHARTTIME'].isnull())]


    # combine two columns to one column with tuple
    patient_note['category_text'] = list(zip(patient_note['CATEGORY'], patient_note['TEXT']))

    # patient labels
    patient_label = patient_note[['SUBJECT_ID', 'ADMITTIME','DOD', 'DISCHTIME', 'DEATHTIME', 'HOSPITAL_EXPIRE_FLAG']]
    patient_label = patient_label.drop_duplicates()

    # combine several duplicate records along the column into one entry
    note = patient_note[['SUBJECT_ID', 'category_text']]
    aggregated = note.groupby('SUBJECT_ID')['category_text'].apply(tuple)
    aggregated.name = 'full_text'
    note = note.join(aggregated, on='SUBJECT_ID')
    note = note.drop(['category_text'], axis=1)
    note = note.drop_duplicates()

    patient_note_label = pd.merge(patient_label, note, on='SUBJECT_ID', how='inner')


    # calculate the dead date for patient
    patient_note_label['dead_date'] = patient_note_label['DOD'] - patient_note_label['DISCHTIME']
    # transfer time to date
    patient_note_label['dead_date'] = patient_note_label['dead_date'].dt.days
    patient_note_label['dead_after_disch_date'] = [-1.0 if f==1 else d for f,d in zip(patient_note_label['HOSPITAL_EXPIRE_FLAG'], patient_note_label['dead_date'])]

    # return the useful dataframe
    patient_note_with_label = patient_note_label[['full_text', 'dead_after_disch_date']]
    # also return the patient index
    patient_subjectid2index = patient_note_label['SUBJECT_ID']

    return patient_note_with_label, patient_subjectid2index



def train_embedding(patient_note_label):
    sent_ls = []
    for index, row in patient_note_label.iterrows():
        for x in row['full_text']:
            doc = x[1]
            sentences = split_doc(doc)
            for sent in sentences:
                cleaned_tokens = tokenize(sent, mimic3_embedding=None, check_in_embedding=False)
                if len(cleaned_tokens) > 0:
                    sent_ls.append(cleaned_tokens)
    model = Word2Vec(sent_ls, size=100, window=5, max_final_vocab =300000, workers=4)
    model.save(HP.embedding_file)


def load_embedding():
    model = Word2Vec.load(HP.embedding_file)
    embedding_map = {}
    for word in model.wv.vocab:
        vector = model.wv[word]
        coef = np.asarray(vector, dtype='float32')
        embedding_map[word] = coef
    return embedding_map



def split_doc(d):
    """Split sentences in a document and saved the sentences to a list.
    
    Args:
        d: a document
        final_d: a list of sentences
    """
    
    d = d.strip().split(".") # split document by "." to sentences
    final_d = []
    for s in d:
        if s != "":  # ignore if the sentence is empty
            final_d.append(s.strip())
    return final_d  # Now the sentences are splitted from documents and saved to a list


def tokenize(sent, mimic3_embedding=None, check_in_embedding=True):
    """Tokenize the sentences according to the existing word from embedding. 
    
    Args:
        sent: input a sentence
        mimic3_embedding: find the existing word in embedding files
        cleaned_tokens: the tokens are cleaned and mapped to the mimic embedding 
    """
    
    tokenizer = re.compile('\w+|\*\*|[^\s\w]')
    tokens = tokenizer.findall(sent.lower())
    cleaned_tokens = []
    for tok in tokens:
        tok = _clean_token(tok)
        if tok not in stopwords.words('english') and len(tok)>1 :         
            if check_in_embedding==True:
                if tok in mimic3_embedding:
                    cleaned_tokens.append(tok)
            else:
                cleaned_tokens.append(tok)
    return cleaned_tokens


def _clean_token(s):
    """If the token is digit, then round the actual value into the nearest 10 times value.
    Args:
        s: original digit, 65 -> 60
        """
    if len(s) > 1:
        if s.isdigit():
            l = len(s)
            s = str(int(s)//(10**(l-1)) * 10**(l-1))
    return s.lower()
