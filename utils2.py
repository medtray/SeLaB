import numpy as np
import string
import os
import re
from collections import Counter
import torch
from scipy import signal
from glob import glob
import json
import csv
import codecs
import numpy as np
import math
from collections import Counter
import os
import pickle
import random
import pandas as pd
from metadata import *
from multiprocessing import Pool as ThreadPool
import numpy
from nltk.tokenize import ToktokTokenizer
from sklearn.metrics import precision_recall_fscore_support

def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

def camel_case_split(identifier):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0) for m in matches]


def preprocess(input,wpt,stop_words):

    res=re.sub("([\(\[]).*?([\)\]])", "\g<1>\g<2>", input)

    to_remove=['(',')','{','}','[',']']

    ss=[s for s in res if s not in to_remove]
    ss=''.join(ss)


    w = re.sub(r'[^a-zA-Z0-9@$%\s]', ' ', ss, re.I | re.A)
    w = w.strip()
    # tokenize document
    tokens = wpt.tokenize(w)
    # filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]

    camel_tokens=[]

    for w in filtered_tokens:
        inter = camel_case_split(w)
        camel_tokens += inter

    tokens=camel_tokens

    # convert to lower case
    tokens = ' '.join(tokens)
    tokens = tokens.lower()


    return tokens


def topn_accuracy_from_probabilities(probs, y_test, topn=10,freq_filter=0,only_seen=False,seen_labels=None):

    if freq_filter != 0:
        label_ct = Counter(y_test)
        new_X = []
        new_y = []
        for idx in range(len(y_test)):
            if label_ct[y_test[idx]] <= freq_filter:
                continue
            new_X.append(probs[idx])
            new_y.append(y_test[idx])
        probs = np.array(new_X)
        y_test = new_y
    if only_seen:

        new_X = []
        new_y = []
        for idx in range(len(y_test)):
            if y_test[idx] in seen_labels:
                new_X.append(probs[idx])
                new_y.append(y_test[idx])
        probs = np.array(new_X)
        y_test = new_y

    correct_ct = 0
    for i in range(len(probs)):
        prob = probs[i]
        topn_rs = np.argpartition(prob, -topn)[-topn:]
        if y_test[i] in topn_rs:
            correct_ct += 1
    return correct_ct * 1.0 / probs.shape[0]


def load_checkpoint(model, optimizer, losslogger, filename,device):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename,map_location=device)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        losslogger = checkpoint['losslogger']
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
        del checkpoint
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch, losslogger


def load_checkpoint_for_eval(model, filename):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
        model=model.eval()

    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model



class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


class dataFile(object):
    '''
    TODO:
    a lot of cases to handle the errors
    1. no such file file (due to download errors)
    2. have the file with data.csv, but is not csv file -> col errors
    '''

    def __init__(self, d_path):
        self.path = d_path
        self.df_id = d_path.split('\\')[-1]
        self.header = None
        self.status = False
        self.reason = None
        self.col_num = 0
        self.data_size = 0
        self.num_rows=0


    def show_few_data(self, num_line=3):
        try:
            f = open(self.path, 'r')
            for i in range(num_line):
                print(f.readline())
        except:
            print("reading file error.")

    def get_raw_content(self):
        try:
            f = open(self.path, 'r')
            content = f.read()
        except:
            content = ""
            print("reading file error.")
        return content

    def read_data_cols(self):
        try:
            self.data_size = os.path.getsize(self.path)
            df = pd.read_csv(self.path, delimiter=',', quotechar='"', dtype=str, na_filter=False)
            self.header = list(df)
            self.status = True
            self.col_num = len(self.header)

            # with codecs.open(self.path, encoding="utf-8", errors="replace") as ifile:
            #     reader = csv.reader(ifile, delimiter=',', quotechar='"')
            #     for header in reader:
            #         self.header = header
            #         self.status = True
            #         self.col_num = len(self.header)
            #         break
            return True
        except:
            self.reason = "open error"
            return False

    def row_and_cell_count(self, cell_counter, normalize=False):
        self.row_num = 0
        try:
            with codecs.open(self.path , encoding="utf-8", errors="replace") as ifile:
                reader = csv.reader(ifile, delimiter=',', quotechar='"')
                reader.next()  # jump the header
                for row in reader:
                    self.row_num += 1
                    cell_counter.update(row)
            return cell_counter
        except:
            self.reason = "open error"
            return cell_counter



class resource(object):

    def __init__(self,
                 r_path,ids=None):
        self.path = r_path
        self.data_num = 0
        self.data_type = []
        self.ids = ids
        self.read_datasets()


    @staticmethod
    def r_notes_prep():
        pass

    def read_datasets(self):
        if self.ids is not None:
            self.resources=self.ids
        else:
            self.resources = os.listdir(self.path)
        self.num_data = len(self.resources)
        self.data_files = []
        # read all dataset columns
        for rs in self.resources:
            rs=os.path.join(self.path,rs)
            df = dataFile(rs)
            df.read_data_cols()
            self.data_files.append(df)


def resource_preprocessing(all_resources):
    '''
    1. wrong csv files
    2. normalize special characters
    '''
    resources = wrong_csv(all_resources)
    resources = normalize_headers(resources)
    return resources


def wrong_csv(resources):
    '''
    1. if line length <=1, false
    2. if line length >1, len(line1.splits) != len(line2.splits), false
    3. no headers ?
    '''

    for df_idx in range(len(resources.data_files)):
        #f = open(resources.data_files[df_idx].path, 'r')
        try:
            df = pd.read_csv(resources.data_files[df_idx].path, delimiter=',', quotechar='"', dtype=str, na_filter=False)
            resources.data_files[df_idx].num_rows=df.shape[0]
        except:
            resources.data_files[df_idx].status = False

        # try:
        #     line1 = f.readline()
        #     line2 = f.readline()
        #     line3 = f.readline()
        # except:
        #     resources.data_files[df_idx].status = False
        #     resources.data_files[df_idx].reason = 'few lines'
        #     continue
        # if len(line1.strip().split(',')) <= 1:
        #     resources.data_files[df_idx].status = False
        #     resources.data_files[df_idx].reason = 'few columns'
        #     continue
        # if len(line2.strip().split(',')) != len(line3.strip().split(',')):
        #     resources.data_files[df_idx].status = False
        #     resources.data_files[df_idx].reason = "not aligned"
        #     continue
        # if line2 == '':
        #     resources.data_files[df_idx].status = False
        #     resources.data_files[df_idx].reason = "no data"
        #     continue
        #
        # false_header = False
        # for each in line1.split(','):
        #     if len(each) == 0:
        #         false_header = True
        #         break
        # if false_header:
        #     resources.data_files[df_idx].status = False
        #     resources.data_files[df_idx].reason = "false header"
        #     continue

    return resources


def normalize_headers(resources):
    '''

    '''
    for idx in range(len(resources)):
        for df_idx in range(len(resources[idx].data_files)):
            if not resources[idx].data_files[df_idx].status:
                continue
            for h_idx in range(len(resources[idx].data_files[df_idx].header)):
                resources[idx].data_files[df_idx].header[h_idx] = resources[idx].data_files[df_idx].header[
                    h_idx].lower()
                chrs = list(resources[idx].data_files[df_idx].header[h_idx])
                for ch_idx in range(len(chrs)):
                    if (not chrs[ch_idx].isalpha()) and (not chrs[ch_idx].isdigit()):
                        chrs[ch_idx] = '_'
                resources[idx].data_files[df_idx].header[h_idx] = ''.join(chrs).lower()
    return resources

def normalize_header_name(resources,new_folder,wpt,stop_words):

    for df_idx in range(len(resources.data_files)):
        df = pd.read_csv(resources.data_files[df_idx].path, delimiter=',', quotechar='"', dtype=str, na_filter=False)

        columns=list(df)
        #columns=['speed (in km/h)','weight [kg]','team\'s loosing']
        new_columns=[]
        for colum in columns:
            new_columns+=[preprocess(colum, wpt, stop_words)]

        df.columns = new_columns

        head,tail=os.path.split(resources.data_files[df_idx].path)
        new_path=os.path.join(new_folder,tail)
        df.to_csv(new_path,encoding='utf-8-sig',index=False)

    print('done')

def filter_empty_header_name(resources,new_folder):

    for df_idx in range(len(resources.data_files)):
        df = pd.read_csv(resources.data_files[df_idx].path, delimiter=',', quotechar='"', dtype=str, na_filter=True)

        columns=list(df)

        columns_to_drop=[column for column in columns if 'unnamed' in column.lower() or len(column)==0]

        for col in columns:
            if len(col)==0:
                print('here')

        if len(columns_to_drop)>0:
            df.drop(columns_to_drop, axis=1, inplace=True)
            head,tail=os.path.split(resources.data_files[df_idx].path)
            new_path=os.path.join(new_folder,tail)
            df.to_csv(new_path,index=False)

    print('done')

def filter_columns(resources,new_folder,column_dict,threshold):

    for df_idx in range(len(resources.data_files)):
        df = pd.read_csv(resources.data_files[df_idx].path, delimiter=',', quotechar='"', dtype=str, na_filter=False)

        columns=list(df)

        columns_to_drop=[]
        for column in columns:
            if column_dict[column.lower()]<threshold:
                columns_to_drop.append(column)

        df.drop(columns_to_drop, axis=1, inplace=True)

        new_att=list(df)

        if len(new_att)>=3:
            head,tail=os.path.split(resources.data_files[df_idx].path)
            new_path=os.path.join(new_folder,tail)

            df.to_csv(new_path, encoding='utf-8-sig', index=False)


    print('done')


def pre_cell(cell, case_folding=True):
    '''
    preprocess cell before extracting features
    1. case folding
    2. replace numbers ?
    '''
    cell = cell.strip()
    if case_folding:
        cell = cell.lower()
    return cell


def get_col(col):

    context=col[1].copy()
    col=col[0]
    if len(col) == 0:
        return False


    feature_dict = dict()

    feature_dict['content']=col
    feature_dict['context'] = context



    return feature_dict


def build_headers_dict(all_resources):
    word2int = {}
    headers_counter=Counter()
    ll = -1
    for each_data in all_resources.data_files:
        # d_path = each_data.path
        # df = pd.read_csv(d_path, delimiter=',', quotechar='"', dtype=str, na_filter=True)
        # cols = df.columns
        # with codecs.open(d_path, encoding="utf-8", errors="replace") as ifile:
        #     reader = csv.reader(ifile, delimiter=',', quotechar='"')
        #     for headers in reader:
        #         cols=headers
        #         break

        cols = each_data.header

        for ii in range(len(cols)):
            header_name = cols[ii].lower()
            if header_name not in word2int:
                ll += 1
                word2int[header_name] = ll
        headers_counter.update([el.lower() for el in cols])

    np.save(word2int_path, word2int)
    return headers_counter


def extract_values(all_resources,train=True):
    # extracting features:
    if train:
        fdict_path=train_features_path
    else:
        fdict_path=test_features_path

    f = open(fdict_path, 'w')
    pool = ThreadPool(8)

    total = len(all_resources.data_files)
    count = 0
    toktok = ToktokTokenizer()

    for each_data in all_resources.data_files:
        #try:

            tid = each_data.df_id
            print(tid)
            d_path = each_data.path
            df = pd.read_csv(d_path, delimiter=',', quotechar='"', dtype=str, na_filter=True)
            cols = df.columns
            contents = [[df[each_col].dropna().tolist()] for each_col in cols]

            for ii in range(len(cols)):
                inter_cols = list(cols.copy())
                del inter_cols[ii]
                contents[ii].append(inter_cols)

            print("extract content finished")

            cols_features = pool.map(get_col, contents)

            all_col_features = list(zip(cols, cols_features))
            all_col_features = list(filter(lambda x: x[1], all_col_features))

            # tid=os.path.join(output_folder,'1')
            f.write(json.dumps({tid: all_col_features}, cls=MyEncoder) + '\n')
            count += 1
            print("finish {0} out of {1}".format(count, total))
        #except Exception as e:
        #    print(e)

    f.close()

def MRR(ranked_indices,y):

    rs = [(ranked_indices[i, :] == y[i]).nonzero()[0] for i in range(ranked_indices.shape[0])]
    mrr = np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])

    return mrr


def topn_accuracy(clf, X_test, y_test, topn=10):
    preds = []
    iters = int(X_test.shape[0] / 1000)
    for i in range(iters + 1):
        preds.append(clf.predict_proba(X_test[i * 1000:(i * 1000 + 1000)]))
    probs = np.concatenate(preds)
    correct_ct = 0
    for i in range(len(probs)):
        prob = probs[i]
        #topn_rs = np.argpartition(prob, -topn)[-topn:]
        topn_rs=np.argsort(prob)[::-1][:topn]
        if y_test[i] in clf.classes_[topn_rs]:
            correct_ct += 1
    return correct_ct * 1.0 / X_test.shape[0]


def precision_recall_f1(preds,y_test,freq_filter=0,only_seen=False,seen_labels=None):

    if freq_filter != 0:
        label_ct = Counter(y_test)
        new_X = []
        new_y = []
        for idx in range(len(y_test)):
            if label_ct[y_test[idx]] <= freq_filter:
                continue
            new_X.append(preds[idx])
            new_y.append(y_test[idx])
        preds = np.array(new_X)
        y_test = new_y
    if only_seen:

        new_X = []
        new_y = []
        for idx in range(len(y_test)):
            if y_test[idx] in seen_labels:
                new_X.append(preds[idx])
                new_y.append(y_test[idx])
        preds = np.array(new_X)
        y_test = new_y

    pred_label=np.argmax(preds,axis=1)

    macro = precision_recall_fscore_support(y_test, pred_label, average='macro')
    micro = precision_recall_fscore_support(y_test, pred_label, average='micro')

    return macro,micro