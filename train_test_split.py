from utils2 import resource,wrong_csv,build_headers_dict,extract_values
import shutil
from random import shuffle
from metadata import *
import matplotlib.pyplot as plt

data_collection=resource(data_folder)
#headers_counter=build_headers_dict(data_collection)
data_collection = wrong_csv(data_collection)
valid_data=[]
wrong_data=[]
for item in data_collection.data_files:
    if item.status:
        valid_data.append(item)
    else:
        wrong_data.append(item)

data_collection.data_files=valid_data
data_collection.num_data=len(valid_data)

for item in wrong_data:
    shutil.move(item.path,data_folder_debug)

num_rows=[data_collection.data_files[i].num_rows for i in range(len(data_collection.data_files))]
stats=[min(num_rows),max(num_rows),np.mean(num_rows),np.std(num_rows),np.median(num_rows)]

indices=[i for i in range(len(valid_data))]
shuffle(indices)
size_train=int(.8*len(indices))
train_indices=indices[:size_train]
test_indices=indices[size_train:]

train_ids=[data_collection.data_files[i].df_id for i in train_indices]
test_ids=[data_collection.data_files[i].df_id for i in test_indices]
np.save(train_files,train_ids)
np.save(test_files, test_ids)

headers_counter=build_headers_dict(data_collection)

sortd_vocab=[[l,k] for k,l in sorted([(j,i) for i,j in headers_counter.items()], reverse=True)]
frequencies=[el[1] for el in sortd_vocab]
vocabulary=[el[0] for el in sortd_vocab]
nb_columns=sum(frequencies)
column_ratio=np.array(frequencies)/nb_columns


train_ids=np.load(train_files)
test_ids = np.load(test_files)

train=True
data_collection=resource(data_folder,train_ids)
extract_values(data_collection,train)

train=False
data_collection=resource(data_folder,test_ids)
extract_values(data_collection,train)

plt.plot(frequencies,'*')
plt.xlabel('header')
plt.ylabel('frequency')
plt.show()

