from utils2 import *
from transformers import *
from table_reader import *
from torch.utils.data import DataLoader
from metadata import *

import random

import argparse

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
}
ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, XLMConfig)), ())

parser = argparse.ArgumentParser()

parser.add_argument("--model_type", default='bert', type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
parser.add_argument("--model_name_or_path", default='bert-base-uncased', type=str,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
parser.add_argument("--output_dir", default=None, type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

parser.add_argument("--do_lower_case",default=True, action='store_true',
                        help="Set this flag if you are using an uncased model.")

parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
parser.add_argument("--tokenizer_name", default=None, type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
parser.add_argument("--task_name", default='classification', type=str,
                        help="The name of the task")
parser.add_argument("--ignore_logits_layer", action='store_true',
                        help="whether to skip initialization of logits layers.")
parser.add_argument("--ignore_sequence_summary_layer", action='store_true',
                        help="whether to skip initialization of sequence summary layers.")

#### learning rate difference between original BertAdam and now paramters.
parser.add_argument("--learning_rate", default=5e-5, type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float,
                    help="Weight decay if we apply some.")
parser.add_argument("--lr_layer_decay", default=1.0, type=float,
                    help="layer learning rate decay.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                    help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float,
                    help="Max gradient norm.")
parser.add_argument("--num_train_epochs", default=30, type=int,
                    help="Total number of training epochs to perform.")
parser.add_argument("--max_steps", default=-1, type=int,
                    help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
parser.add_argument("--warmup_steps", default=0, type=int,
                    help="Linear warmup over warmup_steps.")
parser.add_argument("--warmup_proportion", default=0.1, type=float,
                    help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
parser.add_argument('--device', type=int, default=0)
args = parser.parse_args()

device=args.device
print(torch.cuda.current_device())
torch.cuda.set_device(device)
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(device))
print(torch.cuda.is_available())
args.device='cuda:'+str(device)
#args.device='cpu'

input_word2int=np.load(word2int_path,allow_pickle=True)
input_word2int=input_word2int[()]
input_vocab_size = len(input_word2int)
sortd_vocab=[[l,k] for k,l in sorted([(j,i) for i,j in input_word2int.items()], reverse=False)]

int2word=[el[0] for el in sortd_vocab]

topk=input_vocab_size

num_labels=input_vocab_size
args.model_type = args.model_type.lower()
config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
config.output_hidden_states=True

bert_model_hp = model_class.from_pretrained(args.model_name_or_path,config=config).to(args.device)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

def extract_n_layer(n, max_n_layer=-1):
    n = n.split('.')
    try:
        idx = n.index("layer")
        n_layer = int(n[idx + 1]) + 1
    except:
        if any(nd in n for nd in ["embeddings", "word_embedding", "mask_emb"]):
            n_layer = 0
        else:
            n_layer = max_n_layer
    return n_layer

max_n_layer = max([extract_n_layer(n) for n, p in bert_model_hp.named_parameters()]) + 1
# Prepare optimizer and schedule (linear warmup and decay)
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = []  # group params by layers and weight_decay params.
for n_layer in range(max_n_layer + 1):
    #### n_layer and decay
    optimizer_grouped_parameters.append({
        'params': [p for n, p in bert_model_hp.named_parameters() if (
                extract_n_layer(n, max_n_layer=max_n_layer) == n_layer and not any(nd in n for nd in no_decay))],
        'weight_decay': args.weight_decay,
        'lr_decay': args.lr_layer_decay ** (max_n_layer - n_layer)
    })
    #### n_layer and no_decay
    optimizer_grouped_parameters.append({
        'params': [p for n, p in bert_model_hp.named_parameters() if (
                extract_n_layer(n, max_n_layer=max_n_layer) == n_layer and any(nd in n for nd in no_decay))],
        'weight_decay': 0.0,
        'lr_decay': args.lr_layer_decay ** (max_n_layer - n_layer)
    })

optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

loss=np.inf

bert_model_hp, optimizer, start_epoch, loss=load_checkpoint(bert_model_hp, optimizer, loss, model_file,args.device)


train_ids=np.load(train_files)
test_ids = np.load(test_files)

nb_train_tables=len(np.load(train_files))
nb_test_tables=len(np.load(test_files))

all_labels_in_train=[]

max_values_tokens=100
max_headers_tokens=50
max_tokens=max_values_tokens+max_headers_tokens+3
batch_size=16

seen_labels=Counter()
loaded=False

training=True

threshold_context=-100
prob_use_context=1

nb_warmup_iter=0

debug_print=20
if training:
    train=True

    fdict_path = train_features_path
    tab_index=0

    nb_epochs = args.num_train_epochs
    for epoch in range(start_epoch, start_epoch + nb_epochs):
        f = open(fdict_path, 'r')


        for table in f:
            try:

                only_values = True
                contexts=None
                train_table = DataReaderBert(table,tokenizer,input_word2int,max_values_tokens,max_headers_tokens,max_tokens,only_values,contexts,loaded)
                #print(len(train_table))
                train_iter = DataLoader(train_table, batch_size = batch_size, shuffle = False)
                all_outputs=[]
                all_labels=[]
                tab_index+=1

                all_probs=[]

                for feat,mask,labels in train_iter:
                    feat,mask,labels=feat.to(args.device),mask.to(args.device),labels.to(args.device)

                    seen_labels.update(labels.squeeze(1).tolist())

                    inputs = {'input_ids': feat,
                              'attention_mask': mask,
                              'token_type_ids': None,
                              'labels': labels}

                    preds = bert_model_hp(**inputs)
                    #all_outputs += preds[1].tolist()
                    #all_outputs+=torch.argmax(preds[1], dim=1).tolist()
                    all_labels+=labels.squeeze(1).tolist()

                    probs,indices=torch.max(preds[1],dim=1)

                    all_outputs+=indices.tolist()
                    all_probs+=probs.tolist()

                    all_labels_in_train+=labels.squeeze(1).tolist()

                all_labels_names = [int2word[l] for l in all_labels]
                predicted_labels = [int2word[index] for index in all_outputs]


                if tab_index>nb_warmup_iter:

                    if random.random()<prob_use_context:


                        contexts=[list(set([predicted_labels[i] for i in range(len(predicted_labels)) if i!=j and predicted_labels[i]!=all_labels_names[j] and all_probs[i]>threshold_context]))
                                  for j in range(len(predicted_labels))]

                        #table_cols = list(json.loads(table).values())[0]

                        only_values=False
                        train_table = DataReaderBert(table, tokenizer, input_word2int, max_values_tokens, max_headers_tokens,
                                                     max_tokens, only_values, contexts,loaded)
                        #print(len(train_table))
                        train_iter = DataLoader(train_table, batch_size=batch_size, shuffle=False)

                all_outputs=[]

                for feat,mask,labels in train_iter:
                    feat,mask,labels=feat.to(args.device),mask.to(args.device),labels.to(args.device)

                    inputs = {'input_ids': feat,
                              'attention_mask': mask,
                              'token_type_ids': None,
                              'labels': labels}

                    preds = bert_model_hp(**inputs)

                    loss = preds[0]
                    print('loss={}'.format(loss))
                    #embeds = preds[2][-1][:, 0, :]
                    all_outputs+=torch.argmax(preds[1], dim=1).tolist()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                final_predicted_labels = [int2word[index] for index in all_outputs]

                if tab_index%debug_print==0:
                    print(predicted_labels)
                    print(final_predicted_labels)
                    print(all_labels_names)

            except:
                pass


        print('done with epoch {}'.format(epoch + 1))
        state = {'epoch': epoch + 1, 'state_dict': bert_model_hp.state_dict(),
                 'optimizer': optimizer.state_dict(), 'losslogger': loss, }
        torch.save(state, model_file)

np.save(seen_labels_file,seen_labels)

seen_labels_freq=Counter(all_labels_in_train)

np.save(freq_seen_labels_file,seen_labels_freq)


