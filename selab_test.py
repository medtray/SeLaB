
from transformers import *
from table_reader import *
from torch.utils.data import DataLoader

from metadata import *

from utils2 import *
import argparse

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
}
ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, XLMConfig)),
                 ())

parser = argparse.ArgumentParser()

parser.add_argument("--model_type", default='bert', type=str,
                    help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
parser.add_argument("--model_name_or_path", default='bert-base-uncased', type=str,
                    help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
parser.add_argument("--output_dir", default=None, type=str,
                    help="The output directory where the model predictions and checkpoints will be written.")

parser.add_argument("--do_lower_case", default=True, action='store_true',
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
parser.add_argument("--num_train_epochs", default=3.0, type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--max_steps", default=-1, type=int,
                    help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
parser.add_argument("--warmup_steps", default=0, type=int,
                    help="Linear warmup over warmup_steps.")
parser.add_argument("--warmup_proportion", default=0.1, type=float,
                    help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
parser.add_argument('--device', type=int, default=0)
args = parser.parse_args()

device=0
print(torch.cuda.current_device())
torch.cuda.set_device(device)
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(device))
print(torch.cuda.is_available())
args.device='cuda:'+str(device)
#args.device = 'cpu'

input_word2int = np.load(word2int_path, allow_pickle=True)
input_word2int = input_word2int[()]
input_vocab_size = len(input_word2int)
sortd_vocab = [[l, k] for k, l in sorted([(j, i) for i, j in input_word2int.items()], reverse=False)]

int2word = [el[0] for el in sortd_vocab]

num_labels = input_vocab_size
args.model_type = args.model_type.lower()
config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                      num_labels=num_labels, finetuning_task=args.task_name)
config.output_hidden_states = True

bert_model_hp = model_class.from_pretrained(args.model_name_or_path, config=config).to(args.device)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

bert_model_hp = load_checkpoint_for_eval(bert_model_hp, model_file)

max_values_tokens = 100
max_headers_tokens = 50
max_tokens = max_values_tokens + max_headers_tokens + 3
batch_size = 5
topk=3

fdict_path = test_features_path
f = open(fdict_path, 'r')

all_labels=[]
final_predicted_prob=[]


seen_labels=np.load(seen_labels_file,allow_pickle=True)
seen_labels=seen_labels[()]

freq_seen_labels=np.load(freq_seen_labels_file,allow_pickle=True)
freq_seen_labels=freq_seen_labels[()]

use_context_for_prediction=True
unique_predicted_labels_constrain=True

for table in f:

    print(list(json.loads(table).keys())[0])

    contexts = None
    only_values=True
    loaded=False
    test_table = DataReaderBert(table, tokenizer, input_word2int, max_values_tokens, max_headers_tokens, max_tokens,
                                 only_values, contexts,loaded)
    #print(len(test_table))

    #if len(test_table)>30:
    #    continue

    train_iter = DataLoader(test_table, batch_size=batch_size, shuffle=False)
    all_outputs = []
    all_probs=[]
    initial_table_pred_dist=[]

    index_of_header=[i for i in range(len(test_table))]

    to_del_from_contexts=[]

    table_freq_seen_label=[]



    table_labels=[]


    for feat, mask, labels in train_iter:
        feat, mask, labels = feat.to(args.device), mask.to(args.device), labels.to(args.device)

        inputs = {'input_ids': feat,
                  'attention_mask': mask,
                  'token_type_ids': None,
                  'labels': None}

        preds = bert_model_hp(**inputs)
        max_values,indices=torch.max(preds[0], dim=1)
        all_outputs += indices.tolist()
        all_labels += labels.squeeze(1).tolist()
        table_labels += labels.squeeze(1).tolist()
        all_probs += max_values.tolist()
        initial_table_pred_dist+=preds[0].tolist()

    predicted_labels = [int2word[index] for index in all_outputs]
    #predicted_labels = [-1 for _ in range(len(table_labels))]
    only_values_predicted_labels=predicted_labels.copy()
    groundtruth_labels = [int2word[index] for index in table_labels]

    table_freq_seen_label=[freq_seen_labels[el] if el in seen_labels else 0 for el in table_labels]

    final_probs=all_probs.copy()

    if use_context_for_prediction:



        table_predictions=[-1 for _ in range(len(predicted_labels))]

        table_pred_dist=[[None] for _ in range(len(predicted_labels))]
        table_cols = list(json.loads(table).values())[0]

        for j in range(len(test_table)):
            if j==0:
                max_index=np.argmax(np.array(all_probs))
                table_predictions[max_index]=int2word[all_outputs[max_index]]

                table_pred_dist[max_index]=initial_table_pred_dist[max_index]
                to_del_from_contexts.append(max_index)
                index_of_new_predicted_label=max_index
            else:


                predicted_labels[index_of_new_predicted_label]=table_predictions[index_of_new_predicted_label]

                contexts = [list(set([predicted_labels[i] for i in range(len(predicted_labels)) if
                                      i != k and predicted_labels[i]!=-1]))
                            for k in range(len(predicted_labels))]

                del table_cols[max_index]
                contexts=[cont for ind,cont in enumerate(contexts) if ind not in to_del_from_contexts]
                del index_of_header[max_index]

                loaded=True
                only_values=False
                test_table = DataReaderBert(table_cols, tokenizer, input_word2int, max_values_tokens, max_headers_tokens,
                                            max_tokens,
                                            only_values, contexts, loaded)


                #print(len(test_table))
                train_iter = DataLoader(test_table, batch_size=batch_size, shuffle=False)
                all_outputs = []
                all_probs = []
                initial_table_pred_dist = []

                for feat, mask, labels in train_iter:
                    feat, mask, labels = feat.to(args.device), mask.to(args.device), labels.to(args.device)

                    inputs = {'input_ids': feat,
                              'attention_mask': mask,
                              'token_type_ids': None,
                              'labels': None}

                    preds = bert_model_hp(**inputs)

                    if not unique_predicted_labels_constrain:
                        max_values, indices = torch.max(preds[0], dim=1)

                        # dd = torch.topk(preds[1], k=5, dim=1)
                        # lll = int2word[33]

                        all_outputs += indices.tolist()
                        all_probs += max_values.tolist()
                        initial_table_pred_dist += preds[0].tolist()

                    else:


                        max_values, indices = torch.topk(preds[0], k=topk, dim=1)


                        all_outputs += indices.tolist()
                        all_probs += max_values.tolist()
                        initial_table_pred_dist += preds[0].tolist()


                if not unique_predicted_labels_constrain:

                    max_index = np.argmax(np.array(all_probs))
                    index_of_new_predicted_label=index_of_header[max_index]
                    table_predictions[index_of_new_predicted_label] = int2word[all_outputs[max_index]]
                    table_pred_dist[index_of_new_predicted_label] = initial_table_pred_dist[max_index]

                    max_cinf = np.max(np.array(all_probs))

                    to_del_from_contexts.append(index_of_new_predicted_label)

                else:

                    all_probs=np.array(all_probs)
                    all_outputs=np.array(all_outputs)
                    max_index=np.argmax(all_probs[:,0])
                    new_label=int2word[all_outputs[max_index,0]]
                    sum_stop=-1*all_outputs.shape[0]*all_outputs.shape[1]
                    sum_outputs=np.sum(all_outputs)

                    first_best_max_index = max_index
                    first_best_label = new_label

                    max_conf = np.max(all_probs[:, 0])

                    while new_label in table_predictions:

                        all_outputs[max_index,0:topk-1]=all_outputs[max_index,1:topk]
                        all_probs[max_index,0:topk-1]=all_probs[max_index,1:topk]
                        all_outputs[max_index,topk-1]=-1
                        all_probs[max_index,topk-1]=-np.inf
                        sum_outputs = np.sum(all_outputs)

                        if sum_outputs==sum_stop:
                            new_label = first_best_label
                            max_index = first_best_max_index
                            break

                        max_index = np.argmax(all_probs[:, 0])
                        new_label = int2word[all_outputs[max_index,0]]

                        max_conf=np.max(all_probs[:, 0])

                    index_of_new_predicted_label = index_of_header[max_index]
                    table_predictions[index_of_new_predicted_label] = new_label
                    table_pred_dist[index_of_new_predicted_label] = initial_table_pred_dist[max_index]

                    final_probs[index_of_new_predicted_label] = max_conf

                    to_del_from_contexts.append(index_of_new_predicted_label)

    else:
        table_predictions=only_values_predicted_labels.copy()
        table_pred_dist=initial_table_pred_dist.copy()

    print(only_values_predicted_labels)
    print(table_predictions)
    print(groundtruth_labels)
    print(table_freq_seen_label)
    #print('\n')




    final_predicted_prob.append(table_pred_dist)

all_outputs = np.concatenate(final_predicted_prob)

ranked_labels=np.argsort(all_outputs,axis=1)[:,::-1]

mrr_score=MRR(ranked_labels,all_labels)
print('MRR={}'.format(mrr_score))

macro, micro = precision_recall_f1(all_outputs, all_labels, freq_filter=0, only_seen=True, seen_labels=seen_labels)
print('macro: {0} \n micro:{1} \n '.format(macro, micro))

topn_acc = [topn_accuracy_from_probabilities(all_outputs, all_labels, topn=i + 1, freq_filter=0, only_seen=True,
                                             seen_labels=seen_labels) for i in range(5)]
for i, acc in enumerate(topn_acc):
    print('top {} accuracy = {}'.format(i + 1, acc))



