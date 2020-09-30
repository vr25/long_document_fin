"""
Unfortunately, one cannot include the exact data utilized to the train both the clinical models due to HIPPA constraints.
The data can be found here if you fill out the appropriate online forms:
https://portal.dbmi.hms.harvard.edu/data-challenges/

For training, simply alter the config.ini present in /examples file for your purposes. Relevant variables are:

model_storage_directory: directory to store logging information, tensorboard checkpoints, model checkpoints

bert_model_path: the file path to a pretrained bert mode. can be the pytorch-transformers alias.

labels: an ordered list of labels you are training against. this should match the order given in a .fit() instance.


"""

from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
#from .bert_document_classification.document_bert import document_bert
from document_bert import BertForDocumentClassification
from pprint import pformat
import pandas as pd 
import os
import sys
import csv
import time, logging, torch, configargparse, os, socket

log = logging.getLogger()

def _initialize_arguments(p: configargparse.ArgParser):
    p.add('--model_storage_directory', help='The directory caching all model runs')
    p.add('--bert_model_path', help='Model path to BERT')
    p.add('--labels', help='Numbers of labels to predict over', type=str)
    p.add('--architecture', help='Training architecture', type=str)
    p.add('--freeze_bert', help='Whether to freeze bert', type=bool)

    p.add('--batch_size', help='Batch size for training multi-label document classifier', type=int)
    p.add('--bert_batch_size', help='Batch size for feeding 510 token subsets of documents through BERT', type=int)
    p.add('--epochs', help='Epochs to train', type=int)
    #Optimizer arguments
    p.add('--learning_rate', help='Optimizer step size', type=float)
    p.add('--weight_decay', help='Adam regularization', type=float)

    p.add('--evaluation_interval', help='Evaluate model on test set every evaluation_interval epochs', type=int)
    p.add('--checkpoint_interval', help='Save a model checkpoint to disk every checkpoint_interval epochs', type=int)

    #Non-config arguments
    p.add('--cuda', action='store_true', help='Utilize GPU for training or prediction')
    p.add('--device')
    p.add('--timestamp', help='Run specific signature')
    p.add('--model_directory', help='The directory storing this model run, a sub-directory of model_storage_directory')
    p.add('--use_tensorboard', help='Use tensorboard logging', type=bool)
    args = p.parse_args()

    args.labels = [x for x in args.labels.split(', ')]


    #Set run specific envirorment configurations
    args.timestamp = time.strftime("run_%Y_%m_%d_%H_%M_%S") + "_{machine}".format(machine=socket.gethostname())
    args.model_directory = os.path.join(args.model_storage_directory, args.timestamp) #directory
    os.makedirs(args.model_directory, exist_ok=True)

    #Handle logging configurations
    log.handlers.clear()
    formatter = logging.Formatter('%(message)s')
    fh = logging.FileHandler(os.path.join(args.model_directory, "log.txt"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    log.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    log.setLevel(logging.INFO)
    log.addHandler(ch)
    log.info(p.format_values())


    #Set global GPU state
    if torch.cuda.is_available() and args.cuda:
        if torch.cuda.device_count() > 1:
            log.info("Using %i CUDA devices" % torch.cuda.device_count() )
        else:
            log.info("Using CUDA device:{0}".format(torch.cuda.current_device()))
        args.device = 'cuda'
    else:
        log.info("Not using CUDA :(")
        args.dev = 'cpu'

    return args


if __name__ == "__main__":
    p = configargparse.ArgParser(default_config_files=["config.ini"])
    args = _initialize_arguments(p)

    torch.cuda.empty_cache()

    #documents and labels for training
    #train_documents, train_labels = ["sample train document", "zeroth train document" * 1700, "first train document sample" * 1200 , "second train document", "third train document", "fourth train document", "fifth train document", "sixth train document", "seventh train document", "eighth train document", "ninth train document", "tength train document"],[[0,0,0,0], [0,0,1,1], [0,1,0,0], [0,0,0,1], [1,1,0,0], [0,1,1,0], [0,1,0,1], [0,1,1,0], [0,1,0,0], [1,0,0,0], [0,1,1,0], [0,1,1,1]]


    #documents and labels for development
    #dev_documents, dev_labels = ["sample development document", "zeroth dev document" * 2000, "first development document", "second development document", "third development document", "fourth development document"],[[1,1,1,1], [1,0,0,0], [0,0,1,0], [0,0,0,1], [0,1,1,1], [1,1,0,0]]


    #train_documents, train_labels = ["sample train document", "zeroth train document", "first train document sample" , "second train document", "third train document", "fourth train document", "fifth train document", "sixth train document", "seventh train document", "eighth train document", "ninth train document", "tength train document"],[[0.123], [0.346], [0.364], [0.587], [0.563], [0.348], [0.346], [0.487], [0.374], [0.674], [0.2134], [0.3487]]

    #dev_documents, dev_labels = ["sample development document", "zeroth dev document" * 2000, "first development document", "second development document", "third development document", "fourth development document"],[[0.347], [0.348], [0.341], [0.3874], [0.3487], [0.3641]]

    #train_documents, train_labels = ["hi", "sample train document", "zeroth train document" * 5000, "first train document sample"],[[0.44], [0.123], [0.346], [0.364]]

    #dev_documents, dev_labels = ["sample development document", "zeroth dev document"],[[0.347], [0.348]]

    df = pd.read_csv("new_all_2436_mda_roa.csv")
    df_train = df[:1700]
    df_dev = df[1700:]

    train_documents = df_train['mda'].tolist()
    train_labels = df_train['roa'].tolist()
    train_labels = [train_labels[i:i+1] for i in range(0,len(train_labels), 1)]
    #print("train_labels: ", train_labels)
    #sys.exit(0)

    dev_documents = df_dev['mda'].tolist()
    dev_labels = df_dev['roa'].tolist()
    dev_labels = [dev_labels[i:i+1] for i in range(0,len(dev_labels), 1)]
    #print("dev_labels: ", dev_labels)
    
    model = BertForDocumentClassification(args=args)
    model.fit((train_documents, train_labels), (dev_documents,dev_labels))