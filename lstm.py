import numpy as np
from mlp.pytorch_experiment_scripts.arg_extractor import get_args
from mlp.pytorch_experiment_scripts.experiment_builder import ExperimentBuilder
from mlp.pytorch_experiment_scripts.model_architectures import NeuralNet, AttentionIsAllYouNeed,SuperAttention, RNN
from mlp.data_providers import DataProvider
import torch
from torch.autograd import Variable


args = get_args()  # get arguments from command line
rng = np.random.RandomState(seed=args.seed)  # set the seeds for the experiment
torch.manual_seed(seed=args.seed) # sets pytorch's seed

embed_size = 300
max_features = 120000
maxLen = 72
batch_size = args.batch_size

train_X = np.load("train_X.npy")
train_y = np.load("train_y.npy")
test_X = np.load("test_X.npy")
test_y = np.load("test_y.npy")
val_X = np.load("dev_X.npy")
val_y = np.load("dev_y.npy")
embedding_matrix= np.load("embed.npy")
#train_y = to_categorical(train_y)
#val_y = to_categorical(val_y)
#test_y = to_categorical(val_y)

x_train_fold = torch.tensor(train_X, dtype=torch.long)
y_train_fold = torch.tensor(train_y, dtype=torch.long)
train_dataset = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)
#train_data = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
train_data = DataProvider(x_train_fold,y_train_fold,batch_size)


x_dev_fold = torch.tensor(val_X, dtype=torch.long)
y_dev_fold = torch.tensor(val_y, dtype=torch.long)
dev_dataset = torch.utils.data.TensorDataset(x_dev_fold, y_dev_fold)
#val_data = torch.utils.data.DataLoader(dev_dataset, batch_size=batch_size, shuffle=True)
val_data =DataProvider(x_dev_fold,y_dev_fold,batch_size)

x_test_fold = torch.tensor(test_X, dtype=torch.long)
y_test_fold = torch.tensor(test_y, dtype=torch.long)
test_dataset = torch.utils.data.TensorDataset(x_test_fold, y_test_fold)
#test_data = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
test_data = DataProvider(x_test_fold,y_test_fold,batch_size)
if args.mode == 'att':
    custom_conv_net = NeuralNet(hidden_size = args.hidden, vocab_size=max_features,embed_size=300,embedding_tensor=embedding_matrix)
elif args.mode == 'tf':
    custom_conv_net = AttentionIsAllYouNeed(embedding_matrix)
elif args.mode == 'super':
    custom_conv_net = SuperAttention(embedding_matrix,args.hidden)
elif args.mode == 'LSTM':
    custom_conv_net = RNN(rnn_model = args.mode, hidden_size = args.hidden,vocab_size = max_features,embed_size = 300, num_output = 1, embedding_tensor = embedding_matrix) 

conv_experiment = ExperimentBuilder(network_model=custom_conv_net,
                                    experiment_name=args.experiment_name,
                                    num_epochs=args.num_epochs,
                                    weight_decay_coefficient=args.weight_decay_coefficient,
                                    use_gpu=args.use_gpu,
                                    continue_from_epoch=args.continue_from_epoch,
                                    train_data=train_data, val_data=val_data,
                                    test_data=test_data)  # build an experiment object
experiment_metrics, test_metrics = conv_experiment.run_experiment()  # run experiment and return experiment metrics
