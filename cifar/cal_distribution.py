import argparse
import os
import time
from numpy import linalg as LA

from utee import misc
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# import dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from IPython import embed

from imagenet import alexnet
plt.switch_backend('agg')



parser = argparse.ArgumentParser(description='PyTorch CIFAR-X Example')
parser.add_argument('--type', default='cifar100', help='cifar10|cifar100')
parser.add_argument('--channel', type=int, default=128, help='first conv channel (default: 32)')
parser.add_argument('--wd', type=float, default=0.00, help='weight decay')
parser.add_argument('--batch_size', type=int, default=200, help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=150, help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 1e-3)')
parser.add_argument('--gpu', default=None, help='index of gpus to use')
parser.add_argument('--ngpu', type=int, default=1, help='number of gpus to use')
parser.add_argument('--seed', type=int, default=117, help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=100,  help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=5,  help='how many epochs to wait before another test')
parser.add_argument('--logdir', default='log/default', help='folder to save to the log')
parser.add_argument('--decreasing_lr', default='80,120', help='decreasing strategy')
args = parser.parse_args()
args.logdir = os.path.join(os.path.dirname(__file__), args.logdir)
misc.logger.init(args.logdir, 'train_log')
print = misc.logger.info

# select gpu
args.gpu = misc.auto_select_gpu(mem_bound=7000,utility_bound=60, num_gpu=args.ngpu, selected_gpus=args.gpu)
args.ngpu = len(args.gpu)

# logger
misc.ensure_dir(args.logdir)
print("=================FLAGS==================")
for k, v in args.__dict__.items():
    print('{}: {}'.format(k, v))
print("========================================")

# seed
args.cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# data loader and model
assert args.type in ['cifar10', 'cifar100'], args.type
if args.type == 'cifar10':
    # train_loader, test_loader = dataset.get10(batch_size=args.batch_size, num_workers=1)
    model = alexnet.alexnet()
    model.load_state_dict(torch.load('log/default/latest.pth'))
else:
    # train_loader, test_loader = dataset.get100(batch_size=args.batch_size, num_workers=1)
    model = alexnet.alexnet()
    model.load_state_dict(torch.load('log/default/latest.pth'))

#to GPU
model = torch.nn.DataParallel(model, device_ids= range(args.ngpu))
if args.cuda:
    model.cuda()


# optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
decreasing_lr = list(map(int, args.decreasing_lr.split(',')))
print('decreasing_lr: ' + str(decreasing_lr))
best_acc, old_file = 0, None
t_begin = time.time()


norm_train=[]
norm_test=[]

def get_indices(dataset,class_name):
    indices =  []
    for i in range(len(dataset.targets)):
        if dataset.targets[i] == class_name:
            indices.append(i)
    return indices

dataset = datasets.CIFAR100(
                root='/tmp/public_dataset/pytorch', train=True, download=True,
                transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]))
idx = get_indices(dataset, 1)
train_loader = torch.utils.data.DataLoader(dataset,batch_size=args.batch_size, sampler = torch.utils.data.sampler.SubsetRandomSampler(idx))

dataset = datasets.CIFAR100(
                root='/tmp/public_dataset/pytorch', train=False, download=True,
                transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]))
idx = get_indices(dataset, 1)
test_loader = torch.utils.data.DataLoader(dataset,batch_size=args.batch_size, sampler = torch.utils.data.sampler.SubsetRandomSampler(idx))

try:
    # ready to test
    model.eval()
    test_loss = 0
    correct = 0

    i = 0
    for data, target in train_loader:
        indx_target = target.clone()
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        norm_train.append(LA.norm(output.cpu().detach().numpy()))#norm_train
        test_loss += F.cross_entropy(output, target).data
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.cpu().eq(indx_target).sum()
        i += 1
        if i >= 500:
            break

    j = 0
    for data, target in test_loader:
        indx_target = target.clone()
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        norm_test.append(LA.norm(output.cpu().detach().numpy()))#norm_test
        test_loss += F.cross_entropy(output, target).data
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.cpu().eq(indx_target).sum()
        j += 1
        if j >= 500:
            break
    
    # print(norm_test)
    # print(norm_train)
    plt.hist([norm_train,norm_test],bins=40, alpha=0.8)
    plt.legend()
    plt.savefig('grad_hist')
    

except Exception as e:
    import traceback
    traceback.print_exc()
finally:
    print("Total Elapse: {:.2f}, Best Result: {:.3f}%".format(time.time()-t_begin, best_acc))