# Imports

import argparse
import utils

parser = argparse.ArgumentParser(description='train.py')

parser.add_argument('data_dir', nargs='*', action="store", type=str, default="/home/workspace/aipnd-project/flowers/", help="File path of data directory, default is /home/workspace/aipnd-project/flowers/")
parser.add_argument('--save_dir', dest="save_dir", action="store", type=str, default="/home/workspace/aipnd-project/checkpoint.pth", help="File path of checkpoint, default is /home/workspace/aipnd-project/checkpoint.pth")
parser.add_argument('--gpu', dest="gpu", action="store", type=str, default="gpu", help="gpu or no_gpu, default is gpu")
parser.add_argument('--arch', dest="arch", action="store", type=str, default="vgg16", help="vgg11, vgg13, vgg16 or vgg19, default is vgg16")
parser.add_argument('--learning_rate', dest="learning_rate", action="store", type=float, default=0.001, help="Learning rate, default is 0.001")
parser.add_argument('--hidden_units', dest="hidden_units", action="store", type=int, default=4096, help="Hidden units, default is 4096")
parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=3, help="Epochs, default is 3")

args = parser.parse_args()

data_dir = args.data_dir
save_dir = args.save_dir
gpu = args.gpu
arch = args.arch
learning_rate = args.learning_rate
hidden_units = args.hidden_units
epochs = args.epochs

acceptable_arch = ["vgg11", "vgg13", "vgg16", "vgg 19"]
if arch not in acceptable_arch:
    print("Error message: Please use a valid model (vgg11, vgg13, vgg16 or vgg19)")
    sys.exit()    

acceptable_gpu = ["gpu", "no_gpu"]
if gpu not in acceptable_gpu:
    print("Error message: Please use a valid gpu (gpu or no_gpu)")
    sys.exit()

if epochs < 1:
    print("Error message: Please select a number greater than 0 for epochs")
    sys.exit() 

if hidden_units < 1:
    print("Error message: Please select a number greater than 0 for hidden_units")
    sys.exit()   
    
if learning_rate <= 0:
    print("Error message: Please select a number greater than 0 for learning rate")
    sys.exit()

train_loader, valid_loader, test_loader = utils.load_data(data_dir)
model = utils.build_classifier(arch, hidden_units)
model, optimizer = utils.train_classifier(model, train_loader, valid_loader, gpu, learning_rate, epochs)
utils.save_checkpoint(arch, hidden_units, model, optimizer, save_dir)
