# Imports
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import os
import json

def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'    
    
    # Define transforms for the training, validation, and testing sets
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(), 
                                           normalize])
    
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(), 
                                           normalize])
    
    test_transforms = valid_transforms    
    
    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform = train_transforms) 
    valid_data = datasets.ImageFolder(valid_dir, transform = valid_transforms) 
    test_data = datasets.ImageFolder(train_dir, transform = test_transforms)
    
    global image_datasets
    image_datasets = {'train': train_data, 'valid': valid_data, 'test': test_data}
    
    # Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = 128, shuffle = True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size = 64)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = 64)
    
    return train_loader, valid_loader, test_loader

def build_classifier(arch, hidden_units):
    model = None
    if arch == "vgg11":
        model = models.vgg11(pretrained = True)
    elif arch == "vgg13":
        model = models.vgg13(pretrained = True)
    elif arch == "vgg16":
        model = models.vgg16(pretrained = True)
    elif arch == "vgg19":
        model = models.vgg19(pretrained = True)
    else:
        print("Error message: Please use a valid model (vgg11, vgg13, vgg16 or vgg19)")
        sys.exit()
        
    # Freeze the parameters so that we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False    
    
    # Remake the classifier for the 102 species of plants
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088, hidden_units)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(hidden_units, 1000)),
                              ('relu', nn.ReLU()),
                              ('fc3', nn.Linear(1000, 500)),
                              ('relu', nn.ReLU()),
                              ('fc4', nn.Linear(500, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
    
    model.classifier = classifier            
    
    return model

def train_classifier(model, train_loader, valid_loader, gpu, learning_rate, epochs):
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)    
    
    print_every = 10
    steps = 0
    
    # change to cuda
    if torch.cuda.is_available() and gpu == 'gpu':
            model.to('cuda')
    
    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(train_loader):
            steps += 1
            if torch.cuda.is_available() and gpu == 'gpu':
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            
            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                validation_loss, validation_accuracy = test_network(model, gpu, valid_loader)
    
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Training Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Loss: {:.4f}".format(validation_loss),
                      "Validation Accuracy: {:.2f}%".format(validation_accuracy))
 
                running_loss = 0
                
    return model, optimizer

def test_network(model, gpu, loader):
    correct = 0
    total = 0
    running_loss = 0
    criterion = nn.NLLLoss()
    
    with torch.no_grad():
        for data in loader:
            images, labels = data
            if torch.cuda.is_available() and gpu == 'gpu':
                images = images.to('cuda')
                labels = labels.to('cuda')
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            if torch.cuda.is_available() and gpu == 'gpu':
                outputs = outputs.cpu()
                labels = labels.cpu()
            predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted[1] == labels).sum().item()  
    accuracy = 100*correct/total
    running_loss = running_loss/len(loader)
                    
    return running_loss, accuracy
                    
def save_checkpoint(arch, hidden_units, model, optimizer, save_dir):
    checkpoint = {'arch': arch, 
                  'hidden_units': hidden_units,
                  'optimizer' : optimizer.state_dict(), 
                  'state_dict': model.state_dict(), 
                  'map': image_datasets['train'].class_to_idx}
    torch.save(checkpoint, save_dir)
    
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = build_classifier(checkpoint['arch'], checkpoint['hidden_units'])
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['map']
    return model

def predict (image_path, model, top_k, gpu):
    
    # LOAD PILLOW IMAGE
    image = Image.open(image_path)
    
    # PROCESS OUR IMAGE
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])    
    test_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(), 
                                         normalize])
    tensor_image = test_transforms(image)
    tensor_image.unsqueeze_(0)
    
    # FEED IMAGE INTO  MODEL
    
    if torch.cuda.is_available() and gpu == 'gpu':
        model = model.to('cuda')
        tensor_image = tensor_image.to('cuda')
    outputs = model(tensor_image)
    if torch.cuda.is_available() and gpu == 'gpu':
        outputs = outputs.cpu()
    outputs = torch.exp(outputs)
    
    # GET TOP K PREDICTIONS

    
    output_arr = outputs.data.numpy()[0]
    
    if top_k > len(output_arr):
        print ("Error message: top_k is larger than the number of classes, top_k will be reduced to the number of outputs")
        top_k = len(output_arr)
    
    classes = (-output_arr).argsort()[:top_k]
    probs = output_arr[classes]
    classes = classes.tolist()
    
    map = model.class_to_idx
    
    for i in range(len(classes)):
        current_class = classes[i]
        classes[i] = [key for key, value in map.items() if value == current_class][0]
    
    return probs, classes       


def get_labels(image_path, classes, cat_to_name):

    n_groups = len(classes)
    index = np.arange(n_groups)
    index = index[::-1]

    labels = []
    for c in classes:
        labels.append(cat_to_name[c])
    name = ""
    potential_names = image_path.split("/")
    for n in potential_names:
        try:
            name = cat_to_name[n]
        except:
            pass    
    return name, labels