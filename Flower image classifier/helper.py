# project 1's database + the .ipynb file for project 2, part 1
# helped a lot to create this file

# ----------------------------------------------------------------
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models

from collections import OrderedDict

from PIL import Image
import numpy as np

import matplotlib.pyplot as plt

import argparse

# ----------------------------------------------------------------
# function to load data
def load_data(data_dir='./flower'):
    '''
        Loads data for training, validation, and testing
    
        ___Inputs:
        None

        ___Returns:
        train_dataloader   (`torch.utils.data.DataLoader`) <- dataloader for training
        valid_Dataloader   (`torch.utils.data.DataLoader`)   <- dataloader for validation
        test_dataloader    (`torch.utils.data.DataLoader`) <- dataloader for testing
        class_to_idx

    '''
    
    # finding data paths
    # data_dir = './flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # * Applying transforms
    train_transform = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomVerticalFlip(),
                                          transforms.RandomRotation(degrees=(0, 30)),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])
    ## Not performing transforms like Grayscale, Jitter etc. that might effect color quality heavily
    ## Unfortunately this version of torchvision doesn't support RandomPerspective
    ## And I'm avoiding upgrading which might be incompatible with other modules

    valid_transform = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    test_transform  = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    # obtaininig data and applying transformation
    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transform)
    test_data  = datasets.ImageFolder(test_dir, transform=test_transform)

    # creating dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=64)

    return train_dataloader, valid_dataloader, test_dataloader, train_data.class_to_idx

# ----------------------------------------------------------------
# mapping architecture to its feature extractor's last convolution layer's `out_channels`
archs = {'alexnet'     :  9216,
         'densenet121' :  1024,
         'vgg19'       : 25088}

# function to build the neural network to classify the flowers
def nn_artist(arch='vgg19', dp=0.5, num_hiddens=1024, num_out=102, lr=0.001):
    '''
        Designs a neural network, creates a loss function, and a model optimizer.
        
        ___inputs:
        arch        (`string`)        <- model architecture [default : vgg19]
        dp          (`float` : [0-1]) <- dropout probability
        num_hiddens (`int`)           <- number of the units in 1st hidden layers
        num_out     (`int`)           <- number of output units in the output layer
        lr          (`float`)         <- learning rate to assign to the model optimizer

        ___returns:
        model       (`torchvision.models` : (pretrained)) <- model for our flower classifier
        criterion   (`torch.nn` Loss Function) <- Loss function
        optimizer   (`torch.optim` Optimizer) <- optimizer to update weights of the model
        arch        (`string`)        <- model architecture {might be altered}
    '''
    
    if arch == 'alexnet':
        model = models.alexnet(pretrained = True)      
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    else:
        model = models.vgg19(pretrained=True)
        if arch != 'vgg19':
            print('Architecture not implemented'
                  'Creating a vgg19 model')
            arch = 'vgg19'
        
    
    # freezing
    for param in model.parameters():
        param.requires_grad = False

    # updating the classifier
    classifier = nn.Sequential(OrderedDict([
                                ('fc0', nn.Linear(archs[arch], num_hiddens)),
                                ('relu0', nn.ReLU()),
                                ('drop0', nn.Dropout(dp)),
                                ('fc1', nn.Linear(num_hiddens, 512)),
                                ('relu1', nn.ReLU()),
                                ('drop1', nn.Dropout(dp)),
                                ('fc2', nn.Linear(512, 256)),
                                ('relu2', nn.ReLU()),
                                ('drop2', nn.Dropout(dp)),
                                ('fc3', nn.Linear(256, num_out)),
                                ('output', nn.LogSoftmax(dim=1))
                                ]))
    
    model.classifier = classifier
    optimizer = optim.Adam(model.classifier.parameters(), lr)

    criterion = nn.NLLLoss()
    
    return model, criterion, optimizer, arch

# ----------------------------------------------------------------
# function to train the nn model
def train_model(model, criterion, optimizer, train_dataloader, valid_dataloader, arch='vgg19', 
                num_epochs=30, print_every=32, num_hiddens=1024, num_out=102, lr=0.001, gpu=True):
    '''
        Creates and trains a model.

        ___Inputs:
        model       (`torchvision.models`)     <- model to be trained
        criterion   (`torch.nn` Loss Function) <- Loss function
        optimizer   (`torch.optim` Optimizer)  <- optimizer to update weights of the model
        training dataloader
        validation dataloader
        arch        (`str`)                    <- model architecture
        num_epochs  (`int`)                    <- number of epochs to train [default : 30]
        print_every (`int`)                    <- the frequency to print the losses and accuracy [default : 32]
        num_hiddens (`int`)                    <- number of the units in 1st hidden layers
        num_out     (`int`)                    <- number of output units in the output layer
        lr          (`float`)                  <- learning rate
        gpu         (`bool`)                   <- Train using GPU? [default : True]

        ___Returns:
        model       (`torchvision.models`)     <- model after training

    '''

    model, criterion, optimizer,_ = nn_artist(arch, 0.5, num_hiddens, num_out, lr)
    
    device = torch.device('cuda' if (gpu and torch.cuda.is_available()) else 'cpu')
    model.to(device)
    
    steps = 0
    running_loss = 0
    for e in range(num_epochs):
        print(f'___Epoch {e+1}/{num_epochs}___')
        for inputs, labels in train_dataloader:
            steps += 1
            
            
            inputs, labels = inputs.to(device), labels.to(device)
            
            # evaluating loss
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            
            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # keeping track of loss
            running_loss += loss.item()
            
            if steps % print_every == 0:
                validation_loss = 0
                accuracy = 0
                model.eval()
                # no backpropagation needed
                with torch.no_grad():
                    for inputs, labels in valid_dataloader:
                        inputs, labels = inputs.to(device), labels.to(device)

                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        validation_loss += batch_loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = (top_class == labels.view(*top_class.shape)).type(torch.FloatTensor)
                        accuracy += torch.mean(equals).item()
                
                train_loss = running_loss / print_every
                validation_loss = validation_loss / len(valid_dataloader)
                
                print(f'Train loss: {train_loss:0.3f}... '
                        f'Validation loss: {validation_loss:0.3f}... '
                        f'Accuracy: {accuracy/len(valid_dataloader):0.3%}')
                
                running_loss = 0
                model.train()
    
    return model

# ----------------------------------------------------------------
# function to check accuracy of the model
def accuracy(model, test_dataloader, gpu=True):
    '''
        Checks the accuracy using the test data.

        ___Inputs:
        model           (`torchvision.models`)     <- model to classify flowers
        test_dataloader
        gpu             (`bool`)                   <- Train using GPU? [default : True]

        ___Returns:
        None
    '''
    accuracy = 0
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            
            device = torch.device('cuda' if (gpu and torch.cuda.is_available()) else 'cpu')
            model.to(device)
            inputs, labels = inputs.to(device), labels.to(device)

            logps = model.forward(inputs)
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = (top_class == labels.view(*top_class.shape)).type(torch.FloatTensor)
            accuracy += torch.mean(equals).item()

    print(f'Accuracy on the test data: {accuracy/len(test_dataloader):0.3%}')

# ----------------------------------------------------------------
# function to save the model as a .pth checkpoint
def save_checkpoint(arch, model, optimizer, class_to_idx, num_hiddens=1024, num_epochs=40, checkpoint_path='checkpoint.pth'):
    '''
        Saves the model as .pth checkpoint at 'checkpoint.pth'

        ___Inputs:
        arch        (`string`)                <- model architecture
        model       (`torchvision.models`)    <- model for our flower classifier
        optimizer   (`torch.optim` Optimizer) <- optimizer to update weights of the model
        class_to_idx
        num_hiddens (`int`)                   <- number of the units in 1st hidden layers
        num_epochs  (`int`)                   <- number of epochs to train [default : 10]
        checkpoint_path (`str`)               <- path to the checkpoint

        ___Return:
        None
    '''

    checkpoint = {'arch'       : arch,
                  'num_hiddens' : num_hiddens,
                  'num_out' : len(class_to_idx),
                  'num_epochs' : num_epochs,
                  'class_to_idx' : class_to_idx,
                  'classifier_state_dict' : model.classifier.state_dict(),
                  'optimizer_state_dict' : optimizer.state_dict()}

    torch.save(checkpoint, checkpoint_path)

# ----------------------------------------------------------------
# function to load the model from .pth checkpoint
def load_model_optimizer(checkpoint_path='checkpoint.pth'):
    '''
        Load the stored model.

        ___Inputs:
        checkpoint_path (`str`) <- file path of the .pth checkpoint [default : 'checkpoint.pth']

        ___Returns:
        model     (`torchvision.models` : (pretrained)) <- loaded state of the classifier
        optimizer (`torch.optim` Optimizer)  <- loaded state of the optimizer
    '''
    
    checkpoint = torch.load(checkpoint_path)
    arch = checkpoint['arch']
    num_hiddens = checkpoint['num_hiddens']
    num_out = checkpoint['num_out']

    model,_,optimizer,_ = nn_artist(arch, 0.5, num_hiddens, num_out)
    model.classifier.load_state_dict(checkpoint['classifier_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer

# ----------------------------------------------------------------
# function to process image
def process_image(path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a `torch.Tensor`
    '''

    # creating a `PIL.Image` object
    pil_img = Image.open(path)

    transforms_to_do = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),                            # converting to tensor
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])

    img = transforms_to_do(pil_img)

    # `torch.Tensor` automatically does all the work w/o passing through an extra np.array (mentioned by the project rubric)
    return img

# ----------------------------------------------------------------
# function to predict the class of a given image file
def predict(image_path, model, topk=5, gpu=False):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # gets the torch tensor for the image
    img = process_image(image_path)
    # resize the tensor (add dimension for batch)
    img = img.unsqueeze_(0)
    # reference: project 1 - classifier.py

    device = torch.device('cuda' if (gpu and torch.cuda.is_available()) else 'cpu')
    model.to(device)
    img = img.to(device)

    # find probabilities
    with torch.no_grad():
        logps = model.forward(img)
    ps = torch.exp(logps)
    top_ps, top_cs = ps.topk(topk, dim=1)
    
    return top_ps, top_cs

# ----------------------------------------------------------------
# ----------------------------------------------------------------
# ----------------------------------------------------------------