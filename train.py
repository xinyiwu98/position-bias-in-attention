from data import TrainDataset, TestDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from models import SAN
import argparse

def parse_index_list(index_str):
    try:
        indices = [int(idx) for idx in index_str.split(',')]
        return indices
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid index list: '{index_str}'. Must be comma-separated integers.")


# helper function to calculate accuracy
def calculate_accuracy(outputs, targets):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == targets).sum().item()
    total = targets.size(0)
    return correct / total


# define train and validation function 
def train(model, train_dataloader, 
                IC_middle_first_dataloader,
                IC_first_middle_dataloader,
                IC_first_last_dataloader,
                IC_last_first_dataloader,
                IC_middle_last_dataloader,
                IC_last_middle_dataloader,
                criterion, optimizer, 
                device, num_iter=150000, rec_freq=1):
    
    model.train()
    for i, (inputs, targets) in enumerate(train_dataloader):
        if i >= num_iter:
            break

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        if i % rec_freq == 0:
            IC_middle_first_loss, IC_middle_first_acc = validate(model, IC_middle_first_dataloader, criterion, device)
            IC_first_middle_loss, IC_first_middle_acc = validate(model, IC_first_middle_dataloader, criterion, device)
            IC_first_last_loss, IC_first_last_acc = validate(model, IC_first_last_dataloader, criterion, device)
            IC_last_first_loss, IC_last_first_acc = validate(model, IC_last_first_dataloader, criterion, device)
            IC_middle_last_loss, IC_middle_last_acc = validate(model, IC_middle_last_dataloader, criterion, device)
            IC_last_middle_loss, IC_last_middle_acc = validate(model, IC_last_middle_dataloader, criterion, device)
        
            print(f"Iter {i}:"
                  f"IC First Middle Acc - Middle First Acc: {IC_first_middle_acc - IC_middle_first_acc:.4f}, "
                  f"IC First Last Acc - IC Last First Acc: {IC_first_last_acc - IC_last_first_acc:.4f}, "
                  f"IC Middle Last Acc - IC Last Middle Acc: {IC_middle_last_acc - IC_last_middle_acc:.4f}, ")
    

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            running_accuracy += calculate_accuracy(outputs, targets)
    return running_loss / len(dataloader), running_accuracy / len(dataloader)



def main(args):
    # Hyperparameters from command line arguments
    mask_type = args.mask
    gamma = args.gamma
    bs = args.bs
    lr = args.lr
    num_iter = args.iter
    test_size = args.test_size
    N = args.seq_length
    B = args.burst
    eps = args.eps
    wd = args.wd
    num_classes = args.classes
    dim_features = args.features
    pe = args.pe
    pos_bias = args.train_bias
    index = args.index
    num_attn_layers = args.num_attn_layers
    num_prefixes = args.num_prefixes
    window_size = args.window_size
    data_type = args.data_type
    lamb = args.lamb
    residual = not args.not_res

    if pe == 'no':
        pe = None

    # Initialize the datasets
    train_dataset = TrainDataset(N=N, K=num_classes, D=dim_features, B=B, eps=eps, pos_bias=pos_bias, index=index, data_type=data_type, lamb=lamb)
    IC_first_middle_dataset = TestDataset(num_seqs=test_size, test_type='IC_first_middle', train_dataset=train_dataset, N=N, K=num_classes, D=dim_features, B=B, eps=eps, data_type=data_type, lamb=lamb)
    IC_middle_first_dataset = TestDataset(num_seqs=test_size, test_type='IC_middle_first', train_dataset=train_dataset, N=N, K=num_classes, D=dim_features, B=B, eps=eps, data_type=data_type, lamb=lamb)
    IC_first_last_dataset = TestDataset(num_seqs=test_size, test_type='IC_first_last', train_dataset=train_dataset, N=N, K=num_classes, D=dim_features, B=B, eps=eps, data_type=data_type, lamb=lamb)
    IC_last_first_dataset = TestDataset(num_seqs=test_size, test_type='IC_last_first', train_dataset=train_dataset, N=N, K=num_classes, D=dim_features, B=B, eps=eps, data_type=data_type, lamb=lamb)
    IC_middle_last_dataset = TestDataset(num_seqs=test_size, test_type='IC_middle_last', train_dataset=train_dataset, N=N, K=num_classes, D=dim_features, B=B, eps=eps, data_type=data_type, lamb=lamb)
    IC_last_middle_dataset = TestDataset(num_seqs=test_size, test_type='IC_last_middle', train_dataset=train_dataset, N=N, K=num_classes, D=dim_features, B=B, eps=eps, data_type=data_type, lamb=lamb)

    train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    IC_first_middle_dataloader = DataLoader(IC_first_middle_dataset, batch_size=bs, shuffle=False)
    IC_middle_first_dataloader = DataLoader(IC_middle_first_dataset, batch_size=bs, shuffle=False)
    IC_first_last_dataloader = DataLoader(IC_first_last_dataset, batch_size=bs, shuffle=False)
    IC_last_first_dataloader = DataLoader(IC_last_first_dataset, batch_size=bs, shuffle=False)
    IC_middle_last_dataloader = DataLoader(IC_middle_last_dataset, batch_size=bs, shuffle=False)
    IC_last_middle_dataloader = DataLoader(IC_last_middle_dataset, batch_size=bs, shuffle=False)



    # Initialize model, criterion, and optimizer

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SAN(in_channels=dim_features, hidden_channels=dim_features, out_channels=32, mask_type=mask_type, pe=pe, gamma=gamma, num_attn_layers=num_attn_layers, num_prefixes=num_prefixes, window_size=window_size, residual=residual).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()



    train(model, 
        train_dataloader, 
        IC_middle_first_dataloader, 
        IC_first_middle_dataloader,
        IC_first_last_dataloader,
        IC_last_first_dataloader,
        IC_middle_last_dataloader,
        IC_last_middle_dataloader,
        criterion, 
        optimizer, 
        device, 
        num_iter=num_iter, rec_freq=100)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for SAN model")

    # Add arguments
    parser.add_argument("--mask", type=str, default='causal', help="Type of mask to use in the model")
    parser.add_argument("--gamma", type=float, default=1, help="Gamma used in memory decay")
    parser.add_argument("--bs", type=int, default=128, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimizer")
    parser.add_argument("--wd", type=float, default=1e-6, help="Regularization for the optimizer")
    parser.add_argument("--iter", type=int, default=50000, help="Number of iterations to train the model")
    parser.add_argument("--test_size", type=int, default=10000, help="Number of test sequences")
    parser.add_argument("--burst", type=int, default=4, help="Burstiness of input sequences")
    parser.add_argument("--eps", type=float, default=0.75, help="Burstiness of input sequences")
    parser.add_argument("--classes", type=int, default=2048, help="Number of classes")
    parser.add_argument("--features", type=int, default=64, help="Number of feature dimensions")
    parser.add_argument("--pe", type=str, default='rope', help="The type of PE included")
    parser.add_argument("--train_bias", action='store_true', default=False, help="Training is biased with positional information or not")
    parser.add_argument("--index", type=parse_index_list, default=[0], help="Training data is biased toward these indices (comma-separated)")
    parser.add_argument("--seq_length", type=int, default=8, help="Number of samples in each sequence")
    parser.add_argument("--num_attn_layers", type=int, default=2, help="Number of attention layers in SAN")
    parser.add_argument("--num_prefixes", type=int, default=1, help="Number of prefixes in the prefix mask")
    parser.add_argument("--window_size", type=int, default=20, help="Size of the window in the window mask")
    parser.add_argument("--data_type", type=str, default='gaussian', help="Type of data to use for training and inference")
    parser.add_argument("--lamb", type=float, default=0.75, help="Strength of token anisotropy")
    parser.add_argument("--not_res", action='store_false', default=True, help="Training with residual connections or not")


    args = parser.parse_args()
    main(args)
