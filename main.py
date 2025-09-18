import warnings
warnings.filterwarnings('ignore')
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet18

import copy
import numpy as np
import random
from tqdm import trange

from utils.options import args_parser
from utils.sampling import noniid
from utils.dataset import load_data
from utils.test import test_img
from utils.byzantine_fl import krum, trimmed_mean, fang
from utils.attack import compromised_clients, untargeted_attack
from src.aggregation import server_avg, server_dyn
from src.plugin import plugin
from src.update import BenignUpdate, CompromisedUpdate

# Import debugging functions
from debug_main import run_debug_suite, suggest_optimizations, debug_args_config

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    writer = SummaryWriter(f'runs/') if args.tsboard else None

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_train, dataset_test, dataset_val = load_data(args)
    
    # Auto-detect number of classes if using CIFAR-10 fallback
    if hasattr(dataset_train, 'classes'):
        # CIFAR-10 case
        args.num_classes = len(dataset_train.classes)
        print(f"Detected {args.num_classes} classes (using CIFAR-10)")
    elif hasattr(dataset_train, 'dataset') and hasattr(dataset_train.dataset, 'classes'):
        # Subset case (CIFAR-10 with train/val split)
        args.num_classes = len(dataset_train.dataset.classes)
        print(f"Detected {args.num_classes} classes (using CIFAR-10)")
    else:
        print(f"Using {args.num_classes} classes (from args)")
    
    # early stopping hyperparameters
    cnt = 0
    check_acc = 0

    # sample users
    dict_users = noniid(dataset_train, args)
    net_glob = resnet18(num_classes = args.num_classes).to(args.device)

    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    if args.c_frac > 0:
        compromised_idxs = compromised_clients(args)
    else:
        compromised_idxs = []
    
    compromised_num = int(args.c_frac * args.num_clients)
    idxs_users = [i for i in range(args.num_clients)]
    print(idxs_users)
    print(compromised_idxs)
    
    # Run debug suite if debug mode is enabled
    if args.debug:
        print("🔍 Running comprehensive debug suite...")
        debug_args_config(args)
        run_debug_suite(
            model=net_glob,
            dataset_train=dataset_train,
            dataset_test=dataset_test,
            dict_users=dict_users,
            compromised_idxs=compromised_idxs,
            args=args
        )
        
        # Suggest optimizations based on current performance
        suggest_optimizations(current_time_per_iteration=123.21)  # Your reported time
        
        # Ask user if they want to continue with full training
        try:
            response = input("Continue with full training? (y/n): ")
            if response.lower() != 'y':
                print("Exiting for optimization...")
                exit()
        except:
            print("Running in non-interactive mode, continuing with training...")
    
    controls = {}
    
    for iter in trange(args.global_ep):
        w_locals = []
        sorted_idxs = []
        
        for idx in idxs_users:
            if idx in compromised_idxs:
                if args.p == "untarget":
                    w_locals.append(copy.deepcopy(untargeted_attack(net_glob.state_dict(), args)))
                else:
                    local = CompromisedUpdate(args = args, dataset = dataset_train, idxs = dict_users[idx])
                    w = local.train(net = copy.deepcopy(net_glob).to(args.device))
                    w_locals.append(copy.deepcopy(w))
            else:
                local = BenignUpdate(args = args, dataset = dataset_train, idxs = dict_users[idx])
                w = local.train(net = copy.deepcopy(net_glob).to(args.device))
                w_locals.append(copy.deepcopy(w))

        if args.plugin:
            scores = plugin(copy.deepcopy(w_locals), copy.deepcopy(net_glob), args)
            sorted_idxs = scores.sum(dim=0).argsort(descending=True)[: len(w_locals) - compromised_num].tolist()
            w_locals = [copy.deepcopy(w_locals[i]) for i in sorted_idxs]
        
        # update global weights
        if args.method == 'fedavg' or args.method == 'fedprox' or args.method == 'fedrs' or args.method == 'fedsam' or args.method == 'fedspeed':
            w_glob = server_avg(w_locals)
        elif args.method == 'feddyn':
            w_glob, controls = server_dyn(args, global_model=copy.deepcopy(net_glob), w_locals=w_locals, idx_users=idxs_users, controls=controls, sorted_idxs=sorted_idxs)
        elif args.method == 'krum':
            w_glob, _ = krum(w_locals, compromised_num, args)
        elif args.method == 'trimmed_mean':
            w_glob = trimmed_mean(w_locals, compromised_num, args)
        elif args.method == 'fang':
            w_glob = fang(w_locals, dataset_val, compromised_num, args)
        else:
            exit('Error: unrecognized aggregation technique')
        
        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        test_acc, test_loss = test_img(net_glob.to(args.device), dataset_test, args)

        if args.debug:
            print(f"Round: {iter}")
            print(f"Test accuracy: {test_acc}")
            print(f"Test loss: {test_loss}")
            print(f"Check accuracy: {check_acc}")
            print(f"patience: {cnt}")

        if check_acc == 0:
            check_acc = test_acc
        elif test_acc < check_acc + args.delta:
            cnt += 1
        else:
            check_acc = test_acc
            cnt = 0

        # early stopping
        if cnt == args.patience:
            print('Early stopped federated training!')
            break

        # tensorboard
        if args.tsboard:
            writer.add_scalar(f'testacc/{args.method}_{args.plugin}_{args.p}_cfrac_{args.c_frac}_alpha_{args.alpha}', test_acc, iter)
            writer.add_scalar(f'testloss/{args.method}_{args.plugin}_{args.p}_cfrac_{args.c_frac}_alpha_{args.alpha}', test_loss, iter)

    if args.tsboard:
        writer.close()
