import argparse
import numpy as np
import torch
from torchvision import transforms as T
from torchvision.models import resnet50
from torch.utils.data import DataLoader
from collections import OrderedDict
from datetime import datetime

def prep_model(model_path, mode='Original'):
    if mode in ['L2', 'Linf', 'GDPR_CFE', 'SCFE']:
        state_dict = torch.load(model_path)
    elif mode == 'Original':
        state_dict = torch.load(model_path)
    else:
        NotImplementedError
        
    state_dict = OrderedDict((k.replace('model.resnet.', '', 1), v) for k, v in state_dict.items())
    
    model = resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(state_dict=state_dict)
    return model

    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['WaterBirds', 'celebA', 'SpuCO_dogs', 'cheXpert'])
    parser.add_argument('--split', choices=['train', 'test_val', 'val', 'test'])
    parser.add_argument('--show_normal_acc', choices=['yes', 'no'])
    parser.add_argument('--comb_nat_pert', choices=['yes', 'no'])
    parser.add_argument('--percentage', choices=['20', '50'])
    args = parser.parse_args()

    models_path = '../SCRATCH/CFE_models'
    dataset_path = '../SCRATCH'

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    show_normal_acc = True if args.show_normal_acc == 'yes' else False
    n_groups = 4 if args.dataset != 'cheXpert' else 12

    comb_nat_pert = True if args.comb_nat_pert == 'yes' else False

    seeds = [10, 20, 40]
    modes = ['Original','SCFE']#['Original','SCFE','GDPR_CFE','L2','Linf']
    percentage = int(args.percentage)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f'./results/metrics/{args.dataset}_{args.split}_{args.show_normal_acc}_{args.comb_nat_pert}_{timestamp}.txt'
    file_content = ''

    if args.dataset == 'WaterBirds':
        transform = T.Compose([
                    T.Resize((int(224 * (256 / 224)), int(224 * (256 / 224)),)),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]) 
    elif args.dataset == 'celebA':
        transform = T.Compose([
            T.CenterCrop(178),
            T.Resize(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif args.dataset == 'SpuCO_dogs':
        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])    
        ])
    elif args.dataset == 'cheXpert':
        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])    
        ])
    else:
        NotImplementedError
    
    if args.dataset == 'WaterBirds':
        from wilds import get_dataset
        main_data = get_dataset('waterbirds', download=False, root_dir=f'{dataset_path}/')
        train_data = main_data.get_subset('train',transform=transform)
        val_data = main_data.get_subset('val',transform=transform)
        test_data = main_data.get_subset('test',transform=transform)
    elif args.dataset == 'celebA':
        from torchvision.datasets import CelebA
        train_data = CelebA(f'{dataset_path}', split='train', transform=transform)
        val_data = CelebA(f'{dataset_path}', split='valid', transform=transform)
        test_data = CelebA(f'{dataset_path}', split='test', transform=transform)
    elif args.dataset == 'SpuCO_dogs':
        from utils.datasets import SpuCoDogsDataset
        train_data = SpuCoDogsDataset(root=f'{dataset_path}/spuco_dogs/train', transform=transform)
        val_data = SpuCoDogsDataset(root=f'{dataset_path}/spuco_dogs/val', transform=transform)
        test_data = SpuCoDogsDataset(root=f'{dataset_path}/spuco_dogs/test', transform=transform)
    elif args.dataset == 'cheXpert':
        from torchvision.transforms.functional import normalize
        from utils.yang_datasets import CheXpertNoFinding
        train_data = CheXpertNoFinding(data_path=f'{dataset_path}',split='tr',hparams=None)
        val_data = CheXpertNoFinding(data_path=f'{dataset_path}',split='va',hparams=None)
        test_data = CheXpertNoFinding(data_path=f'{dataset_path}',split='te',hparams=None)
    else:
        NotImplementedError

    test_val_data = torch.utils.data.ConcatDataset([val_data, test_data])

    test_dataloader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=3)
    val_dataloader = DataLoader(val_data, batch_size=256, shuffle=False, num_workers=3)
    test_val_dataloader = DataLoader(test_val_data, batch_size=256, shuffle=False, num_workers=3)
    train_data_dataloader = DataLoader(train_data, batch_size=256, shuffle=False, num_workers=3)

    for mode in modes:
        seeds_wga = []
        seeds_bga = []
        seeds_aga = []
        seeds_normal_acc = []

        for seed in seeds:
            if args.dataset in ['WaterBirds', 'SpuCO_dogs']:
                if mode in ['L2', 'Linf', 'GDPR_CFE', 'SCFE']:
                    if comb_nat_pert:
                        model = prep_model(model_path=f"{models_path}/{args.dataset}_{mode}_seed_{seed}_percentage_{percentage}.pt", mode=mode)
                    else:
                        model = prep_model(model_path=f"{models_path}/{args.dataset}_{mode}_seed_{seed}.pt", mode=mode)
                elif mode == 'Original':
                    model = prep_model(model_path=f"{models_path}/{args.dataset}_trained_model_seed_{seed}.pt",mode=mode)
                else:
                    NotImplementedError
            elif args.dataset in ['celebA', 'cheXpert']:
                if mode in ['L2', 'Linf', 'GDPR_CFE', 'SCFE']:
                    model = prep_model(model_path=f"{models_path}/{args.dataset}_{mode}", mode=mode)
                elif mode == 'Original':
                    model = prep_model(model_path=f"{models_path}/{args.dataset}_trained_model.pt",mode=mode)
                else:
                    NotImplementedError

            model = model.to(device)
            model.eval()

            if args.split in ['test_val']:
                dataloader = test_val_dataloader
            elif args.split == 'train':
                dataloader = train_data_dataloader
            elif args.split == 'val':
                dataloader = val_dataloader
            elif args.split == 'test':
                dataloader = test_dataloader
            else:
                NotImplementedError

            with torch.no_grad():
                correct_counts = np.zeros(n_groups)
                total_counts = np.zeros(n_groups)

                if args.dataset in ['SpuCO_dogs', 'WaterBirds']:
                    for x, y_true, metadata in dataloader:
                        assert len(y_true.shape) == 1
                        y_true = y_true.to(device)
                        x = x.to(device)

                        preds = model(x).argmax(dim=1)
                        correct = (preds == y_true).cpu()

                        labels = metadata[:, 1].to(device) if args.dataset == 'WaterBirds' else y_true
                        backgrounds = metadata[:, 0].to(device) if args.dataset == 'WaterBirds' else metadata.to(device)
                        group_ids = 2 * labels + backgrounds  # 0 to 3
                        
                        for g in range(n_groups):
                            mask = (group_ids == torch.full_like(group_ids, g)).cpu()
                            correct_counts[g] += correct[mask].sum().item()
                            total_counts[g] += mask.sum().item()
        
                elif args.dataset == 'celebA':
                    for x, y_true in dataloader:
                        assert len(y_true.shape) == 2
                        labels = y_true[:,9]
                        backgrounds = y_true[:,20]                        
                        y_true = y_true.to(device)
                        x = x.to(device)

                        preds = model(x).argmax(dim=1)
                        correct = (preds == labels.to(device)).cpu()

                        group_ids = 2 * labels + backgrounds  # 0 to 3
                        
                        for g in range(n_groups):
                            mask = (group_ids == torch.full_like(group_ids, g))
                            correct_counts[g] += correct[mask].sum().item()
                            total_counts[g] += mask.sum().item()
                
                elif args.dataset == 'cheXpert':
                    for i, x, y, a in dataloader:
                        assert len(y.shape) == 1
                        y = y.to(device)
                        x = x.to(device)
                        a = a.to(device)

                        preds = model(normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])).argmax(dim=1)
                        correct = (preds == y).cpu()

                        group_ids = (6 * y + a).cpu()  # 0 to 11
                        
                        for g in range(n_groups):
                            mask = (group_ids == torch.full_like(group_ids, g))
                            correct_counts[g] += correct[mask].sum().item()
                            total_counts[g] += mask.sum().item()
                
                else:
                    NotImplementedError

            group_accs = correct_counts / total_counts
            normal_acc = np.sum(correct_counts) / np.sum(total_counts)
            best_group_acc = group_accs.max()
            worst_group_acc = group_accs.min()
            avg_group_acc = group_accs.mean()

            seeds_bga.append(best_group_acc)
            seeds_wga.append(worst_group_acc)
            seeds_aga.append(avg_group_acc)
            seeds_normal_acc.append(normal_acc)

            if args.dataset in ['cheXpert', 'celebA']:
                break

        print('='*100)
        print(f'WGA mean for {mode} on the split {args.split} for the dataset {args.dataset}: {np.mean(seeds_wga):.4f}')
        print(f'WGA std for {mode} on the split {args.split} for the dataset {args.dataset}: {np.std(seeds_wga):.4f}')
        print('-'*100)
        print(f'BGA mean for {mode} on the split {args.split} for the dataset {args.dataset}: {np.mean(seeds_bga):.4f}')
        print(f'BGA std for {mode} on the split {args.split} for the dataset {args.dataset}: {np.std(seeds_bga):.4f}')
        print('-'*100)
        print(f'AGA mean for {mode} on the split {args.split} for the dataset {args.dataset}: {np.mean(seeds_aga):.4f}')
        print(f'AGA std for {mode} on the split {args.split} for the dataset {args.dataset}: {np.std(seeds_aga):.4f}')
        print('-'*100)

        file_content += f'WGA mean for {mode} on the split {args.split} for the dataset {args.dataset}: {np.mean(seeds_wga):.4f}\n'
        file_content += f'WGA std for {mode} on the split {args.split} for the dataset {args.dataset}: {np.std(seeds_wga):.4f}\n'
        file_content += '-'*100 + '\n'
        file_content += f'BGA mean for {mode} on the split {args.split} for the dataset {args.dataset}: {np.mean(seeds_bga):.4f}\n'
        file_content += f'BGA std for {mode} on the split {args.split} for the dataset {args.dataset}: {np.std(seeds_bga):.4f}\n'
        file_content += '-'*100 + '\n'
        file_content += f'AGA mean for {mode} on the split {args.split} for the dataset {args.dataset}: {np.mean(seeds_aga):.4f}\n'
        file_content += f'AGA std for {mode} on the split {args.split} for the dataset {args.dataset}: {np.std(seeds_aga):.4f}\n'
        file_content += '-'*100 + '\n'

        if show_normal_acc:
            print(f'Accuracy mean for {mode} on the split {args.split} for the dataset {args.dataset}: {np.mean(seeds_normal_acc):.4f}')
            print(f'Accuracy std for {mode} on the split {args.split} for the dataset {args.dataset}: {np.std(seeds_normal_acc):.4f}')
            print('='*100)

            file_content += f'Accuracy mean for {mode} on the split {args.split} for the dataset {args.dataset}: {np.mean(seeds_normal_acc):.4f}\n'
            file_content += f'Accuracy std for {mode} on the split {args.split} for the dataset {args.dataset}: {np.std(seeds_normal_acc):.4f}\n'
            file_content += '='*100 + '\n'

    with open(file_name, 'w') as f:
        f.write(file_content)