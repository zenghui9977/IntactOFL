import torchvision.transforms as transforms
from pathlib import Path
from itertools import combinations
from collections import defaultdict


from commona_libs import *

NORMALIZE_DICT = {
    'MNIST': dict(mean=(0.1307,), std=(0.3081,)),
    'FMNIST': dict(mean=(0.2860,), std=(0.3530,)),
    'CIFAR10': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'CIFAR100': dict(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
    'Tiny-ImageNet': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'SVHN': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),    
}

def load_dataset(dataset_name, data_path):
    if dataset_name == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(**NORMALIZE_DICT[dataset_name])
            ])
        train_dataset = torchvision.datasets.MNIST(root=data_path, train=True, transform=transform, download=True)
        test_dataset = torchvision.datasets.MNIST(root=data_path, train=False, transform=transform, download=True)
    elif dataset_name == 'FMNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(**NORMALIZE_DICT[dataset_name])
            ])
        
        train_dataset = torchvision.datasets.FashionMNIST(root=data_path, train=True, transform=transform, download=True)
        test_dataset = torchvision.datasets.FashionMNIST(root=data_path, train=False, transform=transform, download=True)
    elif dataset_name == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(**NORMALIZE_DICT[dataset_name])
            ])
        train_dataset = torchvision.datasets.CIFAR10(root=data_path, train=True, transform=transform, download=True)
        test_dataset = torchvision.datasets.CIFAR10(root=data_path, train=False, transform=transform, download=True)

    elif dataset_name == 'CIFAR100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(**NORMALIZE_DICT[dataset_name])
        ])
        train_dataset = torchvision.datasets.CIFAR100(root=data_path, train=True, transform=transform, download=True)
        test_dataset = torchvision.datasets.CIFAR100(root=data_path, train=False, transform=transform, download=True)

    elif dataset_name == 'Tiny-ImageNet':
        train_transform = transforms.Compose([
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(**NORMALIZE_DICT[dataset_name])
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        train_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_path, 'tiny-imagenet-200/train'), transform=train_transform)
        test_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_path, 'tiny-imagenet-200/val'), transform=test_transform)

    elif dataset_name == 'SVHN':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(**NORMALIZE_DICT[dataset_name])
        ])

        train_dataset = torchvision.datasets.SVHN(root=os.path.join(data_path, 'SVHN'), split='train', download=True, transform=transform)
        test_dataset = torchvision.datasets.SVHN(root=os.path.join(data_path, 'SVHN'), split='test', download=True, transform=transform)

    else:
        logger.error(f'Dataset {dataset_name} not supported')

    logger.info(f'Dataset {dataset_name} loaded')

    return train_dataset, test_dataset


def build_dataset_idxs(dataset, dataset_name):
    preprocess_path = Path('./preprocessed_data/')
    os.makedirs(preprocess_path, exist_ok=True)
    preprocess_path = preprocess_path / f'{dataset_name}.pt'

    if not os.path.exists(preprocess_path):     
        idx_dict = {}
        for idx, data in enumerate(tqdm(dataset, desc=f'Preprocessing data, Build dataset index dict')):
            _, label = data
            if label in idx_dict:
                idx_dict[label].append(idx)
            else:
                idx_dict[label] = [idx]
        torch.save(idx_dict, preprocess_path)

        logger.info(f'Dataset index dict saved to {preprocess_path}')
    else:
        idx_dict = torch.load(preprocess_path)

        logger.info(f'Dataset index dict loaded from {preprocess_path}')

    return idx_dict 


def iid(data_size, num_users):

    idxs = np.random.permutation(data_size)
    batch_idxs = np.array_split(idxs, num_users)
    client_idx_map = {i: batch_idxs[i] for i in range(num_users)}

    return client_idx_map


def non_iid(data_idxs_dict, num_users, label_num_per_client):
    
    client_idx_map = defaultdict(list)
    label_num = len(data_idxs_dict)

    data_num_per_class = [len(data_idxs_dict[i]) for i in data_idxs_dict.keys()]

    datasize_per_client = size_of_division(num_users, sum(data_num_per_class))
    class_comb = generate_class_comb(num_users, label_num, label_num_per_client)
    
    datasize_per_client_per_class = [size_of_division(len(class_comb[i]), datasize_per_client[i]) for i in range(num_users)]

    temp = [set(i) for i in data_idxs_dict.values()]
    for c in range(num_users):
        cur_client_id = c
        for i in range(label_num_per_client):
            num_data_per_cli = datasize_per_client_per_class[c][i]
            cur_client_class_ = class_comb[c][i]

            if len(temp[cur_client_class_]) < num_data_per_cli:
                rand_set = np.random.choice(list(temp[cur_client_class_]), num_data_per_cli, replace=True)
            elif len(temp[cur_client_class_]) == num_data_per_cli:
                rand_set = np.random.choice(list(temp[cur_client_class_]), num_data_per_cli, replace=False)
            else:
                rand_set = np.random.choice(list(temp[cur_client_class_]), num_data_per_cli, replace=False)
                temp[cur_client_class_] = temp[cur_client_class_] - set(rand_set)  

            client_idx_map[cur_client_id].extend(rand_set)    

    return client_idx_map


def size_of_division(num_groups, size):
    if isinstance(num_groups, int):
        num_per_group = [size // num_groups] * num_groups
    elif isinstance(num_groups, list):
        num_per_group = [math.floor(size * w) for w in num_groups]

    for i in np.random.choice(len(num_per_group), size - sum(num_per_group), replace=False):
        num_per_group[i] += 1
    
    return num_per_group


def generate_class_comb(num_groups, num_class, num_class_each_comb):
    class_comb = list(combinations(range(num_class), num_class_each_comb))
    np.random.shuffle(class_comb)
    if len(class_comb) <= num_groups:
        for _ in range(num_groups//len(class_comb)):
            class_comb += class_comb
    class_comb = class_comb[:num_groups]

    return class_comb


def dirichlet(data_idxs_dict, num_users, alpha):
    
    client_idx_map = defaultdict(list)
    
    label_num = len(data_idxs_dict)
    each_label_num = len(data_idxs_dict[0])
    
    scalenum = 10

    image_nums = []

    for n in range(label_num):
        image_num = []
        
        random.shuffle(data_idxs_dict[n])
        sampled_probabilities = int(scalenum) * each_label_num * np.random.dirichlet(
            np.array(num_users * [alpha]))
        for user in range(num_users):
            no_imgs = int(round(sampled_probabilities[user]))
            sampled_list = data_idxs_dict[n][:min(len(data_idxs_dict[n]), no_imgs)]
            
            image_num.append(len(sampled_list))
            
            client_idx_map[user].extend(sampled_list)
            random.shuffle(data_idxs_dict[n])
        
        image_nums.append(image_num)
        # print(image_nums)
    return client_idx_map


def get_fl_dataset(dataset_name, dataset_path, num_users, distribution, distribution_params=2, alpha=0.1):
    train_set, test_set = load_dataset(dataset_name, dataset_path)
    data_idx_dict = build_dataset_idxs(train_set, dataset_name)

    if distribution == 'iid':
        client_idx_map = iid(len(train_set), num_users)
    elif distribution == 'noniid':
        client_idx_map = non_iid(data_idx_dict, num_users, distribution_params)
    elif distribution == 'dirichlet':
        client_idx_map = dirichlet(data_idx_dict, num_users, alpha)
    else:
        raise NotImplementedError
    
    return train_set, test_set, client_idx_map


def get_client_dataloader(client_idxs, trainset, batch_size):
    client_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(client_idxs),
        pin_memory=True, drop_last=True)

    return client_loader

    

