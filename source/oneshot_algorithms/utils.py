from commona_libs import *

# save files
def save_checkpoint(save_path, model, best_model_dict, rounds, best_acc, best_round, acc_list):
    os.makedirs(save_path, exist_ok=True)
    
    checkpoint_data = {
        'rounds': rounds,
        'best_acc': best_acc,
        'best_round': best_round
    }

    save_yaml_config(save_path + "/checkpoint.yaml", checkpoint_data)
    torch.save(model.state_dict(), save_path + "/current_model.pth")
    torch.save(best_model_dict, save_path + "/best_model.pth")
    save_perf_records(save_path=save_path, save_file_name='acc_list', data_dict={'Accuracy': acc_list})
    

# load files
def load_checkpoint(save_path):
    checkpoint_data = load_yaml_config(save_path + "/checkpoint.yaml")
    model_state_dict = torch.load(save_path + "/current_model.pth")
    best_model_state_dict = torch.load(save_path + "/best_model.pth")
    acc_list = read_perf_records(save_path=save_path, save_file_name='acc_list')['Accuracy']

    return checkpoint_data, model_state_dict, best_model_state_dict, acc_list


def save_perf_records(save_path, save_file_name, data_dict, save_mode='w'):
    header = data_dict.keys()

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    with open(os.path.join(save_path, save_file_name)+'.csv', save_mode, encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        if f.tell() == 0:
            w.writerow(header)
        w.writerows(zip(*data_dict.values()))

def read_perf_records(save_path, save_file_name):
    with open(os.path.join(save_path, save_file_name)+'.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        data_dict = {key: [] for key in header}
        for row in reader:
            for i, value in enumerate(row):
                try:
                    data_dict[header[i]].append(float(value))
                except ValueError:
                    data_dict[header[i]].append(value)
    return data_dict

def prepare_checkpoint_dir(config):
    # prepare the save path
    save_path = os.path.join(config['checkpoint']['save_path'], config['exp_name'])

    local_model_dir = os.path.join(save_path, 'local_models')

    os.makedirs(local_model_dir, exist_ok=True)

    return save_path, local_model_dir


def prepare_client_checkpoint(config, client_idx, global_model):

    _, local_model_dir = prepare_checkpoint_dir(config)

    client_local_model_dir = os.path.join(local_model_dir, f"{config['server']['model_name']}_local_model_client_{client_idx}/")  
    os.makedirs(client_local_model_dir, exist_ok=True)
    # whether read checkpoint
    if config['resume'] and \
        os.path.exists(client_local_model_dir + "/checkpoint.yaml") and \
        os.path.exists(client_local_model_dir + "/current_model.pth") and \
        os.path.exists(client_local_model_dir + "/best_model.pth") and \
        os.path.exists(client_local_model_dir + "/acc_list.csv"):

        checkpoint_data, model_state_dict, best_state_dict, acc_list = load_checkpoint(client_local_model_dir)

        start_round = checkpoint_data['rounds']
        best_acc = checkpoint_data['best_acc']
        best_round = checkpoint_data['best_round']
        
        local_model = copy.deepcopy(global_model)
        
        if config['resume_best']:
            local_model.load_state_dict(best_state_dict)
        else:
            local_model.load_state_dict(model_state_dict)
    
    else:
        start_round = 0
        best_acc = 0.0
        best_round = 0
        
        local_model = copy.deepcopy(global_model)
        acc_list = []


    return client_local_model_dir, start_round, best_acc, best_round, acc_list, local_model


# common training functions
def init_optimizer(model, optim_name, lr, momentum):
    if optim_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif optim_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise NotImplementedError(f"Optimizer {optim_name} is not implemented.")
    return optimizer

def init_loss_fn(loss_name):
    if loss_name == 'ce':
        criterion = torch.nn.CrossEntropyLoss()
    elif loss_name == 'mse':
        criterion = torch.nn.MSELoss()
    elif loss_name == 'nll':
        criterion = torch.nn.NLLLoss()
    elif loss_name == 'l1':
        criterion = torch.nn.L1Loss()

    else:
        raise NotImplementedError(f"Loss function {loss_name} is not implemented.")
    return criterion


def local_training(model, training_data, test_dataloader, 
                   start_epoch, local_epochs, 
                   optim_name, lr, momentum, 
                   loss_name, history_loss_list, best_acc, best_epoch, client_model_dir,
                   device='cpu', num_classes=10, save_freq=1):
    model.train()

    model.to(device)

    optimizer = init_optimizer(model, optim_name, lr, momentum)
    criterion = init_loss_fn(loss_name)


    for e in range(start_epoch, local_epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(training_data):
            
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            output = model(data)
            
            if loss_name in ['mse', 'l1']:
                target = torch.nn.functional.one_hot(target, num_classes=num_classes).float()
            loss = criterion(output, target)
            
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item()

        logger.info(f"Epoch {e} loss: {total_loss}")

        train_acc = test_acc(model, testloader=test_dataloader, device=device)
        history_loss_list.append(train_acc)

        if train_acc > best_acc:
            best_acc = train_acc
            best_epoch = e
            best_model_state_dict = copy.deepcopy(model.state_dict())
        
        # save the results
        if (e+1) % save_freq == 0:
            save_checkpoint(client_model_dir, 
                            model=copy.deepcopy(model), 
                            best_model_dict=best_model_state_dict, 
                            rounds=e+1, 
                            best_acc=best_acc, 
                            best_round=best_epoch, 
                            acc_list=history_loss_list)

    return model


# common aggregation methods

# aggregation 
def mimic_blank_model(model_proto):
    blank_model_dict = dict()
    for name, params in model_proto.items():
        blank_model_dict[name] = torch.zeros_like(params)
    return blank_model_dict

# FedAvg    
def parameter_averaging(local_models, weight_list):
    '''
    client_models: models 
    weight_list: list of (n_clients,)
    '''
    '''Aggregate the local updates by using FedAvg

    Args:
        local_models: list of models dict
        weight_list: list of model weight
    
    Return:
        aggregated model
    '''

    updates_num = len(local_models)
    aggregated_model_dict = mimic_blank_model(local_models[0])
    with torch.no_grad():
        for name, param in aggregated_model_dict.items():
                for i in range(updates_num):
                    param = param + torch.mul(local_models[i][name], weight_list[i])
                aggregated_model_dict[name] = param

    return aggregated_model_dict


# common evaluation methods
def test_acc(model, testloader, device='cpu'):
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total
    return acc


class Ensemble(torch.nn.Module):
    def __init__(self, model_list, weight_list=None):
        super(Ensemble, self).__init__()
        self.models = model_list
        if weight_list is None:
            self.weight_list = [1.0 / len(model_list) for _ in range(len(model_list))]
        else:
            self.weight_list = weight_list

    def forward(self, x, get_feature=False):
        if get_feature:
            feature_total = 0
            logits_total = 0
            for model, weight in zip(self.models, self.weight_list):
                feature, logit = model(x, get_feature=True)
                feature_total += feature * weight
                logits_total += logit * weight
            
            return feature_total, logits_total
        else:
            logits_total = 0
            for model, weight in zip(self.models, self.weight_list):
                logit = model(x) * weight
                logits_total += logit
            return logits_total