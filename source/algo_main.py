from dataset_helper import get_fl_dataset

from models import get_model
# from algorithms import *
from oneshot_algorithms import *

from commona_libs import *

setup_seed(1)

config = load_yaml_config('./yaml_config/oneshot/cifar10_clients5_dir0.1.yaml')
logger.info(f"config: {config}")


trainset, testset, client_idx_map = get_fl_dataset(
    config["dataset"]["data_name"], 
    config["dataset"]["root_path"], 
    config['client']['num_clients'], 
    config['distribution']['type'], 
    config['distribution']['label_num_per_client'], 
    config['distribution']['alpha'])

test_loader = torch.utils.data.DataLoader(testset, batch_size=config['dataset']['test_batch_size'], shuffle=True)

global_model = get_model(
    model_name=config['server']['model_name'],
    num_classes=config['dataset']['num_classes'],
    channels=config['dataset']['channels'],
)

device = config['device']


# OneShotFedAvg(trainset, test_loader, client_idx_map, config, global_model, device)
# OneShotEnsemble(trainset, test_loader, client_idx_map, config, global_model, device)
# OneShotFedDF(trainset, test_loader, client_idx_map, config, global_model, device)
# OneShotFADI(trainset, test_loader, client_idx_map, config, global_model, device)
# OneShotFedDAFL(trainset, test_loader, client_idx_map, config, global_model, device)
# OneShotDENSE(trainset, test_loader, client_idx_map, config, global_model, device)
# OneShotCoBoosting(trainset, test_loader, client_idx_map, config, global_model, device)
OneShotIntactOFL(trainset, test_loader, client_idx_map, config, global_model, device)

