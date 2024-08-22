from .utils import prepare_checkpoint_dir, prepare_client_checkpoint, local_training, test_acc, init_optimizer, Ensemble

from dataset_helper import get_client_dataloader

from commona_libs import *


def divergence(student_logits, teacher_logits):
    divergence = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(student_logits, dim=1),
        torch.nn.functional.softmax(teacher_logits, dim=1),
        reduction="batchmean",
    )  # forward KL
    return divergence

def feddf_server_updates(global_model, local_models, weight_list, test_loader, config, device):
    global_model.train()
    global_model.to(device)

    teacher_model = Ensemble(local_models, weight_list=weight_list)

    optimizer = init_optimizer(global_model, 
                               optim_name=config['feddf']['optimizer'], 
                               lr=config['feddf']['lr'], 
                               momentum=config['feddf']['momentum'])

    best_acc = -1
    with trange(config['feddf']['server_epochs']) as pbar:
        for _ in pbar:
            for _, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()

                output = global_model(data)

                teacher_logits = teacher_model(data)

                loss = divergence(output, teacher_logits)

                loss.backward()
                optimizer.step()


            acc = test_acc(global_model, test_loader, device)
            pbar.set_description(f"acc: {acc:.4f}")
            if acc > best_acc:
                best_acc = acc

    return global_model, best_acc
            

def OneShotFedDF(trainset, test_loader, client_idx_map, config, global_model, device):
    global_model.to(device)
    global_model.train()

    method_name = 'OneShotFedDF'
    save_path, _ = prepare_checkpoint_dir(config)
    # save the config to file
    if not os.path.exists(save_path + "/config.yaml"):
        save_yaml_config(save_path + "/config.yaml", config) 

    # all clients perform local training phrase, the global model is used as the initial model
    local_models = []
    local_data_size = []
    for c in range(config['client']['num_clients']):
        client_local_model_dir, start_round, best_acc, best_round, acc_list, local_model_ckpt \
            = prepare_client_checkpoint(config=config, client_idx=c, global_model=global_model)

        logger.info(f"Client {c} Starts Local Trainning--------|")
        client_dataloader = get_client_dataloader(client_idx_map[c], trainset, config['dataset']['train_batch_size'])
        local_data_size.append(len(client_idx_map[c]))

        local_model = local_training(
            model=copy.deepcopy(local_model_ckpt),
            training_data=client_dataloader,
            test_dataloader=test_loader,
            start_epoch=start_round,
            local_epochs=config['server']['local_epochs'],
            optim_name=config['server']['optimizer'],
            lr=config['server']['lr'],
            momentum=config['server']['momentum'],
            loss_name=config['server']['loss_name'],
            history_loss_list=acc_list,
            best_acc=best_acc,
            best_epoch=best_round,
            client_model_dir=client_local_model_dir,
            device=device,
            num_classes=config['dataset']['num_classes'],
            save_freq=config['checkpoint']['save_freq']
        )


        logger.info(f"Client {c} Finish Local Training--------|")

        local_models.append(copy.deepcopy(local_model))
    
    # local training is finished, start to aggregate the local models
    # the aggregation method is the same as FedAvg
    # average
    if config['server']['aggregated_by_datasize']:
        weights = [i/sum(local_data_size) for i in local_data_size]
    else:
        weights = [1/config['client']['num_clients'] for _ in range(config['client']['num_clients'])]
    

    global_model, best_acc = feddf_server_updates(
        global_model=copy.deepcopy(global_model),
        local_models=local_models,
        weight_list=weights,
        test_loader=test_loader,
        config=config,
        device=device
    )


    acc = test_acc(global_model, test_loader, device)

    logger.info(f"Test Acc: {acc}")

    # save the results
    result_dict = {
        method_name: acc,
        'best_acc': best_acc
    }
    save_yaml_config(save_path + f"/{method_name}_result.yaml", result_dict)