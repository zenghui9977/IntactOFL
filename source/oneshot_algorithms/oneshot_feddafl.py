from .utils import prepare_checkpoint_dir, prepare_client_checkpoint, local_training, test_acc, Ensemble

from dataset_helper import get_client_dataloader

from commona_libs import *

from .synthesizer_utils import Generator

def kdloss(y, teacher_scores):
    p = torch.nn.functional.log_softmax(y, dim=1)
    q = torch.nn.functional.softmax(teacher_scores, dim=1)
    l_kl = torch.nn.functional.kl_div(p, q, reduction="batchmean")
    return l_kl

def feddafl_server_process(config, global_model, teacher_model, test_loader):
    generator = Generator(img_size=config['dataset']['image_size'], 
                          nz=config['feddafl']['latent_dim'], 
                          nc=config['dataset']['channels']).to(config['device'])

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=config['feddafl']['lr_g'])
    optimizer_S = torch.optim.Adam(global_model.parameters(), lr=config['feddafl']['lr_s'])

    criterion = torch.nn.CrossEntropyLoss().to(config['device'])

    best_acc = -1
    with trange(config['feddafl']['server_epochs']) as pbar:
        for _ in pbar:
            for i in range(config['feddafl']['inter_epochs']):
                global_model.train()
                z = torch.randn(config['feddafl']['bs'], config['feddafl']['latent_dim']).to(config['device'])
                z.requires_grad = True
                optimizer_G.zero_grad()
                optimizer_S.zero_grad()
                gen_imgs = generator(z)

                outputs_T, features_T = teacher_model(gen_imgs, get_feature=True)

                pred = outputs_T.data.max(1)[1]
                loss_activation = -features_T.abs().mean()
                loss_one_hot = criterion(outputs_T,pred)
                softmax_o_T = torch.nn.functional.softmax(outputs_T, dim = 1).mean(dim = 0)
                loss_information_entropy = (softmax_o_T * torch.log10(softmax_o_T)).sum()
                loss = loss_one_hot * config['feddafl']['oh'] +\
                        loss_information_entropy * config['feddafl']['ie'] +\
                        loss_activation * config['feddafl']['a']
                loss_kd = kdloss(global_model(gen_imgs.detach()), outputs_T.detach()) 
                loss += loss_kd       
                loss.backward()
                optimizer_G.step()
                optimizer_S.step()             

            acc = test_acc(global_model, test_loader, config['device'])
            pbar.set_description(f"acc: {acc:.4f}")

            if acc > best_acc:
                best_acc = acc

    return global_model, best_acc



def OneShotFedDAFL(trainset, test_loader, client_idx_map, config, global_model, device):
    global_model.to(device)
    global_model.train()

    method_name = 'OneShotFedDAFL'
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

    teacher_model = Ensemble(local_models, weight_list=weights)

    global_model, best_acc = feddafl_server_process(
        config=config,
        global_model=global_model,
        teacher_model=teacher_model,
        test_loader=test_loader,
    )


    acc = test_acc(global_model, test_loader, device)

    logger.info(f"Test Acc: {acc}")

    # save the results
    result_dict = {
        method_name: acc,
        'best_acc': best_acc,
    }
    save_yaml_config(save_path + f"/{method_name}_result.yaml", result_dict)