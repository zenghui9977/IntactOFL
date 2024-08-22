from .utils import prepare_checkpoint_dir, prepare_client_checkpoint, local_training, test_acc, Ensemble

from dataset_helper import get_client_dataloader, NORMALIZE_DICT

from commona_libs import *

from .synthesizer_utils import Generator, KLDiv, DENSESynthesizer, Normalizer


def dense_kd_train(synthesizer, model, criterion, optimizer, device):

    student, teacher = model
    student.train()
    teacher.eval()
    for idx, (images, labels) in enumerate(synthesizer.get_data(labeled=True)):
        optimizer.zero_grad()
        images = images.to(device); labels = labels.to(device)
        loss_ce = torch.tensor(0).to(device)
        s_out = student(images.detach())
        with torch.no_grad():
            try:
                t_out, t_feat = teacher(images, labels, get_feature=True)
            except:
                t_out, t_feat = teacher(images, get_feature=True)
            try:
                loss_ce = torch.nn.functional.cross_entropy(s_out, labels)
            except:
                continue
        loss_kd = criterion(s_out, t_out.detach())
        loss = loss_kd
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=student.parameters(), max_norm=10)
        optimizer.step()


def dense_server_process(config, model_list, weight_list, student_model, test_loader, save_dir, methods_transform):

    method_name = 'OneShotDENSE'
    ensemble_model = Ensemble(model_list, weight_list).to(config['device'])
    # test teacher
    t_acc = test_acc(ensemble_model, test_loader, config['device'])
    logger.info(f'Teacher accuracy: {t_acc}')

    generator = Generator(nz=config['dense']['nz'], 
                          ngf=config['dense']['ngf'], 
                          img_size=config['dataset']['image_size'], 
                          nc=config['dataset']['channels']).to(config['device'])

    real_image_size = (config['dataset']['channels'], config['dataset']['image_size'], config['dataset']['image_size'])
    normalizer = Normalizer(**NORMALIZE_DICT[config['dataset']['data_name']])

    criterion = KLDiv(T=1)
    synthesizer = DENSESynthesizer(
        teacher=ensemble_model,
        mdl_list=model_list,
        student=student_model,
        generator=generator,
        nz=config['dense']['nz'],
        num_classes=config['dataset']['num_classes'],
        img_size=real_image_size,
        save_dir=os.path.join(save_dir, method_name),
        iterations=config['dense']['g_steps'],
        lr_g=config['dense']['lr_g'],
        synthesis_batch_size=config['dense']['synthesis_batch_size'],
        sample_batch_size=config['dense']['batch_size'],
        adv=config['dense']['adv'],
        bn=config['dense']['bn'],
        oh=config['dense']['oh'],
        criterion=criterion,
        transform=methods_transform,
        normalizer=normalizer,
        his=config['dense']['his'],
        batchonly=config['dense']['batchonly'],
        batchused=config['dense']['batchused'],
        device=config['device'] 
    )


    optimizer = torch.optim.SGD(student_model.parameters(), config['dense']['kd_lr'], weight_decay=config['dense']['weight_decay'], momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['dense']['epochs'])

    best_acc = -1
    with trange(config['dense']['epochs']) as pbar:
        for e in pbar:
            synthesizer.synthesize(cur_ep=e)
            teacher = ensemble_model
            teacher.to(config['device'])
            kd_criterion = KLDiv(T=config['dense']['kd_T'])

            dense_kd_train(synthesizer, [student_model, teacher], kd_criterion, optimizer, device=config['device'])

            acc = test_acc(student_model, test_loader, config['device'])

            scheduler.step()
            pbar.set_description(f"Epoch {e}, Acc {acc:.4f}")

            if acc > best_acc:
                best_acc = acc
                best_round = e

            if e % config['checkpoint']['save_freq'] == 0:
                result_dict = {
                    method_name: acc,
                    'best_acc': best_acc
                }
                save_yaml_config(save_dir + f"/{method_name}_result.yaml", result_dict)               

    return student_model, best_acc



def OneShotDENSE(trainset, test_loader, client_idx_map, config, global_model, device):
    global_model.to(device)
    global_model.train()

    method_name = 'OneShotDENSE'
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
 

    global_model, best_acc = dense_server_process(
        config=config,
        model_list=local_models,
        weight_list=weights,
        student_model=global_model,
        test_loader=test_loader,
        save_dir=save_path,
        methods_transform=trainset.transform,   
    )

    acc = test_acc(global_model, test_loader, device)

    logger.info(f"Test Acc: {acc}")

    # save the results
    result_dict = {
        method_name: acc,
        'best_acc': best_acc
    }
    save_yaml_config(save_path + f"/{method_name}_result.yaml", result_dict)