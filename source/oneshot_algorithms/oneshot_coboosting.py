from .utils import prepare_checkpoint_dir, prepare_client_checkpoint, local_training, test_acc

from dataset_helper import get_client_dataloader, NORMALIZE_DICT

from commona_libs import *

from .synthesizer_utils import WEnsemble, Generator, KLDiv, COBOOSTSynthesizer, Normalizer

def cb_kd_train(synthesizer, model, criterion, optimizer, device, odseta=8):
    student, teacher = model
    student.train()
    teacher.eval()

    for idx, (images, labels) in enumerate(synthesizer.get_data(labeled=True)):
        optimizer.zero_grad()
        images = images.to(device); labels = labels.to(device)
        loss_ce = torch.tensor(0).to(device)
        images.requires_grad = True
        try:
            random_w = torch.FloatTensor(*teacher(images, labels).shape).uniform_(-1., 1.).to(device)
            loss_ods = (random_w * torch.nn.functional.softmax(teacher(images, labels) / 4, dim=1)).sum()
        except:
            random_w = torch.FloatTensor(*teacher(images).shape).uniform_(-1., 1.).to(device)
            loss_ods = (random_w * torch.nn.functional.softmax(teacher(images) / 4, dim=1)).sum()
        loss_ods.backward()
        images = (torch.sign(images.grad) * odseta + images).detach()

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
  

def coboosting_server_process(config, model_list, student_model, test_loader, save_dir, methods_transform):

    method_name = 'OneShotCoBoosting'

    ww = torch.zeros(size=(len(model_list), 1))
    for _ww in range(len(model_list)):
        ww[_ww] = 1.0 / len(model_list)
    ww = ww.to(config['device'])
    ensemble_model = WEnsemble(model_list, ww).to(config['device'])

    # test teacher
    t_acc = test_acc(ensemble_model, test_loader, config['device'])
    logger.info(f'Teacher accuracy: {t_acc}')

    generator = Generator(nz=config['coboosting']['nz'], 
                          ngf=config['coboosting']['ngf'], 
                          img_size=config['dataset']['image_size'], 
                          nc=config['dataset']['channels']).to(config['device'])
    
    real_image_size = (config['dataset']['channels'], config['dataset']['image_size'], config['dataset']['image_size'])
    normalizer = Normalizer(**NORMALIZE_DICT[config['dataset']['data_name']])

    criterion = KLDiv(T=1)
    synthesizer = COBOOSTSynthesizer(
        teacher=ensemble_model,
        mdl_list=model_list,
        student=student_model,
        generator=generator,
        nz=config['coboosting']['nz'],
        num_classes=config['dataset']['num_classes'],
        img_size=real_image_size,
        save_dir=os.path.join(save_dir, method_name),
        iterations=config['coboosting']['g_steps'],
        lr_g=config['coboosting']['lr_g'],
        synthesis_batch_size=config['coboosting']['synthesis_batch_size'],
        sample_batch_size=config['coboosting']['batch_size'],
        adv=config['coboosting']['adv'],
        bn=config['coboosting']['bn'],
        oh=config['coboosting']['oh'],
        criterion=criterion,
        transform=methods_transform,
        normalizer=normalizer,
        weighted=config['coboosting']['weighted'],
        hs=config['coboosting']['hs'],
        wa_steps=config['coboosting']['wa_steps'],
        mu=config['coboosting']['mu'],
        wdc=config['coboosting']['wdc'],
        his=config['coboosting']['his'],
        batchonly=config['coboosting']['batchonly'],
        batchused=config['coboosting']['batchused'],
        device=config['device']
    )
    
    optimizer = torch.optim.SGD(student_model.parameters(), config['coboosting']['kd_lr'], weight_decay=config['coboosting']['weight_decay'], momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['coboosting']['epochs'])

    best_acc = -1
    with trange(config['coboosting']['epochs']) as pbar:
        for e in pbar:
            synthesizer.synthesize(cur_ep=e)
            teacher = synthesizer.teacher
            teacher.to(config['device'])
            kd_criterion = KLDiv(T=config['coboosting']['kd_T'])

            cb_kd_train(synthesizer, [student_model, teacher], kd_criterion, optimizer, device=config['device'], odseta=config['coboosting']['odseta'] / 255)

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


def OneShotCoBoosting(trainset, test_loader, client_idx_map, config, global_model, device):
    global_model.to(device)
    global_model.train()

    method_name = 'OneShotCoBoosting'
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

   

    global_model, best_acc = coboosting_server_process(
        config=config,
        model_list=local_models,
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