from .utils import prepare_checkpoint_dir, prepare_client_checkpoint, local_training, test_acc, init_optimizer, Ensemble

from dataset_helper import get_client_dataloader

from commona_libs import *

from torch.utils.data import TensorDataset, DataLoader
from .synthesizer_utils import DeepInversionHook


def get_image_prior_losses(inputs_jit):
    # COMPUTE total variation regularization loss
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    loss_var_l1 = (diff1.abs() / 255.0).mean() + (diff2.abs() / 255.0).mean() + (
            diff3.abs() / 255.0).mean() + (diff4.abs() / 255.0).mean()
    loss_var_l1 = loss_var_l1 * 255.0
    return loss_var_l1, loss_var_l2


def DeepInversion(config, teacher_model, student_model, device):
    teacher_model.eval()
    student_model.eval()
    
    # init the targets and inputs
    targets = torch.randint(low=0, 
                            high=config['dataset']['num_classes'], 
                            size=(config['fedadi']['bs'],)).to(device)
    inputs = torch.randn((config['fedadi']['bs'], 
                            config['dataset']['channels'], 
                            config['dataset']['image_size'], 
                            config['dataset']['image_size'])).to(device)
    inputs.requires_grad = True
    
    
    # init the optimizer
    optimizer = torch.optim.Adam([inputs], lr=config['fedadi']['lr_g'], eps=1e-8)
    ce_loss_fn = torch.nn.CrossEntropyLoss().to(device)
    kl_loss = torch.nn.KLDivLoss(reduction='batchmean').to(device)

    loss_r_feature_layers = []
    for module in teacher_model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            loss_r_feature_layers.append(DeepInversionHook(module))

    for iter_loc in tqdm(range(config['fedadi']['generation_epochs']), desc='DeepInversion'):
        optimizer.zero_grad()
        teacher_model.zero_grad()

        outputs = teacher_model(inputs)

        # compute the loss
        loss = ce_loss_fn(outputs, targets)

        # R_prior loss 
        loss_var_l1, loss_var_l2 = get_image_prior_losses(inputs)

        # R_feature loss
        rescale = [config['fedadi']['first_bn_multiplier']] + [1. for _ in range(len(loss_r_feature_layers)-1)]
        loss_r_feature = sum([mod.r_feature * rescale[idx] for (idx, mod) in enumerate(loss_r_feature_layers)])

        # R_ADI
        loss_verifier_cig = torch.zeros(1)
        if config['fedadi']['adi_scale'] != 0.0:
            outputs_student = student_model(inputs)  
            T = 3.0
            P = torch.nn.functional.softmax(outputs_student / T, dim=1)
            Q = torch.nn.functional.softmax(outputs / T, dim=1)
            M = 0.5 * (P + Q) 

            P = torch.clamp(P, 0.01, 0.99)
            Q = torch.clamp(Q, 0.01, 0.99)
            M = torch.clamp(M, 0.01, 0.99)
            eps = 0.0

            loss_verifier_cig = 0.5 * kl_loss(torch.log(P + eps), M) + 0.5 * kl_loss(torch.log(Q + eps), M)
            loss_verifier_cig = 1.0 - torch.clamp(loss_verifier_cig, 0.0, 1.0)
        
        # l2 loss on images
        loss_l2 = torch.norm(inputs.view(config['fedadi']['bs'], -1), dim=1).mean()        

        # combining losses
        loss_aux = config['fedadi']['tv_l2'] * loss_var_l2 + \
                    config['fedadi']['tv_l1'] * loss_var_l1 + \
                    config['fedadi']['r_feature'] * loss_r_feature + \
                    config['fedadi']['l2'] * loss_l2

        if config['fedadi']['adi_scale'] != 0.0:
            loss_aux += config['fedadi']['adi_scale'] * loss_verifier_cig

        loss = config['fedadi']['main_loss_factor'] * loss + loss_aux

        loss.backward()
        optimizer.step()
    
    return inputs, targets

def divergence(student_logits, teacher_logits):
    divergence = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(student_logits, dim=1),
        torch.nn.functional.softmax(teacher_logits, dim=1),
        reduction="batchmean",
    )  # forward KL
    return divergence

def distillation(teacher_model, student_model, inputs, targets, config, device, test_loader):
    student_model.train()
    student_model.to(device)
    teacher_model.eval()
    teacher_model.to(device)

    optimizer = init_optimizer(student_model, 
                               optim_name=config['fedadi']['optimizer'], 
                               lr=config['fedadi']['lr'], 
                               momentum=config['fedadi']['momentum'])

    train_data = TensorDataset(inputs, targets)
    train_loader = DataLoader(train_data, batch_size=config['fedadi']['distill_bs'], shuffle=True)

    best_acc = -1
    with trange(config['fedadi']['distillation_epochs']) as pbar:
        for _ in pbar:
            for _, (data, label) in enumerate(train_loader):
                data, label = data.to(device), label.to(device)
                optimizer.zero_grad()

                s_outputs = student_model(inputs)
                t_outputs = teacher_model(inputs)
                loss = divergence(s_outputs, t_outputs)

                loss.backward()
                optimizer.step()

            acc = test_acc(student_model, test_loader, device)
            pbar.set_description(f"acc: {acc:.4f}")

            if acc > best_acc:
                best_acc = acc

    return student_model, best_acc


def OneShotFADI(trainset, test_loader, client_idx_map, config, global_model, device):
    global_model.to(device)
    global_model.train()

    method_name = 'OneShotFedADI'
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
    

    # data generation phrase
    # generated_inputs = torch.Tensor([]).to(device)
    # generated_targets = torch.Tensor([]).to(device)
    # for c in range(config['client']['num_clients']):
    #     inputs, targets = DeepInversion(config=config, 
    #                                     teacher_model=local_models[c], 
    #                                     student_model= global_model,
    #                                     device=device)
    #     generated_inputs = torch.cat((generated_inputs, inputs), dim=0)
    #     generated_targets = torch.cat((generated_targets, targets), dim=0)

    # distillation phrase
    teacher_model = Ensemble(local_models, weight_list=weights)

    generated_inputs, generated_targets = DeepInversion(config=config, 
                                        teacher_model=teacher_model, 
                                        student_model= global_model,
                                        device=device)

    global_model, best_acc = distillation(
        teacher_model=teacher_model,
        student_model=global_model,
        inputs=generated_inputs,
        targets=generated_targets,
        config=config,
        device=device,
        test_loader=test_loader
    )

    acc = test_acc(global_model, test_loader, device)

    logger.info(f"Test Acc: {acc}")

    # save the results
    result_dict = {
        method_name: acc,
        'best_acc': best_acc
    }
    save_yaml_config(save_path + f"/{method_name}_result.yaml", result_dict)