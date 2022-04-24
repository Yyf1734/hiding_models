from torch._C import device
from models.nnfunc import *
from utils import progress_bar_local
import math
import random

## For classification
def train_one_batch_for_clf(task_model = None, inputs = None, targets = None, weight_group = None):
    device = task_model.device
    model = task_model.model#.to(device)
    optimizer = task_model.optimizer
    loss_fn = task_model.loss_fn
    # inputs, targets = inputs.to(device), targets.to(device).long()
    targets = targets.long()
    model.zero_grad()
    copy_param_val(model, params = weight_group, rand_perm=task_model.rand_perm, wp_start_pos = task_model.wp_start_pos)
    model, inputs, targets = model.to(device), inputs.to(device), targets.to(device)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
    loss.backward()
    optimizer.step()
    # model, inputs, targets = model.to(CPU), inputs.to(CPU), targets.to(CPU)
    local_param_delta = copy_param_delta(model, params = weight_group, rand_perm=task_model.rand_perm, rand_perm_reverse = task_model.rand_perm_reverse, wp_start_pos = task_model.wp_start_pos)
    train_loss_for_one_batch = loss.item() ## .item(): tensor -> float
    task_model.train_loss_for_one_epoch += train_loss_for_one_batch
    # _, predicted = outputs.max(1)
    return local_param_delta

def train_one_batch_ssd(task_model = None, inputs = None, targets = None, weight_group = None):
    device = task_model.device
    model = task_model.model#.to(device)
    optimizer = task_model.optimizer
    loss_fn = task_model.loss_fn
    images = inputs
    boxes, labels = targets
    model.zero_grad()
    copy_param_val(model, params = weight_group, rand_perm=task_model.rand_perm, wp_start_pos = task_model.wp_start_pos)
    model, images, boxes, labels = model.to(device), images.to(device), boxes.to(device), labels.to(device)
    optimizer.zero_grad()
    confidence, locations = model(images)
    regression_loss, classification_loss = loss_fn(confidence, locations, labels, boxes)  # TODO CHANGE BOXES
    loss = regression_loss + classification_loss
    loss.backward()
    optimizer.step()
    local_param_delta = copy_param_delta(model, params = weight_group, rand_perm=task_model.rand_perm, rand_perm_reverse = task_model.rand_perm_reverse, wp_start_pos = task_model.wp_start_pos)
    train_loss_for_one_batch = loss.item() ## .item(): tensor -> float
    task_model.train_loss_for_one_epoch += train_loss_for_one_batch
    return local_param_delta

## For Linear Regression
def train_one_batch_for_lr(task_model = None, inputs = None, targets = None, weight_group = None):
    device = task_model.device
    model = task_model.model#.to(device)
    optimizer = task_model.optimizer
    loss_fn = task_model.loss_fn
    # inputs, targets = inputs.to(device), targets.to(device).squeeze()
    targets = targets.squeeze()
    model.zero_grad()
    copy_param_val(model, params = weight_group, rand_perm=task_model.rand_perm, wp_start_pos = task_model.wp_start_pos)
    model, inputs, targets = model.to(device), inputs.to(device), targets.to(device)
    optimizer.zero_grad()
    outputs = model(inputs).squeeze()
    loss = loss_fn(outputs, targets)
    loss.backward()
    optimizer.step()
    # model, inputs, targets = model.to(CPU), inputs.to(CPU), targets.to(CPU)
    local_param_delta = copy_param_delta(model, params = weight_group, rand_perm=task_model.rand_perm, rand_perm_reverse = task_model.rand_perm_reverse, wp_start_pos = task_model.wp_start_pos)
    train_loss_for_one_batch = loss.item() ## .item(): tensor -> float
    task_model.train_loss_for_one_epoch += train_loss_for_one_batch
    return local_param_delta

def train_one_batch_for_seq2seq(task_model = None, inputs = None, targets = None, weight_group = None):
    max_length = 10
    SOS_token = 0
    EOS_token = 1
    teacher_forcing_ratio = 0.5
    device = task_model.device
    model = task_model.model#.to(device)
    optimizer_encoder = task_model.optimizer_encoder
    optimizer_decoder = task_model.optimizer_decoder
    loss_fn = task_model.loss_fn
    model.zero_grad()
    copy_param_val(model, params = weight_group, rand_perm=task_model.rand_perm, wp_start_pos = task_model.wp_start_pos)
    model, input_tensor, target_tensor = model.to(device), inputs.squeeze(dim=0).to(device), targets.squeeze(dim=0).to(device)
    encoder_hidden = model.encoder.initHidden().to(device)
    optimizer_encoder.zero_grad()
    optimizer_decoder.zero_grad() 
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    ## encoder
    encoder_outputs = torch.zeros(max_length, model.encoder.hidden_size, device=device)
    loss = 0
    for ei in range(input_length):
        encoder_output, encoder_hidden = model.encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]
    ## decoder
    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = model.decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += loss_fn(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = model.decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += loss_fn(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break
    loss.backward()
    optimizer_encoder.step()
    optimizer_decoder.step()
    local_param_delta = copy_param_delta(model, params = weight_group, rand_perm=task_model.rand_perm, rand_perm_reverse = task_model.rand_perm_reverse, wp_start_pos = task_model.wp_start_pos)
    train_loss_for_one_batch = loss.item() / target_length ## .item(): tensor -> float
    task_model.train_loss_for_one_epoch += train_loss_for_one_batch
    return local_param_delta

# def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length
## For GAN and generator
def Image_Pair_Loss(real_images,g_iamges):
    num_pair = real_images.shape[0]
    loss  = torch.mean(torch.norm(torch.sub(real_images,g_iamges).reshape(num_pair,-1),dim=1))
    return loss

def train_one_batch_onlyG(task_model = None, inputs = None, targets = None, weight_group = None):
    ## TODO: Bugs that loss is for each batch not all the target dataset
    model = task_model.model
    optimizer = task_model.optimizer
    loss_fn = task_model.loss_fn
    model.train()
    optimizer.zero_grad()
    model.zero_grad()
    copy_param_val(model, params = weight_group, rand_perm=task_model.rand_perm, wp_start_pos = task_model.wp_start_pos)
    batch_idx = task_model.idx%task_model.pair_size
    model = model.to(task_model.device)
    task_model.g_image_pair = model(task_model.fix_noise[batch_idx].to(task_model.device))
    image_pair_loss = loss_fn(task_model.g_image_pair, task_model.image_pair[batch_idx].to(task_model.device))
    task_model.GAN_image_pair_loss_list.append(image_pair_loss.item())
    image_pair_loss.backward()
    optimizer.step()
    # model = model.to(CPU)
    local_param_delta = copy_param_delta(model, params = weight_group, rand_perm=task_model.rand_perm, rand_perm_reverse = task_model.rand_perm_reverse, wp_start_pos = task_model.wp_start_pos)
    image_pair_loss = image_pair_loss.item()
    task_model.train_loss_for_one_epoch += image_pair_loss
    task_model.idx +=1 
    return local_param_delta

def train_one_batch_onlyG_shuffle(task_model = None, inputs = None, targets = None, weight_group = None):
    # shuffle -  multi-batches 
    device = task_model.device
    model = task_model.model
    optimizer = task_model.optimizer
    loss_fn = task_model.loss_fn
    model.train()
    optimizer.zero_grad()
    model.zero_grad()
    copy_param_val(model, params = weight_group, rand_perm=task_model.rand_perm, wp_start_pos = task_model.wp_start_pos)
    model, inputs, targets = model.to(device), inputs.to(device), targets.to(device)
    task_model.g_image_pair = model(inputs.to(device))
    image_pair_loss = loss_fn(task_model.g_image_pair, targets.to(device))
    task_model.GAN_image_pair_loss_list.append(image_pair_loss.item())
    image_pair_loss.backward()
    optimizer.step()
    local_param_delta = copy_param_delta(model, params = weight_group,rand_perm=task_model.rand_perm, rand_perm_reverse = task_model.rand_perm_reverse, wp_start_pos = task_model.wp_start_pos)
    image_pair_loss = image_pair_loss.item()
    task_model.train_loss_for_one_epoch += image_pair_loss
    task_model.idx +=1 
    return local_param_delta

def train_one_batch_gan(task_model = None, inputs = None, targets = None, weight_group = None):
    device = task_model.device
    G = task_model.model.to(device)
    D = task_model.D
    optimizer = task_model.optimizer
    optimizerD = task_model.optimizerD
    loss_fn = task_model.loss_fn
    G.train()
    D.train()
    real_image = inputs.to(device)
    noise = torch.randn(real_image.shape[0],100,1,1,device=device) 
    copy_param_val(G, params = weight_group, rand_perm=task_model.rand_perm, wp_start_pos = task_model.wp_start_pos)
    task_model.fake_image = G(noise)
    optimizer.zero_grad()
    optimizerD.zero_grad()
    fake_output = D(task_model.fake_image.detach()) # backward process stoped here
    real_output = D(real_image)
    real_label = torch.ones(real_output.shape[0],device=device)
    fake_label = torch.zeros(fake_output.shape[0],device=device)
    d_loss = loss_fn(fake_output,fake_label) + loss_fn(real_output,real_label)
    d_loss.backward()
    optimizerD.step()
    # train generator
    G.zero_grad()
    optimizer.zero_grad()
    fake_output = D(task_model.fake_image)
    g_loss = loss_fn(fake_output,real_label)
    task_model.g_image_pair = G(task_model.fix_noise)
    image_pair_loss = Image_Pair_Loss(task_model.image_pair,task_model.g_image_pair)
    task_model.GAN_image_pair_loss_list.append(image_pair_loss.item())
    g_loss_total = g_loss + image_pair_loss
    g_loss_total.backward()
    optimizer.step()
    local_param_delta = copy_param_delta(G, params = weight_group, rand_perm=task_model.rand_perm, rand_perm_reverse = task_model.rand_perm_reverse, wp_start_pos = task_model.wp_start_pos)
    image_pair_loss = image_pair_loss.item()
    task_model.train_loss_for_one_epoch += image_pair_loss
    g_loss = g_loss.item()
    d_loss = d_loss.item()
    return local_param_delta

## For buffer generator
def train_one_batch_bufferG(task_model = None, inputs = None, targets = None, weight_group = None, device = None):
    targets = weight_group.get_buffers().to(device)
    if targets is None:
        print('No buffers !')
        exit()
    targets = targets
    targets_len = targets.view(-1).shape[0]
    model = task_model.model
    optimizer = task_model.optimizer
    loss_fn = Image_Pair_Loss # task_model.loss_fn
    model.train()
    optimizer.zero_grad()
    model.zero_grad()
    copy_param_val(model, params = weight_group, rand_perm=task_model.rand_perm, wp_start_pos = task_model.wp_start_pos)
    g_buffers = 2 * model(task_model.fix_noise_to_buffers).view(-1)[:targets_len]
    buffer_loss = loss_fn(g_buffers, targets)
    task_model.train_loss_list.append(buffer_loss.item())
    buffer_loss.backward()
    optimizer.step()
    local_param_delta = copy_param_delta(model, params = weight_group, rand_perm=task_model.rand_perm, rand_perm_reverse = task_model.rand_perm_reverse, wp_start_pos = task_model.wp_start_pos)
    buffer_loss = buffer_loss.item()
    return local_param_delta

def test_onlyg_for_buffers(idx = None, task_model = None, weight_group = None, save_dir= None, M_test = None, test_trainset = False, task_model_onlyG = None):
    model = task_model.model if M_test is None else M_test
    dataloader = task_model.train_loader if test_trainset else task_model.test_loader 
    ## When to activate the test_fn for intance[key] ?
    copy_param_val(model, params = weight_group, rand_perm=task_model.rand_perm, wp_start_pos = task_model.wp_start_pos)
    model = model.to(task_model.device)
    if idx >= 0 or ('cifar' not in task_model.name and 'speechcommand' not in task_model.name):
        model.eval()
    elif 'speechcommand' in task_model.name or ('cifar' in task_model.name and task_model_onlyG is None): 
        ## The model initialized with weight group has default buffers.
        print('________________________buffer params from train data________________________')
        model.train()
        for x, _ in task_model.train_loader:
            _ = model(x.to(task_model.device))
        model.eval()
    else: ## test buffers from fake data
        copy_param_val(task_model_onlyG.model, params = weight_group, rand_perm=task_model.rand_perm, wp_start_pos = task_model.wp_start_pos)
        print('________________________test buffers with fake data_________________________')
        model.train()
        for batch_idx in range(task_model_onlyG.pair_size):
            x_ = task_model_onlyG.get_fake_data(batch_idx)
            y_ = model(x_)
        # print('________________________test buffers with real data in OnlyG_________________________')
        # for batch_idx in range(task_model_onlyG.pair_size):
        #     x_ = task_model_onlyG.image_pair[batch_idx]
        #     y_ = model(x_)
        # print('________________________test buffers with origin data from each task_________________________')
        # sum = 0
        # for i,(data,_) in enumerate(task_model.train_loader):
        #     if i >= 50:
        #         break
        #     data = data.to(task_model.device)
        #     y_ = model(data)
        #     sum += data.shape[0]
        # print(sum)
        model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        dataloader_size = len(dataloader)
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(task_model.device), targets.to(task_model.device).long()
            if 'pmnist' in task_model.name:
                targets = task_model.model.change_label(targets)
            total += targets.size(0)
            outputs = model(inputs)
            loss = task_model.loss_fn(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            try:
                cur_lr = task_model.optimizer.state_dict()['param_groups'][0]['lr']
            except:
                cur_lr = task_model.optimizer.state_dict()['hypers'][0]['lr']
            progress_bar_local(batch_idx, len(dataloader), '| Epoch: %s | Loss: %.3f | LR: %.4f | %s | Acc: %.3f%% |'
                        % (task_model.test_epoch, test_loss/(batch_idx+1), cur_lr, 
                        task_model.name, 100.*correct/total))
        if idx >= 0:
            task_model.scheduler_step()
            task_model.test_epoch += 1
            task_model.train_loss_list.append(task_model.train_loss_for_one_epoch / task_model.test_interval) ## average train loss for one batch
            # print(len(task_model.train_loss_list))
            task_model.train_loss_for_one_epoch = 0
            task_model.test_loss_list.append(test_loss / dataloader_size)
            ## To save train loss list and reset train loss & train idx
            np.save(f'{save_dir}/{task_model.name}_train_loss.npy',np.array(task_model.train_loss_list))
            # x = np.load(f'{save_dir}/{task_model.name}_train_loss.npy')
            # print('train_loss', len(task_model.train_loss_list))
            np.save(f'{save_dir}/{task_model.name}_test_loss.npy',np.array(task_model.test_loss_list))
    # model = model.to(CPU)
    return 100.*correct/total 

def test_clf(idx = None, task_model = None, weight_group = None, save_dir= None, M_test = None, test_trainset = False, task_model_onlyG = None):
    model = task_model.model if M_test is None else M_test
    dataloader = task_model.train_loader if test_trainset else task_model.test_loader 
    ## When to activate the test_fn for intance[key] ?
    copy_param_val(model, params = weight_group, rand_perm=task_model.rand_perm, wp_start_pos = task_model.wp_start_pos)
    model = model.to(task_model.device)
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        dataloader_size = len(dataloader)
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(task_model.device), targets.to(task_model.device).long()
            if 'pmnist' in task_model.name:
                targets = task_model.model.change_label(targets)
            total += targets.size(0)
            outputs = model(inputs)
            loss = task_model.loss_fn(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            try:
                cur_lr = task_model.optimizer.state_dict()['param_groups'][0]['lr']
            except:
                cur_lr = task_model.optimizer.state_dict()['hypers'][0]['lr']
            progress_bar_local(batch_idx, len(dataloader), '| Epoch: %s | Loss: %.3f | LR: %.4f | %s | Acc: %.3f%% |'
                        % (task_model.test_epoch, test_loss/(batch_idx+1), cur_lr, 
                        task_model.name, 100.*correct/total))
        if idx >= 0:
            task_model.scheduler_step()
            task_model.test_epoch += 1
            task_model.train_loss_list.append(task_model.train_loss_for_one_epoch / task_model.test_interval) ## average train loss for one batch
            task_model.train_loss_for_one_epoch = 0
            task_model.test_loss_list.append(test_loss / dataloader_size)
            ## To save train loss list and reset train loss & train idx
            np.save(f'{save_dir}/{task_model.name}_train_loss.npy',np.array(task_model.train_loss_list))
            np.save(f'{save_dir}/{task_model.name}_test_loss.npy',np.array(task_model.test_loss_list))
    # model = model.to(CPU)
    return 100.*correct/total 

def test_lr(idx = None, task_model = None, weight_group = None, save_dir= None, M_test = None, test_trainset = False):
    total = 0
    mae = 0.0
    rmse = 0.0
    model = task_model.model.to(task_model.device)
    dataloader = task_model.train_loader if test_trainset else task_model.test_loader 
    copy_param_val(model, params = weight_group, rand_perm=task_model.rand_perm, wp_start_pos = task_model.wp_start_pos)
    model.eval() ## without batchnorm
    with torch.no_grad():
        dataloader_size = len(dataloader)
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(task_model.device), targets.to(task_model.device).squeeze()
            total += targets.size(0)
            outputs = model(inputs).squeeze()
            error = torch.abs(outputs - targets).sum().data
            squared_error = ((outputs - targets) * (outputs - targets)).sum().data
            mae += error.item()
            rmse += squared_error.item()
            progress_bar_local(batch_idx, len(dataloader), '| Epoch: %s | LR: %.4f | %s | mae Loss per batch: %.3f | mae Loss per data: %.3f |'
                        % (task_model.test_epoch, task_model.optimizer.state_dict()['param_groups'][0]['lr'], 
                        task_model.name, mae/(batch_idx+1), mae/total))
        if idx >= 0:
            # print('should step schedule')
            task_model.scheduler_step()
            task_model.test_epoch += 1
            task_model.train_loss_list.append(task_model.train_loss_for_one_epoch / task_model.test_interval) ## average train loss for one batch
            task_model.train_loss_for_one_epoch = 0
            task_model.test_loss_list.append(mae / dataloader_size)
            ## To save train loss list and reset train loss & train idx
            np.save(f'{save_dir}/{task_model.name}_train_loss.npy',np.array(task_model.train_loss_list))
            np.save(f'{save_dir}/{task_model.name}_test_loss.npy',np.array(task_model.test_loss_list))
    return -mae/total

## If you want to see the predicted bounding boxes in the image, please cp the model.pth to '.mobilenets-ssd-pytorch/models',
## then run 'python ./mobilenets-ssd-pytorch/ssd_test_img.py', the result is in saved in '.mobilenets-ssd-pytorch/outputs'.
def test_ssd(idx = None, task_model = None, weight_group = None, save_dir= None, M_test = None, test_trainset = False):
    device = task_model.device
    model = task_model.model.to(device)
    loader = task_model.train_loader if test_trainset else task_model.test_loader 
    copy_param_val(model, params = weight_group, rand_perm=task_model.rand_perm, wp_start_pos = task_model.wp_start_pos)
    model.eval() ## without batchnorm
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    iou_array = []
    dataloader_size = len(loader)
    for batch_idx, data in enumerate(loader):
        images, ys = data
        boxes, labels = ys
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        num += 1
        with torch.no_grad():
            confidence, locations = model(images)
            # regression_loss, classification_loss, iou = task_model.loss_fn(confidence, locations, labels, boxes)
            regression_loss, classification_loss = task_model.loss_fn(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss
        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
        progress_bar_local(batch_idx, len(loader), '| Epoch: %s | LR: %.4f | %s | loss: %.3f | regression_loss: %.3f | classification_loss: %.3f |'
                        % (task_model.test_epoch, task_model.optimizer.state_dict()['param_groups'][0]['lr'], 
                        task_model.name, running_loss / num, running_regression_loss / num, running_classification_loss / num))
        # iou_array.append(iou.view(-1))
    # iou_array=torch.cat(iou_array, dim=0)
    # print('iou_mean: ', iou_array.mean())
    if idx >= 0:
        task_model.scheduler_step()
        task_model.test_epoch += 1
        task_model.train_loss_list.append(task_model.train_loss_for_one_epoch / task_model.test_interval) ## average train loss for one batch
        task_model.train_loss_for_one_epoch = 0
        task_model.test_loss_list.append( running_loss / dataloader_size)
        ## To save train loss list and reset train loss & train idx
        np.save(f'{save_dir}/{task_model.name}_train_loss.npy',np.array(task_model.train_loss_list))
        np.save(f'{save_dir}/{task_model.name}_test_loss.npy',np.array(task_model.test_loss_list))
    return -running_loss / num

def test_seq2seq(idx = None, task_model = None, weight_group = None, save_dir= None, M_test = None, test_trainset = False):
    max_length=10
    SOS_token = 0
    EOS_token = 1
    total = 0
    loss = 0.0
    device = task_model.device
    model = task_model.model.to(device)
    loss_fn = task_model.loss_fn
    dataloader = task_model.train_loader if test_trainset else task_model.test_loader 
    copy_param_val(model, params = weight_group, rand_perm=task_model.rand_perm, wp_start_pos = task_model.wp_start_pos)
    model.eval() ## without batchnorm
    with torch.no_grad():
        candidate = []
        reference = []
        dataloader_size = len(dataloader)
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            model, input_tensor, target_tensor = model.to(device), inputs.squeeze(dim=0).to(task_model.device), targets.squeeze(dim=0).to(task_model.device)
            input_length = input_tensor.size()[0]
            total += 1
            encoder_hidden =model.encoder.initHidden().to(device)
            encoder_outputs = torch.zeros(max_length, model.encoder.hidden_size, device=device)
            for ei in range(input_length):
                encoder_output, encoder_hidden = model.encoder(input_tensor[ei],encoder_hidden)
                encoder_outputs[ei] += encoder_output[0, 0]
            decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
            decoder_hidden = encoder_hidden
            decoded_words = []
            decoder_attentions = torch.zeros(max_length, max_length)
            loss_local = 0
            target_length = target_tensor.size(0)
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention =model.decoder(decoder_input, decoder_hidden, encoder_outputs)
                loss_local += loss_fn(decoder_output, target_tensor[di])
                topv, topi = decoder_output.data.topk(1)
                if topi.item() == EOS_token:
                    decoded_words.append('<EOS>')
                    break
                else:
                    decoded_words.append(task_model.output_lang.index2word[topi.item()])
                decoder_input = topi.squeeze().detach()
            if idx < 0:
                candidate.append(decoded_words)
                reference.append([[task_model.output_lang.index2word[i.item()] for i in target_tensor]])
            # if batch_idx > 50:
            #     break
            # return decoded_words, decoder_attentions[:di + 1]
            loss += loss_local.item() / target_length
            progress_bar_local(batch_idx, len(dataloader), '| Epoch: %s | %s | Loss: %.3f |'
                        % (task_model.test_epoch, 
                        task_model.name, loss/(batch_idx+1)))
        if idx >= 0:
            task_model.scheduler_step()
            task_model.test_epoch += 1
            task_model.train_loss_list.append(task_model.train_loss_for_one_epoch / task_model.test_interval) ## average train loss for one batch
            task_model.train_loss_for_one_epoch = 0
            task_model.test_loss_list.append(loss / dataloader_size)
            ## To save train loss list and reset train loss & train idx
            np.save(f'{save_dir}/{task_model.name}_train_loss.npy',np.array(task_model.train_loss_list))
            np.save(f'{save_dir}/{task_model.name}_test_loss.npy',np.array(task_model.test_loss_list))
        else:
            from torchtext.data.metrics import bleu_score
            for i in range(10):
                print(f'{reference[i]} -> {candidate[i]}')
            bs = bleu_score(candidate, reference)
            print(f'bleu_score is {bs}')
    return -loss/total

def test_gan(task_model,save_dir= None):
    task_model.scheduler_step()
    task_model.test_epoch += 1
    task_model.train_loss_list.append(task_model.train_loss_for_one_epoch / task_model.test_interval) ## average train loss for one batch
    task_model.train_loss_for_one_epoch = 0
    ## To save train loss list and reset train loss & train idx
    np.save(f'{save_dir}/{task_model.name}_train_loss.npy',np.array(task_model.train_loss_list))
    return (-task_model.train_loss_list[-1])


def get_buffer_param(task_model_instance, weight_group, use_test_dataloader = False):
    try:
        for key in task_model_instance:
            if 'ones' not in task_model_instance[key].param_num:
                print('________________________{} No buffer params________________________'.format(key))
                continue
            model = task_model_instance[key].model
            dataloader = task_model_instance[key].test_loader if use_test_dataloader else task_model_instance[key].train_loader 
            copy_param_val(model, params = weight_group, rand_perm=task_model_instance[key].rand_perm, wp_start_pos = task_model_instance[key].wp_start_pos)
            model = model.to(task_model_instance[key].device)
            ## The model initialized with weight group has default buffers.
            print('________________________{} buffer params from {} dataloader________________________'.format(key, 'test' if use_test_dataloader else 'train'))
            model.train()
            for x, _ in dataloader:
                _ = model(x.to(task_model_instance[key].device))
            model.eval()
    except:
        key = task_model_instance.name
        if 'ones' not in task_model_instance.param_num:
            print('________________________{} No buffer params________________________'.format(key))
            return
        model = task_model_instance.model
        dataloader = task_model_instance.test_loader if use_test_dataloader else task_model_instance.train_loader 
        copy_param_val(model, params = weight_group, rand_perm=task_model_instance.rand_perm, wp_start_pos = task_model_instance.wp_start_pos)
        model = model.to(task_model_instance.device)
        ## The model initialized with weight group has default buffers.
        print('________________________{} buffer params from {} dataloader________________________'.format(key, 'test' if use_test_dataloader else 'train'))
        model.train()
        for x, _ in dataloader:
            _ = model(x.to(task_model_instance.device))
        model.eval()