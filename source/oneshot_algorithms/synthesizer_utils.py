from abc import ABC, abstractmethod
from typing import Dict
from PIL import Image

from commona_libs import *

def normalize(tensor, mean, std, reverse=False):
    if reverse:
        _mean = [ -m / s for m, s in zip(mean, std) ]
        _std = [ 1/s for s in std ]
    else:
        _mean = mean
        _std = std
    
    _mean = torch.as_tensor(_mean, dtype=tensor.dtype, device=tensor.device)
    _std = torch.as_tensor(_std, dtype=tensor.dtype, device=tensor.device)
    tensor = (tensor - _mean[None, :, None, None]) / (_std[None, :, None, None])
    return tensor

class Normalizer(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x, reverse=False):
        return normalize(x, self.mean, self.std, reverse=reverse)

def kldiv(logits, targets, T=1.0, reduction='batchmean'):
    q = torch.nn.functional.log_softmax(logits/T, dim=1)
    p = torch.nn.functional.softmax( targets/T, dim=1 )
    return torch.nn.functional.kl_div( q, p, reduction=reduction ) * (T*T)


class KLDiv(torch.nn.Module):
    def __init__(self, T=1.0, reduction='batchmean'):
        super().__init__()
        self.T = T
        self.reduction = reduction

    def forward(self, logits, targets):
        return kldiv(logits, targets, T=self.T, reduction=self.reduction)

class DeepInversionHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = module

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)
        #forcing mean and variance to match between two distributions
        #other ways might work better, i.g. KL divergence
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
            module.running_mean.data - mean, 2)
        self.r_feature = r_feature

    def remove(self):
        self.hook.remove()

class Generator(torch.nn.Module):
    def __init__(self, nz=100, ngf=64, img_size=32, nc=3):
        super(Generator, self).__init__()

        self.init_size = img_size // 4
        self.l1 = torch.nn.Sequential(torch.nn.Linear(nz, ngf * 2 * self.init_size ** 2))

        self.conv_blocks = torch.nn.Sequential(
            torch.nn.BatchNorm2d(ngf * 2),
            torch.nn.Upsample(scale_factor=2),

            torch.nn.Conv2d(ngf*2, ngf*2, 3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(ngf*2),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Upsample(scale_factor=2),

            torch.nn.Conv2d(ngf*2, ngf, 3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(ngf),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            torch.nn.Sigmoid(),  
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class DataIter(object):
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self._iter = iter(self.dataloader)
    
    def next(self):
        try:
            data = next( self._iter )
        except StopIteration:
            self._iter = iter(self.dataloader)
            data = next( self._iter )
        return data

def pack_images(images, col=None, channel_last=False, padding=1):
    # N, C, H, W
    if isinstance(images, (list, tuple) ):
        images = np.stack(images, 0)
    if channel_last:
        images = images.transpose(0,3,1,2) # make it channel first
    assert len(images.shape)==4
    assert isinstance(images, np.ndarray)

    N,C,H,W = images.shape
    if col is None:
        col = int(math.ceil(math.sqrt(N)))
    row = int(math.ceil(N / col))
    
    pack = np.zeros( (C, H*row+padding*(row-1), W*col+padding*(col-1)), dtype=images.dtype )
    for idx, img in enumerate(images):
        h = (idx // col) * (H+padding)
        w = (idx % col) * (W+padding)
        pack[:, h:h+H, w:w+W] = img
    return pack

def save_image_batch(imgs, output, batch_id=None,col=None, size=None, pack=True):
    if isinstance(imgs, torch.Tensor):
        imgs = (imgs.detach().clamp(0, 1).cpu().numpy() * 255).astype('uint8')
    base_dir = os.path.dirname(output)
    if base_dir != '':
        os.makedirs(base_dir, exist_ok=True)
    if pack:
        imgs = pack_images(imgs, col=col).transpose(1, 2, 0).squeeze()
        imgs = Image.fromarray(imgs)
        if size is not None:
            if isinstance(size, (list, tuple)):
                imgs = imgs.resize(size)
            else:
                w, h = imgs.size
                max_side = max(h, w)
                scale = float(size) / float(max_side)
                _w, _h = int(w * scale), int(h * scale)
                imgs = imgs.resize([_w, _h])
        imgs.save(output)
    else:
        output_filename = output
        for idx, img in enumerate(imgs):
            if img.shape[0] == 1:
                img = Image.fromarray(img[0])
            else:
                img = Image.fromarray(img.transpose(1, 2, 0))
            img.save(output_filename + '%d-%d.png' % (batch_id, idx))

def save_image_batch_labeled(imgs, targets, batch_dir, batch_id=None,col=None, size=None, pack=True):
    if isinstance(imgs, torch.Tensor):
        imgs = (imgs.detach().clamp(0, 1).cpu().numpy() * 255).astype('uint8')
    base_dir = os.path.dirname(batch_dir)
    if base_dir != '':
        os.makedirs(base_dir, exist_ok=True)
    if pack == False:
        output_filename = batch_dir
        for idx, img in enumerate(imgs):
            os.makedirs(batch_dir + str(targets[idx].item()) + "/", exist_ok=True)
            if img.shape[0] == 1:
                img = Image.fromarray(img[0])
            else:
                img = Image.fromarray(img.transpose(1, 2, 0))
            img.save(output_filename + str(targets[idx].item()) + "/"+ '%d-%d.png' % (batch_id, idx))

class LabeledImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = os.path.abspath(root)
        self.categories = [int(f) for f in os.listdir(root)]
        images = []
        targets = []
        for c in self.categories:
            category_dir = os.path.join(self.root, str(c))
            _images = [os.path.join(category_dir, f) for f in os.listdir(category_dir)]
            images.extend(_images)
            targets.extend([c for _ in range(len(_images))])
        self.images = images
        self.targets = targets
        self.transform = transform

    def __getitem__(self, idx):
        img, target = Image.open(self.images[idx]), self.targets[idx]
        if self.transform:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.images)


def _collect_all_images(root, postfix=['png', 'jpg', 'jpeg', 'JPEG']):
    images = []
    if isinstance(postfix, str):
        postfix = [postfix]
    for dirpath, dirnames, files in os.walk(root):
        for pos in postfix:
            for f in files:
                if f.endswith(pos):
                    # pdb.set_trace()
                    images.append(os.path.join(dirpath, f))
    return images


class UnlabeledImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, batch_dir, transform=None):
        self.root = os.path.abspath(batch_dir)
        self.images = _collect_all_images(self.root)  # [ os.path.join(self.root, f) for f in os.listdir( root ) ]
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open(self.images[idx])
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        return 'Unlabeled data:\n\troot: %s\n\tdata mount: %d\n\ttransforms: %s' % (
            self.root, len(self), self.transform)


class ImagePool(object):
    def __init__(self, root):
        self.root = os.path.abspath(root)
        self.batch_dir = None
        self.batch_id = 0
        os.makedirs(self.root, exist_ok=True)

    def add(self, imgs, batch_id = None, targets=None, his=True):
        self.batch_id = batch_id
        if targets == None:
            if his == False:
                batch_dir = os.path.join(self.root, "%d" % (batch_id)) + "/"
                self.batch_dir = batch_dir
            else:
                batch_dir = os.path.join(self.root, "%d" % (0)) + "/"
                self.batch_dir = batch_dir
            os.makedirs(self.batch_dir, exist_ok=True)
            save_image_batch(imgs, batch_dir,batch_id=self.batch_id, pack=False)
        else:
            if his == False:
                batch_dir = os.path.join(self.root, "%d" % (batch_id)) + "/"
                self.batch_dir = batch_dir
            else:
                batch_dir = os.path.join(self.root, "%d" % (0)) + "/"
                self.batch_dir = batch_dir

            os.makedirs(self.batch_dir, exist_ok=True)
            save_image_batch_labeled(imgs, targets, batch_dir,batch_id=self.batch_id, pack=False)

    def get_dataset(self, transform=None, labeled=False):
        if labeled == False:
            return UnlabeledImageDataset(self.root, batch_dir = self.batch_dir, transform=transform)
        else:
            return LabeledImageDataset(self.batch_dir, transform=transform)


class BaseSynthesis(ABC):
    def __init__(self, teacher, student):
        super(BaseSynthesis, self).__init__()
        self.teacher = teacher
        self.student = student
    
    @abstractmethod
    def synthesize(self) -> Dict[str, torch.Tensor]:
        """ take several steps to synthesize new images and return an image dict for visualization. 
            Returned images should be normalized to [0, 1].
        """
        pass
    
    @abstractmethod
    def sample(self, n):
        """ fetch a batch of training data. 
        """
        pass



class COBOOSTSynthesizer(BaseSynthesis):
    def __init__(self, teacher, mdl_list, student, generator, nz, num_classes, img_size, save_dir, iterations=1,
                 lr_g=1e-3, synthesis_batch_size=128, sample_batch_size=128,
                 adv=0, bn=0, oh=0, balance=0, criterion=None,transform=None,
                 normalizer=None,
                 # TODO: FP16 and distributed training
                 autocast=None, use_fp16=False, distributed=False, weighted=False, hs=1.0, wa_steps=1, mu=0.01, wdc=0.99, his=True, batchonly=False, batchused=False, device='cuda:0'):
        super(COBOOSTSynthesizer, self).__init__(teacher, student)

        self.mdl_list = mdl_list
        
        assert len(img_size) == 3, "image size should be a 3-dimension tuple"
        self.img_size = img_size
        self.iterations = iterations
        self.save_dir = save_dir
        self.transform = transform

        self.nz = nz
        self.num_classes = num_classes
        if criterion is None:
            criterion = kldiv
        self.criterion = criterion
        self.normalizer = normalizer
        self.synthesis_batch_size = synthesis_batch_size
        self.sample_batch_size = sample_batch_size

        self.data_pool = ImagePool(root=self.save_dir)
        self.data_iter = None
        # scaling factors
        self.lr_g = lr_g
        self.adv = adv
        self.bn = bn
        self.oh = oh
        self.balance = balance
        # generator
        self.generator = generator.to(device).train()
        self.distributed = distributed
        self.use_fp16 = use_fp16
        self.autocast = autocast  # for FP16
        self.hooks = []
        # hooks for deepinversion regularization

        for m_list in self.mdl_list:
            for m in m_list.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    self.hooks.append(DeepInversionHook(m))
        self.clnt_cls_weight_matrix = torch.ones(size=(len(self.mdl_list), self.num_classes))

        self.weighted = weighted
        self.hs = hs
        self.wa_steps = wa_steps
        self.mu = mu
        self.wdc = wdc
        self.his = his
        self.batchonly = batchonly
        self.batchused = batchused

        self.device = device

    def synthesize(self, cur_ep=None):
        ###########
        # 设置eval模式
        ###########

        self.student.eval()
        self.generator.train()
        self.teacher.eval()
        for m in self.mdl_list:
            m.eval()

        if self.bn == 0:
            self.hooks = []
        best_cost = 1e6
        z = torch.randn(size=(self.synthesis_batch_size, self.nz)).to(self.device)
        z.requires_grad = True
        targets = torch.randint(low=0, high=self.num_classes, size=(self.synthesis_batch_size,))
        # targets = targets.sort()[0]
        targets = targets.to(self.device)
        reset_model(self.generator)

        optimizer = torch.optim.Adam([{'params': self.generator.parameters()}, {'params': [z]}], self.lr_g,
                                     betas=[0.5, 0.999])
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( optimizer, T_max=self.iterations )

        for it in range(self.iterations):
            optimizer.zero_grad()
            inputs = self.generator(z)
            
            inputs = self.normalizer(inputs)
            t_out = self.teacher(inputs)

            if len(self.hooks) == 0 or self.bn == 0:
                loss_bn = torch.tensor(0).to(self.device)
            else:
                loss_bn = sum([h.r_feature for h in self.hooks]) / len(self.mdl_list)

            # hard sample mining
            a = torch.nn.functional.softmax(t_out, dim=1)
            mask = torch.zeros_like(a)
            b = targets.unsqueeze(1)
            mask = mask.scatter_(1, b, torch.ones_like(b).float())
            p = a[mask.bool()]
            loss_oh = ((1-p.detach()).pow(self.hs) * torch.nn.CrossEntropyLoss(reduction='none')(t_out, targets)).mean()

            s_out = self.student(inputs)

            loss_adv = -(kldiv(s_out, t_out,T = 3, reduction='none').sum(1)).mean()  # decision adversarial distillation

            loss = self.bn * loss_bn + self.oh * loss_oh + self.adv * loss_adv
            

            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=self.generator.parameters(), max_norm=10)
            for m in self.mdl_list:
                m.zero_grad()
            optimizer.step()
            # scheduler.step()

            # torch.cuda.empty_cache()

            if best_cost > loss.item():
                best_cost = loss.item()
                best_inputs = inputs.data

        if self.weighted and cur_ep != 0:
            mix_weight = self.teacher.mdl_w_list.detach()
            ori_weight = self.teacher.mdl_w_list
            best_loss = 1e3
            for w_adjust in range(self.wa_steps):
                for idx, (images, labels) in enumerate(self.get_data(labeled=True)):
                    images = images.to(self.device); labels = labels.to(self.device)
                    mix_weight.requires_grad = True
                    tmp_model = WEnsemble(self.mdl_list, mix_weight).to(self.device)
                    # forward
                    # tmp_logits = tmp_model(best_inputs)
                    tmp_logits = tmp_model(images)
                    # loss = F.cross_entropy(tmp_logits, targets)
                    loss = torch.nn.functional.cross_entropy(tmp_logits, labels)
                    # backward
                    loss.backward()
                    mix_weight = mix_weight - self.mu * pow(self.wdc, cur_ep) * mix_weight.grad.sign()
                    eta = torch.clamp(mix_weight - ori_weight, min=-1, max=1)
                    mix_weight = torch.clamp(ori_weight + eta, min=0.0, max=1.0).detach_()
                    self.teacher.mdl_w_list = mix_weight

                        # best_loss = loss.item()
                del tmp_model

        self.student.train()
        if self.normalizer:
            best_inputs = self.normalizer(best_inputs, True)
        self.data_pool.add( best_inputs, batch_id = cur_ep, targets=targets, his=self.his)
        dst = self.data_pool.get_dataset(transform=self.transform, labeled=True)
        if self.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dst) if self.distributed else None
        else:
            train_sampler = None
        loader = torch.utils.data.DataLoader(
            dst, batch_size=self.sample_batch_size, shuffle=(train_sampler is None),
            num_workers=4, pin_memory=True, sampler=train_sampler)
        self.data_iter = DataIter(loader)
        del z,targets
        return {'synthetic': best_inputs}


    def sample(self):
        if self.batchonly == True and self.batchused == False:
            self.generator.eval()
            z = torch.randn(size=(self.sample_batch_size, self.nz)).to(self.device)
            images = self.normalizer(self.generator(z))
            return images
        else:
            images, labels = self.data_iter.next()
        return images, labels


    def get_data(self,labeled=True):
        datasets = self.data_pool.get_dataset(transform=self.transform, labeled=labeled)  # 获取程序运行到现在所有的图片
        self.data_loader = torch.utils.data.DataLoader(
            datasets, batch_size=self.sample_batch_size, shuffle=True,
            num_workers=4, pin_memory=True, )
        return self.data_loader


class DENSESynthesizer(BaseSynthesis):
    def __init__(self, teacher, mdl_list, student, generator, nz, num_classes, img_size, save_dir, iterations=1,
                 lr_g=1e-3, synthesis_batch_size=128, sample_batch_size=128,
                 adv=0, bn=0, oh=0, act=0, balance=0, criterion=None,transform=None,
                 normalizer=None,
                 # TODO: FP16 and distributed training
                 autocast=None, use_fp16=False, distributed=False, his=True, batchonly=False, batchused=False, device='cuda:0'):
        super(DENSESynthesizer, self).__init__(teacher, student)
        self.mdl_list = mdl_list
        
        assert len(img_size) == 3, "image size should be a 3-dimension tuple"
        self.img_size = img_size
        self.iterations = iterations
        self.save_dir = save_dir
        self.transform = transform

        self.nz = nz
        self.num_classes = num_classes
        if criterion is None:
            criterion = kldiv
        self.criterion = criterion
        self.normalizer = normalizer
        self.synthesis_batch_size = synthesis_batch_size
        self.sample_batch_size = sample_batch_size

        self.data_pool = ImagePool(root=self.save_dir)
        self.data_iter = None
        # scaling factors
        self.lr_g = lr_g
        self.adv = adv
        self.bn = bn
        self.oh = oh
        self.balance = balance
        self.act = act

        # generator
        self.generator = generator.to(device).train()
        self.distributed = distributed
        self.use_fp16 = use_fp16
        self.autocast = autocast  # for FP16
        self.hooks = []

        # hooks for deepinversion regularization

        for m_list in self.mdl_list:
            for m in m_list.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    self.hooks.append(DeepInversionHook(m))

        self.his = his
        self.batchonly = batchonly
        self.batchused = batchused

        self.device = device

    def synthesize(self, cur_ep=None):
        ###########
        # 设置eval模式
        ###########
        self.student.eval()
        self.generator.train()
        self.teacher.eval()
        for m in self.mdl_list:
            m.eval()

        if self.bn == 0:
            self.hooks = []
        best_cost = 1e6
        z = torch.randn(size=(self.synthesis_batch_size, self.nz)).to(self.device)
        z.requires_grad = True
        targets = torch.randint(low=0, high=self.num_classes, size=(self.synthesis_batch_size,))
        targets = targets.sort()[0]
        targets = targets.to(self.device)
        reset_model(self.generator)

        optimizer = torch.optim.Adam([{'params': self.generator.parameters()}, {'params': [z]}], self.lr_g,
                                     betas=[0.5, 0.999])
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( optimizer, T_max=self.iterations )

        for it in range(self.iterations):


            optimizer.zero_grad()

            inputs = self.generator(z)

            inputs = self.normalizer(inputs)
            t_out = self.teacher(inputs)
            if len(self.hooks) == 0 or self.bn == 0:
                loss_bn = torch.tensor(0).to(self.device)
            else:
                loss_bn = sum([h.r_feature for h in self.hooks]) / len(self.mdl_list)
            loss_oh = torch.nn.functional.cross_entropy(t_out, targets)
            s_out = self.student(inputs)
            mask = (s_out.max(1)[1] != t_out.max(1)[1]).float()
            loss_adv = -(kldiv(s_out, t_out, T = 3,reduction='none').sum(
                1) * mask).mean()  # decision adversarial distillation
            loss = self.bn * loss_bn + self.oh * loss_oh + self.adv * loss_adv


            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=self.generator.parameters(), max_norm=10)
            for m in self.mdl_list:
                m.zero_grad()
            optimizer.step()
            # scheduler.step()

            # torch.cuda.empty_cache()

            if best_cost > loss.item():
                best_cost = loss.item()
                best_inputs = inputs.data

        self.student.train()
        if self.normalizer:
            best_inputs = self.normalizer(best_inputs, True)
        self.data_pool.add( best_inputs, batch_id = cur_ep, targets=targets, his=self.his)
        dst = self.data_pool.get_dataset(transform=self.transform, labeled=True)
        if self.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dst) if self.distributed else None
        else:
            train_sampler = None
        loader = torch.utils.data.DataLoader(
            dst, batch_size=self.sample_batch_size, shuffle=(train_sampler is None),
            num_workers=4, pin_memory=True, sampler=train_sampler)
        self.data_iter = DataIter(loader)
        del z,targets
        return {'synthetic': best_inputs}


    def sample(self):
        if self.batchonly == True and self.batchused == False:
            self.generator.eval()
            z = torch.randn(size=(self.sample_batch_size, self.nz)).to(self.device)
            images = self.normalizer(self.generator(z))
            return images
        else:
            images, labels = self.data_iter.next()
        return images, labels

    def get_data(self,labeled=True):
        datasets = self.data_pool.get_dataset(transform=self.transform, labeled=labeled)  # 获取程序运行到现在所有的图片
        self.data_loader = torch.utils.data.DataLoader(
            datasets, batch_size=self.sample_batch_size, shuffle=True,
            num_workers=4, pin_memory=True, )
        return self.data_loader


class IntactOFLSynthesizer(BaseSynthesis):
    def __init__(self, teacher, mdl_list, student, generator, nz, num_classes, img_size, save_dir, iterations=1,
                 lr_g=1e-3, synthesis_batch_size=128, sample_batch_size=128,
                 adv=0, bn=0, oh=0, act=0, balance=0, criterion=None,transform=None,
                 normalizer=None,
                 # TODO: FP16 and distributed training
                 autocast=None, use_fp16=False, distributed=False, his=True, batchonly=False, batchused=False, device='cuda:0'):    

        super(IntactOFLSynthesizer, self).__init__(teacher, student)
        self.mdl_list = mdl_list
        
        assert len(img_size) == 3, "image size should be a 3-dimension tuple"
        self.img_size = img_size
        self.iterations = iterations
        self.save_dir = save_dir
        self.transform = transform

        self.nz = nz
        self.num_classes = num_classes
        if criterion is None:
            criterion = kldiv
        self.criterion = criterion
        self.normalizer = normalizer
        self.synthesis_batch_size = synthesis_batch_size
        self.sample_batch_size = sample_batch_size

        self.data_pool = ImagePool(root=self.save_dir)
        self.data_iter = None
        # scaling factors
        self.lr_g = lr_g
        self.adv = adv
        self.bn = bn
        self.oh = oh
        self.balance = balance
        self.act = act

        # generator
        self.generator = generator.to(device).train()
        self.distributed = distributed
        self.use_fp16 = use_fp16
        self.autocast = autocast  # for FP16
        self.hooks = []

        # hooks for deepinversion regularization

        for m_list in self.mdl_list:
            for m in m_list.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    self.hooks.append(DeepInversionHook(m))

        self.his = his
        self.batchonly = batchonly
        self.batchused = batchused

        self.device = device


    def synthesize(self, cur_ep=None):
        ###########
        # 设置eval模式
        ###########
        # self.student.eval()
        self.generator.train()
        self.teacher.eval()
        for m in self.mdl_list:
            m.eval()

        if self.bn == 0:
            self.hooks = []
        best_cost = 1e6
        z = torch.randn(size=(self.synthesis_batch_size, self.nz)).to(self.device)
        z.requires_grad = True
        targets = torch.randint(low=0, high=self.num_classes, size=(self.synthesis_batch_size,))
        targets = targets.sort()[0]
        targets = targets.to(self.device)
        reset_model(self.generator)

        optimizer = torch.optim.Adam([{'params': self.generator.parameters()}, {'params': [z]}], self.lr_g,
                                     betas=[0.5, 0.999])
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( optimizer, T_max=self.iterations )

        for it in range(self.iterations):

            optimizer.zero_grad()

            inputs = self.generator(z)

            inputs = self.normalizer(inputs)
            t_out = self.teacher(inputs)
            if len(self.hooks) == 0 or self.bn == 0:
                loss_bn = torch.tensor(0).to(self.device)
            else:
                loss_bn = sum([h.r_feature for h in self.hooks]) / len(self.mdl_list)
            loss_oh = torch.nn.functional.cross_entropy(t_out, targets)
            s_out = self.student(inputs)
            mask = (s_out.max(1)[1] != t_out.max(1)[1]).float()
            loss_adv = -(kldiv(s_out, t_out, T = 3,reduction='none').sum(
                1) * mask).mean()  # decision adversarial distillation
            loss = self.bn * loss_bn + self.oh * loss_oh + self.adv * loss_adv


            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=self.generator.parameters(), max_norm=10)
            for m in self.mdl_list:
                m.zero_grad()
            optimizer.step()
            # scheduler.step()

            # torch.cuda.empty_cache()

            if best_cost > loss.item():
                best_cost = loss.item()
                best_inputs = inputs.data

        self.student.train()
        if self.normalizer:
            best_inputs = self.normalizer(best_inputs, True)
        self.data_pool.add( best_inputs, batch_id = cur_ep, targets=targets, his=self.his)
        dst = self.data_pool.get_dataset(transform=self.transform, labeled=True)
        if self.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dst) if self.distributed else None
        else:
            train_sampler = None
        loader = torch.utils.data.DataLoader(
            dst, batch_size=self.sample_batch_size, shuffle=(train_sampler is None),
            num_workers=4, pin_memory=True, sampler=train_sampler)
        self.data_iter = DataIter(loader)
        del z,targets
        return {'synthetic': best_inputs}        

    def sample(self):
        if self.batchonly == True and self.batchused == False:
            self.generator.eval()
            z = torch.randn(size=(self.sample_batch_size, self.nz)).to(self.device)
            images = self.normalizer(self.generator(z))
            return images
        else:
            images, labels = self.data_iter.next()
        return images, labels

    def get_data(self,labeled=True):
        datasets = self.data_pool.get_dataset(transform=self.transform, labeled=labeled)  # 获取程序运行到现在所有的图片
        self.data_loader = torch.utils.data.DataLoader(
            datasets, batch_size=self.sample_batch_size, shuffle=True,
            num_workers=4, pin_memory=True, )
        return self.data_loader



def reset_model(model):
    for m in model.modules():
        if isinstance(m, (torch.nn.ConvTranspose2d, torch.nn.Linear, torch.nn.Conv2d)):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        if isinstance(m, (torch.nn.BatchNorm2d)):
            torch.nn.init.normal_(m.weight, 1.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)

# Ensemble models
class WEnsemble(torch.nn.Module):
    def __init__(self, model_list, mdl_w_list):
        super(WEnsemble, self).__init__()
        self.models = model_list
        self.mdl_w_list = mdl_w_list

    def forward(self, x, get_feature=False):
        logits_total = 0
        feat_total = 0
        for i in range(len(self.models)):
            if get_feature:
                logits, feat = self.models[i](x, get_feature=get_feature)
                feat_total += self.mdl_w_list[i] * feat
            else:
                logits = self.models[i](x)
            logits_total += self.mdl_w_list[i] * logits
        logits_e = logits_total / torch.sum(self.mdl_w_list)
        if get_feature:
            feat_e = feat_total / torch.sum(self.mdl_w_list)
            return logits_e, feat_e
        return logits_e

    def feat_forward(self, x):
        out_total = 0
        for i in range(len(self.models)):
            out = self.models[i].feat_forward(x)
            out_total += out
        return out_total / len(self.models)


class LinearGatingNetwork(torch.nn.Module):
    def __init__(self, num_experts, topk, channel, image_size):
        super(LinearGatingNetwork, self).__init__()
        self.topk = topk
        self.channel = channel
        self.num_experts = num_experts
        self.image_size = image_size
        self.linear = torch.nn.Linear(channel * image_size * image_size, num_experts)
        self.instance_norm = torch.nn.InstanceNorm1d(num_experts)

    def forward(self, x):
        x = x.view(-1, self.channel * self.image_size * self.image_size) # (batchsize, imagesize)
        x = self.linear(x)

        topk_logits, indices = x.topk(self.topk, dim=-1)
        zeros = torch.full_like(x, 0.0)
        sparse_logits = zeros.scatter(-1, indices, topk_logits)

        return sparse_logits


class MoEEnsemble(torch.nn.Module):
    def __init__(self, model_list, gating_arch, topk, channel, image_size):
        super(MoEEnsemble, self).__init__()
        self.models = model_list
        self.topk = topk
        self.gating_network = self.init_gatingnetwork(gating_arch, len(model_list), topk, channel, image_size)
    
    def init_gatingnetwork(self, gating_arch, num_experts, topk, channel, image_size):
        if gating_arch == 'linear':
            self.gating_network = LinearGatingNetwork(num_experts, topk, channel, image_size)

        return self.gating_network
    
    
    def forward(self, x, get_feature=False, get_weight=False):
        # get weight_list from the gating network

        weight_list = self.gating_network(x) # weight_list shape (batchsize, num_experts)

        if get_feature:
            feature_total = []
            logits_total = []
            for model in self.models:
                feature, logit = model(x, get_feature=True)
                feature_total.append(feature)
                logits_total.append(logit)
            
            # feature_total shape (num_experts, batchsize, featuresize)
            feature_total = torch.stack(feature_total, dim=0)
            # logit (num_experts, batchsize, num_classes)
            logits_total = torch.stack(logits_total, dim=0)

            # weight_list (batchsize, num_experts) -> (batchsize, num_experts, 1)
            weight_list_ = weight_list.unsqueeze(-1)

            # final output (batchsize, featuresize or num_classes)
            feature_total = torch.mean(feature_total.permute(1, 0, 2) * weight_list_, dim=1)
            logits_total = torch.mean(logits_total.permute(1, 0, 2) * weight_list_, dim=1)
       
            if get_weight:
                return feature_total, logits_total, weight_list
            else:
                return feature_total, logits_total
        else:
            logits_total = []
            for model in self.models:
                logit = model(x)
                logits_total.append(logit)
            
            logits_total = torch.stack(logits_total, dim=0)
            weight_list_ = weight_list.unsqueeze(-1)

            logits_total = torch.mean(logits_total.permute(1, 0, 2) * weight_list_, dim=1)
            
            if get_weight:
                return logits_total, weight_list
            else:
                return logits_total    
    
