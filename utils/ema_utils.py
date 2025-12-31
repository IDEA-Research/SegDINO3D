import os
import torch
import torch.distributed as dist

class ModelEma():
    def __init__(self, model, decay=0.9997, seed=''):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.register()
        self.is_gathered = False
        self.seed = seed

    def register(self):
        names = []
        for name, _ in self.model.named_parameters():
            names.append(name)
        for name, _ in self.model.named_buffers():
            names.append(name)

        names = sorted(names)
        tot = len(names)
        bs = tot // self.world_size + 1
        names = names[self.rank * bs: (self.rank + 1) * bs]
        self.names = names

        for name, param in self.model.named_parameters():
            if name in names:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def gather(self):
        print('EMA gather')
        # 若使用dist.all_gather_object会额外占用显存，原因不明，弃用
        if self.rank == 0:
            os.makedirs('.ema_cache/.ema_cache_%s'%self.seed, exist_ok=True)
        dist.barrier()
        torch.save(self.shadow, '.ema_cache/.ema_cache_%s/ema_%d.pth'%(self.seed, self.rank))
        dist.barrier()
        self.is_gathered = True

    def get_shadow(self):
        assert self.is_gathered
        ckpt = {}
        for i in range(self.world_size):
            ckpt.update(torch.load('.ema_cache/.ema_cache_%s/ema_%d.pth'%(self.seed, i), map_location='cpu'))
        return ckpt

    def apply_shadow(self):
        assert self.is_gathered
        print('EMA apply shadow')
        ckpt = {}
        for i in range(self.world_size):
            ckpt.update(torch.load('.ema_cache/.ema_cache_%s/ema_%d.pth'%(self.seed, i), map_location='cpu'))

        for name, param in self.model.named_parameters():
            self.backup[name] = param.data
            param.data = ckpt[name].to(device=param.data.device)

    def restore(self):
        print('EMA restore')
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
                del self.backup[name]

        self.backup = {}
        self.is_gathered = False
        if self.rank == 0:
            os.system('rm -rf .ema_cache/.ema_cache_%s'%self.seed)
        dist.barrier()
