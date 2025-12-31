import torch

class GDType:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getitem__(self, key):
        return self.__dict__[key]

    def __repr__(self):
        return str(self.__dict__)

    def __contains__(self, key):
        return key in self.__dict__

    def to(self, device):
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                self.__dict__[key] = value.to(device)
        return self

    def cpu(self):
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                self.__dict__[key] = value.cpu()
        return self

    def cuda(self, idx=None):
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                if idx:
                    self.__dict__[key] = value.cuda(idx)
                else:
                    self.__dict__[key] = value.cuda()
        return self

    @property
    def shape(self):
        shapes = {}
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                shapes[key] = value.shape
        return shapes

    def keys(self):
        return self.__dict__.keys()


class GD3DTarget(GDType):
    def __init__(
        self,
        labels=None,
        size=None,
        positive_map=None,
        scene_id=None,
        data_source=None,
        prompt_type=None,
        loss_branch=None,
        area=None,
        orig_size=None,
        iscrowd=0,
        masks=None,
        **kwargs,
    ):

        super().__init__(
            labels=labels,
            size=size,
            positive_map=positive_map,
            scene_id=scene_id,
            data_source=data_source,
            loss_branch=loss_branch,
            prompt_type=prompt_type,
            area=area,
            orig_size=orig_size,
            iscrowd=iscrowd,
            masks=masks,
            **kwargs
        )

    def to(self, device):
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                self.__dict__[key] = value.to(device)
            elif isinstance(value, list):
                for i in range(len(value)):
                    if isinstance(value[i], torch.Tensor):
                        self.__dict__[key][i] = value[i].to(device)
            elif isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, torch.Tensor):
                        self.__dict__[key][k] = v.to(device)
        return self
