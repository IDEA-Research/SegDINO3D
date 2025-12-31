from mmengine import Registry, build_from_cfg

BACKBONES = Registry('backbone')
NECKS = Registry('neck')
POS_EMBEDDINGS = Registry('position_embedding')
FUSERS = Registry('fuser')
ENCODERS = Registry('encoder')
DECODERS = Registry('decoder')
ARCHITECTURES = Registry('architecture')
TEXT_ENCODERS = Registry('text_encoder')
HEADS = Registry('head')
PREPARERS = Registry('preparer')


# training only
DATASETS = Registry('dataset')
TRANSFORMS = Registry('transform')
LOSSES = Registry('loss')
MATCHERS = Registry('matcher')

# eval only
EVALUATORS = Registry('evaluator')

def build_backbone(cfg):
    """Build encoder."""
    return build_from_cfg(cfg, BACKBONES)

def build_neck(cfg):
    """Build neck."""
    return build_from_cfg(cfg, NECKS)

def build_position_embedding(cfg):
    """Build position embedding."""
    return build_from_cfg(cfg, POS_EMBEDDINGS)

def build_fuser(cfg):
    """Build fuser."""
    return build_from_cfg(cfg, FUSERS)

def build_encoder(cfg):
    """Build encoder."""
    return build_from_cfg(cfg, ENCODERS)

def build_decoder(cfg):
    """Build decoder."""
    return build_from_cfg(cfg, DECODERS)

def build_architecture(cfg):
    """Build architecture."""
    return build_from_cfg(cfg, ARCHITECTURES)

def build_text_encoder(cfg):
    """Build text encoder"""
    return build_from_cfg(cfg, TEXT_ENCODERS)

def build_head(cfg):
    """Build head"""
    return build_from_cfg(cfg, HEADS)

def build_preparer(cfg):
    """Build preparer."""
    return build_from_cfg(cfg, PREPARERS)

def build_dataset(cfg):
    """Build dataset."""
    return build_from_cfg(cfg, DATASETS)

def build_transform(cfg):
    """Build transform."""
    return build_from_cfg(cfg, TRANSFORMS)

def build_loss(cfg):
    """Build loss."""
    return build_from_cfg(cfg, LOSSES)

def build_matcher(cfg):
    """Build matcher."""
    return build_from_cfg(cfg, MATCHERS)

def build_evaluator(cfg):
    """Build evaluator."""
    return build_from_cfg(cfg, EVALUATORS)

