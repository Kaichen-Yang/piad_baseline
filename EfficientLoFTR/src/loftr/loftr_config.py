from yacs.config import CfgNode as CN

def lower_config(yacs_cfg):
    if not isinstance(yacs_cfg, CN):
        return yacs_cfg
    return {k.lower(): lower_config(v) for k, v in yacs_cfg.items()}

############## ↓ LoFTR Pipeline ↓ ##############
_CN = CN()
_CN.BACKBONE_TYPE = 'RepVGG'
_CN.ALIGN_CORNER = False
_CN.RESOLUTION = (8, 1)
_CN.FINE_WINDOW_SIZE = 8 # window_size in fine_level, must be even
_CN.FP16 = False
_CN.REPLACE_NAN = False
_CN.EVAL_TIMES = 1

# 1. LoFTR-backbone (local feature CNN) config
_CN.BACKBONE = CN()
_CN.BACKBONE.BLOCK_DIMS = [64, 128, 256] # s1, s2, s3

# 2. LoFTR-coarse module config
_CN.COARSE = CN()
_CN.COARSE.D_MODEL = 256
_CN.COARSE.D_FFN = 256
_CN.COARSE.NHEAD = 8
_CN.COARSE.LAYER_NAMES = ['self', 'cross'] * 4
_CN.COARSE.AGG_SIZE0 = 4
_CN.COARSE.AGG_SIZE1 = 4
_CN.COARSE.NO_FLASH = False
_CN.COARSE.ROPE = True
_CN.COARSE.NPE = None

# 3. Coarse-Matching config
_CN.MATCH_COARSE = CN()
_CN.MATCH_COARSE.THR = 0.2#0.1
_CN.MATCH_COARSE.BORDER_RM = 2
_CN.MATCH_COARSE.DSMAX_TEMPERATURE = 0.1
_CN.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.2 # training tricks: save GPU memory
_CN.MATCH_COARSE.TRAIN_PAD_NUM_GT_MIN = 200 # training tricks: avoid DDP deadlock
_CN.MATCH_COARSE.SPARSE_SPVS = True
_CN.MATCH_COARSE.SKIP_SOFTMAX = False
_CN.MATCH_COARSE.FP16MATMUL = False

# 4. Fine-Matching config
_CN.MATCH_FINE = CN()
_CN.MATCH_FINE.SPARSE_SPVS = True
_CN.MATCH_FINE.LOCAL_REGRESS_TEMPERATURE = 1.0
_CN.MATCH_FINE.LOCAL_REGRESS_SLICEDIM = 8

# # 5. LoFTR Losses
# # coarse-level
# _CN.LOSS = CN()
# _CN.LOSS.COARSE_TYPE = 'focal' # ['focal', 'cross_entropy']
# _CN.LOSS.COARSE_WEIGHT = 1.0
# _CN.LOSS.COARSE_SIGMOID_WEIGHT = 1.0
# _CN.LOSS.LOCAL_WEIGHT = 0.5
# _CN.LOSS.COARSE_OVERLAP_WEIGHT = False
# _CN.LOSS.FINE_OVERLAP_WEIGHT = False
# _CN.LOSS.FINE_OVERLAP_WEIGHT2 = False

# # focal loss (coarse)
# _CN.LOSS.FOCAL_ALPHA = 0.25
# _CN.LOSS.FOCAL_GAMMA = 2.0
# _CN.LOSS.POS_WEIGHT = 1.0
# _CN.LOSS.NEG_WEIGHT = 1.0

# # fine-level
# _CN.LOSS.FINE_TYPE = 'l2_with_std' # ['l2_with_std', 'l2']
# _CN.LOSS.FINE_WEIGHT = 1.0
# _CN.LOSS.FINE_CORRECT_THR = 1.0 # for filtering valid fine-level gts (some gt matches might fall out of the fine-level window)

default_cfg = lower_config(_CN)