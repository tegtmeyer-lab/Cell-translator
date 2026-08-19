"""Cell Translator hyperparameters used in this analyses."""

SEED = 42
DROPOUT = 0.10

BATCH_SIZE = 64
LR = 5e-4
EPOCHS = 300
PATIENCE = 25
MIN_DELTA = 1e-4

THR = 1e-6
INPUT_DROP_PROB = 0.05

LAMBDA_BCE = 1.0
LAMBDA_MAG = 5.0
LAMBDA_FP = 2.0
LAMBDA_SOFT = 0.5

HUBER_BETA = 0.1
ALPHA_TAIL = 1.0
TAIL_CLIP = 6.0

CGAN_BATCH_SIZE = 32
CGAN_EPOCHS = 300
GP_WEIGHT = 10.0
DRIFT_EPS = 1e-3
NCE_W = 0.0
K_PAIR = 5
BANK_SAMPLE = 8192
LAMBDA_SUP = 0.0
LAMBDA_CYCLE = 5.0
LAMBDA_IDENTITY = 0.5
PAIR_W = 1.0
MMD_W = 1.0
COV_W = 0.2
MOM_W = 2.0
N_CRITIC = 5

LR_INITIAL = 1e-4
LR_DECAY_STEPS = 10000
LR_DECAY_RATE = 0.9
ADAM_BETA1 = 0.5
CLIPNORM = 1.0

MNN_EVERY = 10
MNN_POOL = 4000

HISTORY_KEYS = [
    "gen_fwd_loss", "gen_bwd_loss", "disc_tgt_loss", "disc_src_loss",
    "cycle_loss", "identity_loss", "pair_loss", "mmd_loss",
    "cov_loss", "mom_loss", "mom_bwd_loss", "sup_loss", "wasserstein_gap",
]
