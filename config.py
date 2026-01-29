import configparser

class Config(object):
    def __init__(self, config_file: str = "config.ini"):
        conf = configparser.ConfigParser()
        conf.read(config_file, encoding="utf-8")

        train = conf["Train"]
        self.State = train.get("State", "Train")
        self.batch_size = train.getint("batch_size", "8")
        self.test_rate = train.getfloat("test_rate")
        self.GPU = train.get("GPU", "0")
        self.epochs = train.getint("epochs")
        self.save_every = train.getint("save_every")
        self.lr = train.getfloat("lr")
        self.seed = train.getint("seed")

        self.emb_dim = train.getint("emb_dim")  # 每个 patch 的embedding维度 D
        self.pq_M = train.getint("pq_M")
        self.pq_K = train.getint("pq_K")
        self.pq_softmax_temp = train.getfloat("pq_softmax_temp")
        self.pq_quant_method = train.get("pq_quant_method")  # ;{"softmax","gumbel","hard"}等
        self.pq_init_neg_curvs = train.getfloat("pq_init_neg_curvs")
        self.pq_clip_r = train.getfloat("pq_clip_r")
        self.pq_use_alpha = train.getboolean("pq_use_alpha")

        self.Spatial_Area_x1 = train.getint("Spatial_Area_x1")
        self.Spatial_Area_x2 = train.getint("Spatial_Area_x2")
        self.Time_Area_n = train.getint("Time_Area_n")

        self.margin_M = train.getfloat("margin_M")
        self.w_min = train.getfloat("w_min")
        self.w_max = train.getfloat("w_max")

        self.mask_ratio = train.getfloat("mask_ratio")
        self.recon_weight = train.getfloat("recon_weight")
        self.cluster_weight = train.getfloat("cluster_weight")

        data = conf["Data"]
        self.sfrq = data.getint("sfrq")
        self.time_window_length = data.getint("time_window_length")
        self.time_window_overlap = data.getfloat("time_window_overlap")
        self.patch_length = data.getfloat("patch_length")
        self.target_domain = data.get("target_domain")