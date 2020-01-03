class ModelConfig:
	def __init__(self):
		self.bs = 32
		self.trn_workers = 8
		self.val_workers = 4
		self.tst_workers = 4
		self.sample_data_path = '/media/saqib/ni/Projects/Microsoft/AI_Challenge_18/data/data_sample.tsv'
		self.data_path = '/media/saqib/ni/Projects/Microsoft/AI_Challenge_18/data/data.tsv'
		self.data_dir = '/media/saqib/ni/Projects/Microsoft/AI_Challenge_18/data/'
		self.sample_trn_data_path = '/media/saqib/ni/Projects/Microsoft/AI_Challenge_18/data/data_sample_trn.tsv'
		self.sample_val_data_path = '/media/saqib/ni/Projects/Microsoft/AI_Challenge_18/data/data_sample_val.tsv'
		self.trn_data_path = '/media/saqib/ni/Projects/Microsoft/AI_Challenge_18/data/data_trn.tsv'
		self.val_data_path = '/media/saqib/ni/Projects/Microsoft/AI_Challenge_18/data/data_val.tsv'
		self.trn_50_20_data_path = '/media/saqib/ni/Projects/Microsoft/AI_Challenge_18/data/data_0.5_20_trn.tsv'
		self.val_50_20_data_path = '/media/saqib/ni/Projects/Microsoft/AI_Challenge_18/data/data_0.5_20_val.tsv'
		self.eval_data_path_01 = '/media/saqib/ni/Projects/Microsoft/AI_Challenge_18/data/eval1_unlabelled.tsv'
		self.random_state = 42


class ULMFiTConfig(ModelConfig):

	def __init__(self):
		super(ULMFiTConfig, self).__init__()


class WideResnetConfig(ModelConfig):

	def __init__(self):
		super().__init__()
		self.n_grps = 3
		self.n_blocks = 3
		self.widening = 5
		self.emb_pretrained = '/media/saqib/ni/Projects/Microsoft/AI_Challenge_18/data/train_lm_data/itovecs.npy'
		self.vocab_size = 60002
		self.emb_dim = 300


class WideResnetParallelConfig(WideResnetConfig):

	def __init__(self):
		super().__init__()
		self.bs = 48
		self.n_grps = 2
		self.n_blocks = 2