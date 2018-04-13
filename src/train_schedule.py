train_schedule_1 = {
	
	'dataset': {
		'name': 'NotreDame',
	},
	
	'model': {
		'name': 'SiameseNet',
	},
	
	'epoch': {
		'start': 0,
		'end': 300,
	},
	
	'lr': {
		'values': [1e-4, 1e-5],
		'ckpts': [100],
	},

	'log':{
		'pretrained': False,
		'ckpt': None,
		'TB_path': '../experiments/siamesenet/',
		'weights_path': '../experiments/siamesenet/',
	},
}