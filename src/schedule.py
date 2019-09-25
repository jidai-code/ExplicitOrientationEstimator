train_schedule_1 = {
	
	'dataset': {
		'name': 'NotreDame',
	},
	
	'model': {
		'name': 'STN_v1',
	},
	
	'epoch': {
		'start': 0,
		'end': 20,
	},
	
	'lr': {
		'values': [1e-3, 1e-4],
		'ckpts': [5],
	},

	'log':{
		'pretrained': False,
		'ckpt': None,
		'TB_path': '../experiments/STN_v2/',
		'weights_path': '../experiments/STN_v2/checkpoint.pth.tar',
	},
}