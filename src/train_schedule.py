train_schedule_1 = {
	
	'dataset': {
		'name': 'NotreDame',
	},
	
	'model': {
		'name': 'StackNet',
	},
	
	'epoch': {
		'start': 0,
		'end': 100,
	},
	
	'lr': {
		'values': [1e-4, 5e-5, 1e-5],
		'ckpts': [40,60],
	},

	'log':{
		'pretrained': False,
		'ckpt': None,
		'TB_path': '../experiments/StackNet/',
		'weights_path': '../experiments/StackNet/checkpoint.pth.tar',
	},
}