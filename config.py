batch_size = 32
EPOCHS = 50
LEARNING_RATE = 1e-05

TRAINFILE = '/home/pavan/Data/sign_data/train_data.csv'
TRAINDIR = '/home/pavan/Data/sign_data/train/'


TESTFILE = '/home/pavan/Data/sign_data/test_data.csv'
TESTDIR = '/home/pavan/Data/sign_data/test/'


run_name = '50ep_low_lr'

MODELDIR = 'experiments/'
# MODELFNAME = 'trained_model.pt'
MODELFNAME = f'trained_model_{run_name}.pt'
MODELFPATH = MODELDIR+MODELFNAME