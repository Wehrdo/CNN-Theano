from multiprocessing import Pool, Manager, Queue
from time import sleep
import json, csv
import cifar_utils
import traceback

if __name__ == '__main__':
    # Load dataset
    with open('comp_config.json') as data_file:
        comp_config = json.load(data_file)
    train_x, train_y, test_x, test_y = cifar_utils.load_data(comp_config['dataset'])

def run_tests():
    with Manager() as m:
        gpus = m.Queue()
        gpus.put(0)
        gpus.put(1)
        with Pool(2) as p:
            p.map(test_vals, gen_params(gpus))


def test_vals(args):
    params, gpu_q = args
    k1_depth, k2_depth, fc_depth, k1_size, k2_size, dropout, alpha = params
    gpu_id = gpu_q.get()
    batch_size = 100
    epochs = 25
    file_name = '-'.join(map(str, params))
    print("Trying " + file_name + f" from GPU {gpu_id}")
    import theano.sandbox.cuda
    theano.sandbox.cuda.use('gpu' + str(gpu_id))
    import conv
    from conv import Layer
    try:
        layers = [Layer((3,32,32), 0),
                  Layer((k1_depth,16,16), k1_size, pool=2, mode='half'),
                  Layer((k2_depth,8,8), k2_size, pool=2, mode='half'),
                  Layer((fc_depth,1,1), 8), Layer((10,1,1), 1)]
        mlp = conv.MLP(layers, dropout)
        losses, train_accuracies, test_accuracies = mlp.train(train_x, train_y, test_x, test_y, alpha, batch_size, epochs)
        with open(f'logs/{file_name}.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_accuracy', 'test_accuracy'])
            for epoch, accuracy in enumerate(zip(train_accuracies, test_accuracies)):
                writer.writerow([epoch, accuracy[0], accuracy[1]])
    except:
        print("Failed on " + file_name)
        with open(f'logs/{file_name}.txt', 'w') as f:
            f.write(traceback.print_exc())
    finally:
        gpu_q.put(gpu_id)
    gpu_q.put(gpu_id)


def gen_params(gpu_q):
    for k1_depth in [128]:
        for k2_depth in [64, 128]:
            for fc_depth in [64, 128]:
                for k1_size in [5]:
                    for k2_size in [3, 5]:
                        for dropout in [0.7, 0.85]:
                            for alpha in [0.001, 0.01]:
                                yield (k1_depth, k2_depth, fc_depth, k1_size, k2_size, dropout, alpha), gpu_q

if __name__ == '__main__':
    run_tests()