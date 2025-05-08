import torch
import torch.nn as nn
import os
from tqdm import tqdm
from opt import opt
from utils.metrics import evaluate
import datasets
from torch.utils.data import DataLoader
from utils.comm import generate_model
from utils.metrics import Metrics


def test(model, test_data_dir):
    test_data_name = test_data_dir.split("/")[1]

    print('Loading data......')
    test_data = getattr(datasets, opt.dataset)(opt.root, test_data_dir, mode='test')
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=opt.num_workers)
    total_batch = int(len(test_data) / 1)

    model.eval()

    # metrics_logger initialization
    metrics = Metrics(['recall', 'specificity', 'precision', 'F1', 'F2', 'ACC_overall', 'IoU_poly', 'IoU_bg', 'IoU_mean'])

    print('Start testing')
    with torch.no_grad():
        bar = tqdm(enumerate(test_dataloader), total=total_batch)
        for i, data in bar:
            img, gt, name = data['image'], data['label'], data['name']

            if opt.use_gpu:
                img = img.cuda()
                gt = gt.cuda()

            output = model(img)
            _recall, _specificity, _precision, _F1, _F2, _ACC_overall, _IoU_poly, _IoU_bg, _IoU_mean = evaluate(output, gt, name, test_data_name)
           

            metrics.update(recall= _recall, specificity= _specificity, precision= _precision, F1= _F1, F2= _F2, ACC_overall= _ACC_overall, IoU_poly= _IoU_poly, IoU_bg= _IoU_bg, IoU_mean= _IoU_mean)

    metrics_result = metrics.mean(total_batch)

    results = open('./checkpoints/exp' + str(opt.expID) + "/testResults.txt", "a+")

    print("\n%s Test Result:" % test_data_name, file=results)
    print('recall: %.4f, specificity: %.4f, precision: %.4f, F1: %.4f, F2: %.4f, ACC_overall: %.4f, IoU_poly: %.4f, IoU_bg: %.4f, IoU_mean: %.4f' % (metrics_result['recall'], metrics_result['specificity'], metrics_result['precision'], metrics_result['F1'], metrics_result['F2'], metrics_result['ACC_overall'], metrics_result['IoU_poly'], metrics_result['IoU_bg'], metrics_result['IoU_mean']), file=results)
    print("\n%s Test Result:" % test_data_name)
    print('recall: %.4f, specificity: %.4f, precision: %.4f, F1: %.4f, F2: %.4f, ACC_overall: %.4f, IoU_poly: %.4f, IoU_bg: %.4f, IoU_mean: %.4f' % (metrics_result['recall'], metrics_result['specificity'], metrics_result['precision'], metrics_result['F1'], metrics_result['F2'], metrics_result['ACC_overall'], metrics_result['IoU_poly'], metrics_result['IoU_bg'], metrics_result['IoU_mean']))

    results.close()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

    print('Loading model......')
    model = generate_model(opt)

    #test_data_list = ["Kvasir/test", "CVC-ClinicDB/test", "CVC-ColonDB", "ETIS-LaribPolypDB"]

    if opt.mode == 'test':
        print('--- PolypSeg Test---')
        test(model, opt.test_data_dir)

        # you could also utilize the following loop operation
        # to directly evaluate the performance on all testsets

        # for test_data_dir in test_data_list:
        #    test(model, "data/" + test_data_dir)

    print('Done')
