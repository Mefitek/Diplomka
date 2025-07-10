import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from torchmetrics.classification import BinaryAUROC

def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=512,
        shuffle=False,
        training=False,
        test=True,
        num_workers=2
    )

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    sum_error_0 = 0.0
    sum_error_1 = 0.0
    count_0 = 0
    count_1 = 0

    all_errors = []
    all_labels = []

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)


            #
            # save sample images, or do something with output here
            # TODO: Document this part of code
            reconstruction_errors = torch.mean(torch.abs(output - data), dim=[1, 2])  # shape: [batch_size]
            labels = target.view(-1).cpu()
            errors = reconstruction_errors.cpu()
            all_labels.append(labels)
            all_errors.append(errors)
            
            
            # Keep track of total errors by label
            
            errors_label_0 = errors[labels.numpy() == 0]
            errors_label_1 = errors[labels.numpy() == 1]
            sum_error_0 += errors_label_0.sum()
            sum_error_1 += errors_label_1.sum()
            count_0 += len(errors_label_0)
            count_1 += len(errors_label_1)
            

            '''
            for err, label in zip(errors, labels):
                print(f"Reconstruction error: {err:.4f} | Label: {label}")
            '''
            #

            # computing loss, metrics on test set
            loss = loss_fn(output, data) # data instead of target, target is now 'label'
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, data) * batch_size # data instead of target, target is now 'label'

    avg_error_0 = sum_error_0 / count_0
    avg_error_1 = sum_error_1 / count_1
    print("\n\n")
    print(f"Average recontruction error for label 0 =\t{avg_error_0:.4f}")
    print(f"Average recontruction error for label 1 =\t{avg_error_1:.4f}")

    all_errors_tensor = torch.cat(all_errors)
    all_labels_tensor = torch.cat(all_labels).int()
    auc_metric = BinaryAUROC()
    auc = auc_metric(all_errors_tensor, all_labels_tensor)
    print(f"AUC score: {auc:.4f}")
    print("\n\n")

    n_samples = len(data_loader.dataset)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
