import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser

def compute_confusion_matrix(preds, targets, num_classes):
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for t, p in zip(targets, preds):
        cm[t, p] += 1
    return cm

def print_test_data(cm):
    num_classes = cm.size(0)
    total = cm.sum().item()
    correct = cm.diag().sum().item()
    
    print("\n------------------------------")
    print("Confusion Matrix (absolute values)")
    print("               Actual")
    header = "         " + "".join([f"[{i}]    " for i in range(num_classes)])
    print(header)
    for i in range(num_classes):
        row = f"Pred [{i}] " + "".join([f"{cm[i, j].item():>6}" for j in range(num_classes)])
        print(row)

    print("\n------------------------------")
    print("Confusion Matrix (percentage)")
    print("               Actual")
    print(header)
    for i in range(num_classes):
        row = f"Pred [{i}] " + "".join([f"{(cm[i, j].item() / total * 100):>6.2f}" for j in range(num_classes)])
        print(row)

    accuracy = correct / total * 100
    print(f"\nAverage Accuracy: {accuracy:.2f} %")

    print("\nF1-scores by class:")
    for i in range(num_classes):
        tp = cm[i, i].item()
        fp = cm[:, i].sum().item() - tp
        fn = cm[i, :].sum().item() - tp
        denom = 2 * tp + fp + fn
        f1 = (2 * tp) / denom if denom != 0 else 0.0
        print(f"  Class {i}: {f1:.2f}")
    
    print("------------------------------\n")


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

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)

            #
            # save sample images, or do something with output here
                # Compute predictions for confusion matrix
            preds = torch.argmax(output, dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(target.cpu())
            #

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size

    n_samples = len(data_loader.dataset)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })

     # Confusion matrix
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    num_classes = config['arch']['args']['num_classes']
    cm = compute_confusion_matrix(all_preds, all_targets, num_classes)
    print("\nConfusion matrix:")
    #print(cm)
    print_test_data(cm)

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
