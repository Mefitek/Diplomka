import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser

def compute_confusion_matrix(preds, targets):
    cm = torch.zeros(2, 2, dtype=torch.int64)
    for t, p in zip(targets, preds):
        cm[t.long(), p.long()] += 1
    return cm

def print_confusion_matrix(cm):
    tp = cm[1, 1].item()
    fp = cm[0, 1].item()
    fn = cm[1, 0].item()
    tn = cm[0, 0].item()

    print("\n------------------------------")
    print("Confusion Matrix")
    print("               Actual")
    print("            [1]       [0]      ")
    print(f"Pred [1]  {tp:>5}     {fp:>5}")
    print(f"Pred [0]  {fn:>5}     {tn:>5}")

    total = tp+fp+fn+tn
    tp = tp/total*100
    fp = fp/total*100
    fn = fn/total*100
    tn = tn/total*100

    print("------------------------------")
    print("Confusion Matrix (percentage)")
    print("               Actual")
    print("            [1]       [0]      ")
    print(f"Pred [1]  {tp:>6.2f}   {fp:>6.2f}")
    print(f"Pred [0]  {fn:>6.2f}   {tn:>6.2f}")
    print("------------------------------\n")

def print_f1_score(cm):
    """
    Computes and prints F1 score from a 2x2 confusion matrix.
    Expects cm to be in the format:
        [[TP, FP],
         [FN, TN]]
    """
    tp = cm[0, 0].item()
    fp = cm[0, 1].item()
    fn = cm[1, 0].item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    print(f"F1 Score: {f1:.4f}")

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
            preds = (output > 0.5).int().view(-1)  # binární prediction with thresh
            all_preds.append(preds.cpu())
            all_targets.append(target.view(-1).int().cpu())

            # Loss a metriky
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
    cm = compute_confusion_matrix(all_preds, all_targets)
    print_confusion_matrix(cm)
    print_f1_score(cm)

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
