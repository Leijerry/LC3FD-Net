import numpy as np
import os
import torch

CLASSES = [
    "label",
    'distances',
]


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(n_class * label_true[mask].astype(int) +
                       label_pred[mask],
                       minlength=n_class**2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(label_trues, label_preds):
    """Returns accuracy score evaluation result.

      - overall accuracy
      - mean accuracy
      - mean IU
    """
    #print("label_trues", label_trues.shape)
    # ("label_preds",label_preds.shape)
    sum_acc = np.sum(np.absolute(label_preds - label_trues))
    mean_acc = np.mean(np.absolute(label_preds - label_trues))
    #get the coordinates in gt
    distance_map = label_trues[0]
    index_result = np.where(distance_map == 1)
    listOfCoords = list(zip(index_result[0], index_result[1]))
    sum_center_off = 0
    iter = 0
    for centers in listOfCoords:
        sum_center_off = sum_center_off + (label_trues[0][centers[0]][centers[1]] - label_preds[0][centers[0]][centers[1]])
        iter += 1
    mean_center_off = sum_center_off / np.float64(iter)
    return sum_acc, mean_acc, np.absolute(sum_center_off), np.absolute(mean_center_off)


def get_log_dir(log_dir, cfg):
    import os
    import yaml
    import os.path as osp
    log_dir = log_dir
    if not osp.exists(log_dir):
        os.makedirs(log_dir)
    with open(osp.join(log_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)
    return log_dir


def get_config():
    return {
        1:
        dict(
            max_iteration=700000,
            lr=0.001,
            momentum=0.99,
            betas = (0.9,0.999),
            weight_decay=0,
            interval_validate=188,
        )
    }


def get_cuda(cuda, _id):
    import torch
    if not cuda:
        return torch.device('cpu')
    else:
        return torch.device('cuda:{}'.format(_id))


def imshow_label(label_show, alpha=1.0):
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    cmap = plt.cm.jet
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmaplist[0] = (0.0, 0.0, 0.0, 1.0)
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    # define the bins and normalize
    bounds = np.arange(0, len(CLASSES))
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    plt.imshow(label_show, cmap=cmap, norm=norm, alpha=alpha)
    cbar = plt.colorbar(ticks=bounds)
    cbar.ax.set_yticklabels(CLASSES)


def fileimg2model(img_file, transform):
    import PIL
    img = PIL.Image.open(img_file).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    return transform(img, img)[0]


def run_fromfile(model, img_file, cuda, transform, val=False):
    import matplotlib.pyplot as plt
    import torch
    if not val:
        img_torch = torch.unsqueeze(fileimg2model(img_file, transform), 0)
    else:
        img_torch = img_file
    img_torch = img_torch.to(cuda)
    model.eval()
    with torch.no_grad():
        if not val:
            img_org = plt.imread(img_file)
        else:
            img_org = transform(img_file[0], img_file[0])[0]

        score = model(img_torch)
        lbl_pred = score.data.max(1)[1].cpu().numpy()

        plt.imshow(img_org, alpha=.9)
        imshow_label(lbl_pred[0], alpha=0.5)
        plt.show()

def load_checkpoint(model, optimizer, filename='checkpoint.pth.tar'):
    start_epoch = 0
    loss = []
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optim_state_dict'])
        loss = checkpoint['loss']
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, start_epoch, optimizer, loss

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
