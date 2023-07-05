import numpy as np
import progressbar


def z_score_normalize(a: np.ndarray):
    return (a - np.average(a)) / np.std(a)


def load_data(path: str, max_size: int = -1, rescale: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Load MNIST data set (csv)
    :param path: the path to the .csv file
    :param max_size: the size of returned data
    :return: an array of features and an array of labels
    """
    with open(path) as f:
        ls = f.readlines()[1:]
    if ls[-1].strip() == '':
        ls.pop()
    n = len(ls[0].split(',')) - 1
    m = len(ls)
    count = 0
    for i, l in enumerate(ls):
        b, *a = [each.strip() for each in l.split(',')]
        if b in ('0', '1'):
            count += 1
    t = min(count, max_size) if max_size >= 0 else count
    x = np.zeros((t, n))
    y = np.zeros((t,))
    j = 0
    print(f"Loading '{path}' ...")
    pbar = progressbar.ProgressBar(
        widgets=[progressbar.Percentage(), ' ', progressbar.Bar('#'), ' ', progressbar.Timer(), ' ',
                 progressbar.ETA(), ' '], maxval=100)
    for i, l in enumerate(ls):
        b, *a = [each.strip() for each in l.split(',')]
        if b in ('0', '1'):
            if j == t:
                break
            else:
                x[j] = np.array(a)
                y[j] = b
                j += 1
                pbar.update(j * 100.0 / t)
    pbar.finish()
    print(f'Loaded data in size {t}.\n')
    return (z_score_normalize(x) if rescale else x), y
