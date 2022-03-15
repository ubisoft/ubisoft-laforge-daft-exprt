import numpy as np
from scipy.signal import resample


def cov(x, y):
    x = x - np.mean(x)
    y = y - np.mean(y)
    return np.mean(np.multiply(x,y))

def pcc(x, y):
    s_x = np.std(x)
    s_y = np.std(y)
    return cov(x,y)/(s_y * s_x)
    
def remove_unvoiced(x):
    y = np.zeros(len(x))
    i = 0
    for v in x:
        if v > 0:
            y[i] = v
            i += 1
    return y[:i]

def pcc_on_2_pitch_curve(ref, dut, remove_unvoiced=True):
    """ This function computes the Pearsons correlation ceofficient (PCC) between 2 pitch curves.

    Args:
        ref (numpy array): sequence of an arbitrary length (assumes unvoiced segments are indicated by numbers <= 0)
        dut (numpy array): sequence of an arbitrary length (assumes unvoiced segments are indicated by numbers <= 0)
        remove_unvoiced (bool, optional):if True, all unvoiced segments will be removed from the pitch curves. Defaults to True.

    Returns:
        float: the PCC between both sequences
    """

    # will remove all part of the signal that are unvoiced
    if remove_unvoiced:
        ref = remove_unvoiced(ref)
        dut = remove_unvoiced(dut)

    # resample the second signal to get the same lenght
    dut_rs = resample(dut, len(ref))

    # compute the Pearson's correlation 
    return pcc(ref, dut)
