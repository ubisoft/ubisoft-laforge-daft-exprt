import logging

import numpy as np
import scipy

from librosa.filters import mel as librosa_mel_fn


_logger = logging.getLogger(__name__)


def _nnls_obj(x, shape, A, B):
    ''' Compute the objective and gradient for NNLS
    '''
    # scipy's lbfgs flattens all arrays, so we first reshape
    # the iterate x
    x = x.reshape(shape)

    # compute the difference matrix
    diff = np.dot(A, x) - B

    # compute the objective value
    value = 0.5 * np.sum(diff ** 2)

    # and the gradient
    grad = np.dot(A.T, diff)

    # flatten the gradient
    return value, grad.flatten()


def _nnls_lbfgs_block(A, B, x_init=None, **kwargs):
    ''' Solve the constrained problem over a single block

    :param A:           the basis matrix -- shape = (m, d)
    :param B:           the regression targets -- shape = (m, N)
    :param x_init:      initial guess -- shape = (d, N)
    :param kwargs:      additional keyword arguments to `scipy.optimize.fmin_l_bfgs_b`

    :return: non-negative matrix x such that Ax ~= B -- shape = (d, N)
    '''
    # if we don't have an initial point, start at the projected
    # least squares solution
    if x_init is None:
        x_init = np.linalg.lstsq(A, B, rcond=None)[0]
        np.clip(x_init, 0, None, out=x_init)

    # adapt the hessian approximation to the dimension of the problem
    kwargs.setdefault("m", A.shape[1])

    # construct non-negative bounds
    bounds = [(0, None)] * x_init.size
    shape = x_init.shape

    # optimize
    x, obj_value, diagnostics = scipy.optimize.fmin_l_bfgs_b(
        _nnls_obj, x_init, args=(shape, A, B), bounds=bounds, **kwargs
    )
    # reshape the solution
    return x.reshape(shape)


def nnls(A, B, **kwargs):
    ''' Non-negative least squares.
        Given two matrices A and B, find a non-negative matrix X
        that minimizes the sum squared error:
            err(X) = sum_i,j ((AX)[i,j] - B[i, j])^2

    :param A:           the basis matrix -- shape = (m, n)
    :param B:           the target matrix -- shape = (m, N)
    :param kwargs:      additional keyword arguments to `scipy.optimize.fmin_l_bfgs_b`
    
    :return: non-negative matrix X minimizing ``|AX - B|^2`` -- shape = (n, N)
    '''
    # if B is a single vector, punt up to the scipy method
    if B.ndim == 1:
        return scipy.optimize.nnls(A, B)[0]

    # constrain block sizes to 256 KB
    MAX_MEM_BLOCK = 2 ** 8 * 2 ** 10
    n_columns = MAX_MEM_BLOCK // (A.shape[-1] * A.itemsize)
    n_columns = max(n_columns, 1)

    # process in blocks
    if B.shape[-1] <= n_columns:
        return _nnls_lbfgs_block(A, B, **kwargs).astype(A.dtype)

    x = np.linalg.lstsq(A, B, rcond=None)[0].astype(A.dtype)
    np.clip(x, 0, None, out=x)
    x_init = x

    for bl_s in range(0, x.shape[-1], n_columns):
        bl_t = min(bl_s + n_columns, B.shape[-1])
        x[:, bl_s:bl_t] = _nnls_lbfgs_block(
            A, B[:, bl_s:bl_t], x_init=x_init[:, bl_s:bl_t], **kwargs
        )
    return x 


def mel_to_linear(mel_spectrogram, hparams):
    ''' Convert a mel-spectrogram to a linear spectrogram

    :param mel_spectrogram:     Numpy array of the input mel spectrogram -- shape = (n_mels, T)
    :param hparams:             hyper-parameters used for pre-processing and training

    :return: numpy array containing the spectrogram in linear frequency space -- shape = (n_fft // 2 + 1, T)
    '''
    # find the number of mel components
    n_mels = mel_spectrogram.shape[0]
    # get filter parameters -- (n_mels, 1 + n_fft//2)
    mel_filter_bank = librosa_mel_fn(hparams.sampling_rate, hparams.filter_length, n_mels, hparams.mel_fmin, hparams.mel_fmax)

    # solve the non-linear least squares problem
    return nnls(mel_filter_bank, mel_spectrogram)


def reconstruct_signal_griffin_lim(magnitude_spectrogram, step_size, iterations, logger):
    ''' Reconstruct an audio signal from a magnitude spectrogram

        Given a magnitude spectrogram as input, reconstruct the audio signal and return it using
        the Griffin-Lim algorithm
        From the paper: "Signal estimation from modified short-time fourier transform" by Griffin and Lim, in IEEE
                        transactions on Acoustics, Speech, and Signal Processing. Vol ASSP-32, No. 2, April 1984.

    :param magnitude_spectrogram:   Numpy array magnitude spectrogram -- shape = (n_fft // 2 + 1, T)
                                    The rows correspond to frequency bins and the columns correspond to time slices
    :param step_size:               length (in samples) between successive analysis windows
    :param iterations:              Number of iterations for the Griffin-Lim algorithm
                                    Typically a few hundred is sufficient
    :param logger:                  logger object

    :return: the reconstructed time domain signal as a 1-dim Numpy array and the spectrogram that was used
             to produce the signal
    '''
    # shape = (T, n_fft // 2 + 1)
    magnitude_spectrogram = np.transpose(magnitude_spectrogram)

    # find the number of samples used in the FFT window and extract the time steps
    n_fft = (magnitude_spectrogram.shape[1] - 1) * 2
    time_slices = magnitude_spectrogram.shape[0]

    # compute the number of samples needed
    len_samples = int(time_slices * step_size + n_fft)

    # initialize the reconstructed signal to noise
    x_reconstruct = np.random.randn(len_samples)
    window = np.hanning(n_fft)
    n = iterations  # number of iterations of Griffin-Lim algorithm

    while n > 0:
        # decrement and compute FFT
        n -= 1
        reconstruction_spectrogram = np.array([np.fft.rfft(window * x_reconstruct[i: i + n_fft])
                                               for i in range(0, len(x_reconstruct) - n_fft, step_size)])

        # Discard magnitude part of the reconstruction and use the supplied magnitude spectrogram instead
        proposal_spectrogram = magnitude_spectrogram * np.exp(1.0j * np.angle(reconstruction_spectrogram))

        # store previous reconstructed signal and create a new one by iFFT
        prev_x = x_reconstruct
        x_reconstruct = np.zeros(len_samples)

        for i, j in enumerate(range(0, len(x_reconstruct) - n_fft, step_size)):
            x_reconstruct[j: j + n_fft] += window * np.real(np.fft.irfft(proposal_spectrogram[i]))

        # normalise signal due to overlap add
        x_reconstruct = x_reconstruct / (n_fft / step_size / 2)

        # compute diff between two signals and report progress
        diff = np.sqrt(sum((x_reconstruct - prev_x) ** 2) / x_reconstruct.size)
        logger.debug(f'Reconstruction iteration: {iterations - n}/{iterations} -- RMSE: {diff * 1e6:.3f}e-6')

    return x_reconstruct, proposal_spectrogram


def griffin_lim_reconstruction_from_mel_spec(mel_spec, hparams, logger):
    ''' Convert a mel-spectrogram into an audio waveform using Griffin-Lim algorithm

    :param mel_spec:        mel-spectrogram corresponding to the audio to generate
    :param hparams:         hyper-parameters used for pre-processing and training
    :param logger:          logger object

    :return: the reconstructed audio waveform
    '''
    # remove np.log
    mel_spec = np.exp(mel_spec)

    # pass from mel to linear
    linear_spec = mel_to_linear(mel_spec, hparams)

    # use Griffin-Lim algorithm
    waveform = []
    if len(linear_spec.shape) == 2:
        waveform, _ = reconstruct_signal_griffin_lim(linear_spec[:, :-2], hparams.hop_length,
                                                     iterations=30, logger=logger)
        waveform = waveform / np.max(abs(waveform))

    return waveform
