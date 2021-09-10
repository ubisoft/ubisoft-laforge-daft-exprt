import logging
import multiprocessing as mp
import sys
import threading
import time

import matplotlib.pyplot as plt

from functools import partial
from multiprocessing import Pool

from dateutil.relativedelta import relativedelta


_logger = logging.getLogger(__name__)


def histogram_plot(data, x_labels, y_labels, figsize=(16, 4)):
    ''' Histogram plot for different set of data
    '''
    # create subplot
    fig, axes = plt.subplots(1, len(data), figsize=figsize, squeeze=False)
    # create an histogram plot for each item in data
    for i in range(len(data)):
        # (B, N) --> (B * N)
        data_vals = data[i].ravel()
        # plot histogram
        axes[0, i].hist(data_vals, bins=50, density=True)
        # add axis labels
        if x_labels is not None:
            axes[0, i].set(xlabel=x_labels[i])
        if y_labels is not None:
            axes[0, i].set(ylabel=y_labels[i])
    plt.close(fig)
    
    return fig


def scatter_plot(data, colors, labels, x_label=None, y_label=None, figsize=(16, 4)):
    ''' Scatter plots of different data points
    '''
    # create subplot
    fig, axes = plt.subplots(1, 1, figsize=figsize, squeeze=False)
    # fill with data
    for item, color in zip(data, colors):
        axes[0, 0].scatter(range(len(item)), item, color=color, marker='o')
    # add plots labels
    axes[0, 0].legend(labels=labels)
    # add axis labels
    if x_label is not None:
        axes[0, 0].set(xlabel=x_label)
    if y_label is not None:
        axes[0, 0].set(ylabel=y_label)
    plt.close(fig)
    
    return fig


def plot_2d_data(data, x_labels=None, y_labels=None, filename=None, figsize=(16, 4)):
    ''' Create several 2D plots for each item given by data

    :param data:            sequence of numpy arrays -- length (L, )
    :param x_labels:        labels to give to each plot on the x axis -- length (L, ) if not None
    :param y_labels:        labels to give to each plot on the y axis -- length (L, ) if not None
    :param filename:        file to save the figure
    :param figsize:         size of the plots

    :return: the 2D plot
    '''
    # initialize the subplot -- put squeeze to false to avoid errors when data is of length 1
    fig, axes = plt.subplots(1, len(data), figsize=figsize, squeeze=False)

    # create a plot for each item given by data
    for i in range(len(data)):
        if len(data[i].shape) == 1:
            axes[0, i].scatter(range(len(data[i])), data[i], alpha=0.5, marker='.', s=10)
        elif len(data[i].shape) == 2:
            axes[0, i].imshow(data[i], aspect='auto', origin='lower', interpolation='none')
        if x_labels is not None:
            axes[0, i].set(xlabel=x_labels[i])
        if y_labels is not None:
            axes[0, i].set(ylabel=y_labels[i])
    
    # save the figure and return it
    if filename is not None:
        fig.savefig(filename)

    plt.close(fig)
    return fig


def chunker(seq, size):
    ''' creates a list of chunks
        https://stackoverflow.com/a/434328

    :param seq: the sequence we want to create chunks from
    :param size: size of the chunks
    '''
    return (seq[pos: pos + size] for pos in range(0, len(seq), size))


def prog_bar(i, n, bar_size=16):
    """ Create a progress bar to estimate remaining time

    :param i:           current iteration
    :param n:           total number of iterations
    :param bar_size:    size of the bar

    :return: a visualisation of the progress bar
    """
    bar = ''
    done = (i * bar_size) // n

    for j in range(bar_size):
        bar += '█' if j <= done else '░'

    message = f'{bar} {i}/{n}'
    return message


def estimate_required_time(nb_items_in_list, current_index, time_elapsed, interval=100):
    """ Compute a remaining time estimation to process all items contained in a list

    :param nb_items_in_list:        all list items that have to be processed
    :param current_index:           current list index, contained in [0, nb_items_in_list - 1]
    :param time_elapsed:            time elapsed to process current_index items in the list
    :param interval:                estimate remaining time when (current_index % interval) == 0

    :return: time elapsed since the last time estimation
    """
    current_index += 1  # increment current_idx by 1
    if current_index % interval == 0 or current_index == nb_items_in_list:
        # make time estimation and put to string format
        seconds = (nb_items_in_list - current_index) * (time_elapsed / current_index)
        time_estimation = relativedelta(seconds=int(seconds))
        time_estimation_string = f'{time_estimation.hours:02}:{time_estimation.minutes:02}:{time_estimation.seconds:02}'

        # extract progress bar
        progress_bar = prog_bar(i=current_index, n=nb_items_in_list)

        # display info
        if current_index == nb_items_in_list:
            sys.stdout.write(f'\r{progress_bar} -- estimated required time = {time_estimation_string} -- Finished!')
        else:
            sys.stdout.write(f'\r{progress_bar} -- estimated required time = {time_estimation_string} -- ')


def get_nb_jobs(n_jobs):
    """ Return the number of parallel jobs specified by n_jobs

    :param n_jobs:      the number of jobs the user want to use in parallel

    :return: the number of parallel jobs
    """
    # set nb_jobs to max by default
    nb_jobs = mp.cpu_count()
    if n_jobs != 'max':
        if int(n_jobs) > mp.cpu_count():
            _logger.warning(f'Max number of parallel jobs is "{mp.cpu_count()}" but received "{int(n_jobs)}" -- '
                            f'setting nb of parallel jobs to {nb_jobs}')
        else:
            nb_jobs = int(n_jobs)

    return nb_jobs


def logger_thread(q):
    ''' Thread logger to listen to log outputs in multi-processing mode
    '''
    while True:
        log_record = q.get()
        if log_record is None:
            break
        _logger.handle(log_record)


def launch_multi_process(iterable, func, n_jobs, chunksize=1, ordered=True, timer_verbose=True, **kwargs):
    """ Calls function using multi-processing pipes
        https://guangyuwu.wordpress.com/2018/01/12/python-differences-between-imap-imap_unordered-and-map-map_async/

    :param iterable:        items to process with function func
    :param func:            function to multi-process
    :param n_jobs:          number of parallel jobs to use
    :param chunksize:       size of chunks given to each worker
    :param ordered:         True: iterable is returned while still preserving the ordering of the input iterable
                            False: iterable is returned regardless of the order of the input iterable -- better perf
    :param timer_verbose:   display time estimation when set to True
    :param kwargs:          additional keyword arguments taken by function func

    :return: function outputs
    """
    # set up a queue and listen to log messages on it in another thread
    m = mp.Manager()
    q = m.Queue()
    lp = threading.Thread(target=logger_thread, args=(q, ))
    lp.start()
    
    # define pool of workers
    pool = Pool(processes=n_jobs)
    # define partial function and pool function
    func = partial(func, log_queue=q, **kwargs)
    pool_func = pool.imap if ordered else pool.imap_unordered

    # initialize variables
    func_returns = []
    nb_items_in_list = len(iterable) if timer_verbose else None
    start = time.time() if timer_verbose else None
    # iterate over iterable
    for i, func_return in enumerate(pool_func(func, iterable, chunksize=chunksize)):
        # store function output
        func_returns.append(func_return)
        # compute remaining time
        if timer_verbose:
            estimate_required_time(nb_items_in_list=nb_items_in_list, current_index=i,
                                   time_elapsed=time.time() - start)
    if timer_verbose:
        sys.stdout.write('\n')

    # wait for all worker to finish and close the pool
    pool.close()
    pool.join()

    # put a null message in the queue so that it stops the logging thread
    q.put(None)
    lp.join()

    return func_returns
