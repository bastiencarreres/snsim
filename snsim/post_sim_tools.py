"""This module contains some tools for post-sim analysis."""
import numpy as np


def SNR_pdet(SNR, SNR_mean, SNRp, p):
    r"""Approximation of the SNR detection probability.

    Parameters
    ----------
    SNR : float or np.array(float)
        The Signal to Noise Ratio.
    SNR_mean : float
        The SNR for which p = 0.5.
    SNRp : float
        The SNR for which we have a probability p of detection.
    p : float
        The probability of detection at SNRp.

    Returns
    -------
    float or np.array(float)
        Probability of detection.

    Notes
    -----
    The detection probability function :

    .. math::
        P_{det}(SNR) = \frac{1}{1+\left(\frac{SNR_{mean}}{SNR}\right)^n}

    where :math:`n = \frac{\ln\left(\frac{1-p}{p}\right)}{\ln(SNR_{mean}) - \ln(SNR_p)}`

    """
    n = np.log((1 - p) / p) / (np.log(SNR_mean) - np.log(SNRp))
    return 1 / (1 + (SNR_mean / SNR)**n)


def SNR_select(selec_function,
               lcs,
               SNR_mean=5,
               SNR_limit=[15, 0.99],
               randseed=np.random.randint(1000, 100000)):
    r"""Run a SNR efficiency detection on all lcs.

    Parameters
    ----------
    selec_function : str
        Can be 'approx' function TODO : add interpolation for function from file.
    lcs : pandas.DataFrame
        SN lcs data.
    SNR_mean : float or dic
        The SNR for which the detection probability is 1/2 -> SNR_mean.
    SNR_limit : list of dic(list)
        A SNR and its probability of detection -> $SNR_p$ and p.
    randseed : int
        Randseed for random detection.

    Returns
    -------
    pandas.DataFrame
        The SNR selected epochs.

    Notes
    -----
    The detection probability function :

    .. math::

        P_\text{det}(SNR) = \frac{1}{1+\left(\frac{SNR_\text{mean}}{SNR}\right)^n}

    where :math:`n = \frac{\ln\left(\frac{1-p}{p}\right)}{\ln(SNR_\text{mean}) - \ln(SNR_p)}`

    """
    rand_gen = np.random.default_rng(randseed)
    SNR_proba = {}
    bands = lcs['band'].unique()
    if selec_function == 'approx':
        if isinstance(SNR_limit, (list, np.ndarray)) and isinstance(SNR_mean, (int, np.integer, float, np.floating)):
            for b in bands:
                SNR_proba[b] = lambda SNR: SNR_pdet(SNR,
                                                    SNR_mean,
                                                    SNR_limit[0],
                                                    SNR_limit[1])
        else:
            for b in bands:
                SNR_proba[b] = lambda SNR: SNR_pdet(SNR,
                                                    SNR_mean[b],
                                                    SNR_limit[b][0],
                                                    SNR_limit[b][1])

    SNR = lcs['flux'] / lcs['fluxerr']
    p_det = np.array([SNR_proba[b](s) for b, s in zip(lcs['band'], SNR)])
    return lcs[rand_gen.random(len(SNR)) < p_det]
