import numpy as np
import scipy.linalg
import scipy.stats


def ks_key(X):
    '''Estimate the key from a pitch class distribution

    Parameters
    ----------
    X : np.ndarray, shape=(12,)
        Pitch-class energy distribution.  Need not be normalized

    Returns
    -------
    major : np.ndarray, shape=(12,)
    minor : np.ndarray, shape=(12,)

        For each key (C:maj, ..., B:maj) and (C:min, ..., B:min),
        the correlation score for `X` against that key.
    '''
    X = np.asarray(X)

    # Coefficients from Krumhansl and Schmuckler
    # as reported here: http://rnhart.net/articles/key-finding/
    major = np.asarray([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])

    minor = np.asarray([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])



    major_Rs = np.zeros(12)
    minor_Rs = np.zeros(12)

    for i in range(0,major.shape[0]):
        major_Rs[i] = scipy.stats.pearsonr(X,np.roll(major,i))[0]
        minor_Rs[i] = scipy.stats.pearsonr(X,np.roll(minor,i))[0]

    return major_Rs, minor_Rs

def get_key_r(X):
    X = [X.count(i) for i in range(0,12)]
    major_keys, minor_keys = ks_key(np.asarray(X))

    major_max_ind = major_keys.argmax()
    major_max_val = major_keys[major_max_ind]

    minor_max_ind = minor_keys.argmax()
    minor_max_val = minor_keys[minor_max_ind]


    if(major_max_val>=minor_max_val):
        max_ind = major_max_ind
        r = major_max_val
        mode = "major"
    else:
        max_ind = minor_max_ind
        r = minor_max_val

        mode = "minor"
    r2 = r * r
    keys = ['C','C#/Db', 'D','D#/Eb', 'E', 'F','F#/Gb', 'G','G#/Ab','A','A#/Bb','B']
    key = keys[max_ind]
    return r, key, mode


