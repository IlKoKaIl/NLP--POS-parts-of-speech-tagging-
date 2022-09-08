import numpy as np

from tagger import *
from collections import defaultdict

def read_data_test(path):
    return open(path, 'r').read().split('\n')

def test_tagging(test_file):
    pred = np.asarray(read_data_test(test_file+'.pred'))
    soln = np.asarray(read_data_test(test_file+'.soln')[:-1])

    acc = np.sum(pred==soln) / len(soln)
    pct = min(acc, tagging_threshold) / tagging_threshold

    return pct

def almost_equal(a, b, diff=1e-4):
    if a == -np.inf:
        return b == -np.inf
    return np.abs(a-b) < diff

def array_almost_equal(a, b, diff=1e-4):
    if a.shape != b.shape:
        return False
    a_flat = a.flatten()
    b_flat = b.flatten()
    for i in range(a_flat.shape[0]):
        if not almost_equal(a_flat[i], b_flat[i]):
            return False
    return True

scheme = {
    'prior': 14,
    'transition':14,
    'emission':14,
    'tagging_small':35,
    'tagging_large':23
}

tagging_threshold = 0.85

test_names = list(scheme.keys())

if __name__ == '__main__':
    prior, transition, emission = train_HMM('data/train-public')

    results = defaultdict(float)
    solutions = np.load('soln.npz')

    # Prior Tests

    try:
        assert(array_almost_equal(prior, solutions['prior']))
        results['prior'] = scheme['prior']
    except Exception:
        pass

    # Transition Tests

    try:
        assert(array_almost_equal(transition, solutions['transition']))
        results['transition'] = scheme['transition']
    except Exception:
        pass

    # Emission Tests

    try:
        assert(array_almost_equal(np.asarray([emission[(key[0], key[1])] for key in solutions['emission_key']]), solutions['emission_val']))
        results['emission'] = scheme['emission']
    except Exception:
        pass

    # Tagging tests

    # Small

    try:
        results['tagging_small'] = test_tagging('data/test-public-small') * scheme['tagging_small']
    except Exception:
        pass

    # Large

    try:
        results['tagging_large'] = test_tagging('data/test-public-large') * scheme['tagging_large']
    except Exception:
        pass

    for t in test_names:
        print(t+": %.1f/%.1f" % (results[t], scheme[t]))
    print("[Total]: %.1f/%.1f" % (np.sum([results[t] for t in test_names]), np.sum([scheme[t] for t in test_names])))

