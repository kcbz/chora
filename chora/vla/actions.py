"""Declared action representations; all executed commands remain joint targets."""
import numpy as np

REPRESENTATIONS = {'absolute', 'relative_arm'}


def encode_actions(actions, reference, representation):
    if representation not in REPRESENTATIONS:
        raise ValueError('Unknown action representation')
    value = np.asarray(actions, dtype=np.float64).copy()
    if representation == 'relative_arm':
        value[:, :5] -= np.asarray(reference)[:5]
    return value


def decode_actions(actions, reference, representation):
    if representation not in REPRESENTATIONS:
        raise ValueError('Unknown action representation')
    value = np.asarray(actions, dtype=np.float64).copy()
    if representation == 'relative_arm':
        value[:, :5] += np.asarray(reference)[:5]
    return value
