import argparse
import numpy as np
import hmms


def get_args():
    parser = argparse.ArgumentParser(description="Perform HMM inference or learning.")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test'],
                        help='''If train, load observations and initial parameters, run expectation maximization,
                                and save the final parameters. If test, load observations and parameters,
                                predict individually most likely states, and save those predictions.''')
    parser.add_argument('--data', type=str, default='data.txt',
                        help='''The file to read observations from.''')
    parser.add_argument('--predictions-file', type=str, default='predictions-file.txt',
                        help='''The file to save predictions to. Only used if mode == test.''')
    parser.add_argument('--model-input-file', type=str, default='model-input-file.txt',
                        help='''Parameters file to read from. These serve as initial parameter estimates if
                                mode == train, and as fixed parameters for inference if mode == test.''')
    parser.add_argument('--model-output-file', type=str, default='model-output-file.txt',
                        help='''Parameters file to write to. Only used if mode == train.''')
    parser.add_argument('--iterations', type=int, default=100,
                        help='The number of expectation-maximization iterations. Only used if mode == train.')
    args = parser.parse_args()
    return args


def load_parameters(path):
    """ Load a plain-text file of parameters.

    Args:
        path: A string.

    Returns:
        A tuple containing
        pi: A 1-D float NumPy array.
        A: A 2-D float NumPy array with shape [N_z, N_z].
        B: A 2-D float NumPy array with shape [N_z, N_x].
    """
    params = np.loadtxt(path)
    N_z = params.shape[0]
    pi, A_and_B = np.split(params, [1], axis=1)
    A, B = np.split(A_and_B, [N_z], axis=1)
    return pi.flatten(), A, B


def save_parameters(path, pi, A, B):
    """ Save a plain-text file of parameters.

    Args:
        path: A string.
        pi: A 1-D float NumPy array.
        A: A 2-D float NumPy array with shape [N_z, N_z].
        B: A 2-D float NumPy array with shape [N_z, N_x].
    """
    N_z = pi.size
    assert A.ndim == 2 and A.shape[0] == N_z and A.shape[1] == N_z
    assert B.ndim == 2 and B.shape[0] == N_z
    params = np.concatenate([pi.reshape(-1, 1), A, B], axis=1)
    np.savetxt(path, params, fmt='%.6f')


def save_hidden_states(path, Z):
    """Save a plain-text file of hidden states.

    Each hidden state must be a nonnegative integer in
    (0, 1, 2, ..., N_z - 1), where N_z is the number of
    possible values that each state can take on.

    Args:
        path: A string.
        Z: A 2-D int NumPy array with shape [N, T], where N
            is the number of sequences and T is the length
            of each sequence. All entries of Z also must be
            nonnegative.
    """
    assert Z.ndim == 2
    assert issubclass(Z.dtype.type, np.integer)
    assert np.all(Z >= 0)
    np.savetxt(path, Z, fmt='%d')


def load_observations(path):
    """Load a plain-text file of observations.

    Each observation must be a nonnegative integer in
    (0, 1, 2, ..., N_x - 1), where N_x is the number of
    possible values that each observation can take on.

    Args:
        path: A string.

    Returns: A 2-D int NumPy array with shape [N, T], where N
        is the number of sequences and T is the length of each
        sequence.
    """
    X = np.loadtxt(path).astype(np.int)
    assert np.all(X >= 0)
    return X


if __name__ == '__main__':
    args = get_args()
    pi, A, B = load_parameters(args.model_input_file)
    X = load_observations(args.data)
    if args.mode == 'test':
        Z = hmms.individually_most_likely_states(X, pi, A, B)
        save_hidden_states(args.predictions_file, Z)
    elif args.mode == 'train':
        for _ in range(args.iterations):
            _, A, B = hmms.take_EM_step(X, pi, A, B)
        save_parameters(args.model_output_file, pi, A, B)
