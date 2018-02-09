import numpy as np


def forward(x, pi, A, B):
    """ Run the forward algorithm for a single example.

    Args:
        x: A 1-D int NumPy array with shape [T], where each element
            is either 0, 1, 2, ..., or N_x - 1. T is the length of
            the observation sequence and N_x is the number of possible
            values that each observation can take on.
        pi: A 1-D float NumPy array with shape [N_z]. N_z is the number
            of possible values that each hidden state can take on.
        A: A 2-D float NumPy array with shape [N_z, N_z]. A[i, j] is
            the probability of transitioning from state i to state j:
            A[i, j] = P(z_t = j | z_t-1 = i).
        B: A 2-D float NumPy array with shape [N_z, N_x]. B[i, j] is
            the probability of from state i emitting observation j:
            B[i, j] = P(x_t = j | z_t = i).

    Returns:
        alpha, a 2-D float NumPy array with shape [T, N_z].
    """
    T = x.shape[0]
    N_z = pi.shape[0]
    alpha = np.zeros((T,N_z), dtype=np.float64)

    #calculate alpha_0z_0
    x_0 = x[0]
    alpha[0] = pi * B[:,x_0]
    for i in range(1, x.shape[0]):
        x_t = x[i]
        b_t = B[:,x_t]
        alpha[i] = b_t*(np.matmul(alpha[i-1,:], A))
    return alpha

def backward(x, pi, A, B):
    """ Run the backward algorithm for a single example.

    Args:
        x: A 1-D int NumPy array with shape [T], where each element
            is either 0, 1, 2, ..., or N_x - 1. T is the length of
            the observation sequence and N_x is the number of possible
            values that each observation can take on.
        pi: A 1-D float NumPy array with shape [N_z]. N_z is the number
            of possible values that each hidden state can take on.
        A: A 2-D float NumPy array with shape [N_z, N_z]. A[i, j] is
            the probability of transitioning from state i to state j:
            A[i, j] = P(z_t = j | z_t-1 = i).
        B: A 2-D float NumPy array with shape [N_z, N_x]. B[i, j] is
            the probability of from state i emitting observation j:
            B[i, j] = P(x_t = j | z_t = i).

    Returns:
        beta, a 2-D float NumPy array with shape [T, N_z].
    """

    T = x.shape[0]
    N_z = pi.shape[0]
    beta = np.empty((T,N_z), dtype=np.float64)

    beta[T-1] = (1,1)
    for i in range(x.shape[0]-2, -1, -1):
        x_t1 = x[i+1]
        b_t = B[:,x_t1]
        beta[i] = (A * b_t).dot(beta[i+1])
    return beta

def individually_most_likely_states(X, pi, A, B):
    """ Computes individually most-likely states.

    By "individually most-likely states," we mean that the *marginal*
    distributions are maximized. In other words, for any particular
    time step of any particular sequence, each returned state i is
    chosen to maximize P(z_t = i | x).

    All sequences in X are assumed to have the same length, T.

    Args:
        X: A 2-D int NumPy array with shape [N, T], where each element
            is either 0, 1, 2, ..., or N_x - 1. N is the number of observation
            sequences, T is the length of every sequence, and N_x is the number
            of possible values that each observation can take on.
        pi: A 1-D float NumPy array with shape [N_z]. N_z is the number
            of possible values that each hidden state can take on.
        A: A 2-D float NumPy array with shape [N_z, N_z]. A[i, j] is
            the probability of transitioning from state i to state j:
            A[i, j] = P(z_t = j | z_t-1 = i).
        B: A 2-D float NumPy array with shape [N_z, N_x]. B[i, j] is
            the probability of from state i emitting observation j:
            B[i, j] = P(x_t = j | z_t = i).

    Returns:
        Z, a 2-D int NumPy array with shape [N, T], where each element
            is either 0, 1, 2, ..., N_z - 1.
    """
    #define Z numpy_matrix N*T
    Z = np.empty((X.shape[0], X.shape[1]), dtype=np.int)
    for i in range(X.shape[0]):
        x = X[i,:]
        alpha = forward(x, pi, A, B)
        beta = backward(x, pi, A, B)
        p_x = np.sum(alpha[alpha.shape[0]-1])

        m = alpha * beta / p_x
        z = np.argmax(m, axis = 1)
        Z[i] = z
    return Z

def take_EM_step(X, pi, A, B):
    """ Take a single expectation-maximization step.

    Args:
        X: A 2-D int NumPy array with shape [N, T], where each element
            is either 0, 1, 2, ..., or N_x - 1. N is the number of observation
            sequences, T is the length of every sequence, and N_x is the number
            of possible values that each observation can take on.
        pi: A 1-D float NumPy array with shape [N_z]. N_z is the number
            of possible values that each hidden state can take on.
        A: A 2-D float NumPy array with shape [N_z, N_z]. A[i, j] is
            the probability of transitioning from state i to state j:
            A[i, j] = P(z_t = j | z_t-1 = i).
        B: A 2-D float NumPy array with shape [N_z, N_x]. B[i, j] is
            the probability of from state i emitting observation j:
            B[i, j] = P(x_t = j | z_t = i).

    Returns:
        A tuple containing
        pi_prime: pi after the EM update.
        A_prime: A after the EM update.
        B_prime: B after the EM update.
    """
    #set pi_hat, A_hat, B_hat to zero
    N, T = X.shape
    N_z = pi.shape[0]
    N_x = B.shape[1]

    pi_prime = np.zeros((2), dtype=np.float64)
    A_prime = np.zeros((N_z, N_z), dtype=np.float64)
    B_prime = np.zeros((N_z, N_x), dtype=np.float64)

    for seq_id in range(N):
        x = X[seq_id]
        #call forward, backward to obtain alpha and beta for each sequence
        alpha = forward(x, pi, A, B)
        beta = backward(x, pi, A, B)
        
        p_x = alpha[-1].sum()
        pi_prime += (alpha[0] * beta[0])/p_x

        #calucalte A_prime
        a_prime_temp=np.zeros((A.shape))
        for t in range(len(alpha)-1):
            alpha_temp = alpha[t]
            b_t = B[:, x[t+1]]
            a_prime_temp += (A * b_t * beta[t+1]) * alpha_temp[:, np.newaxis]
        
        b_prime_temp=np.zeros((B.shape))
        for t in range(len(alpha)):
            b_prime_temp[:,x[t]] += alpha[t] * beta[t]

        A_prime += a_prime_temp/p_x
        B_prime += b_prime_temp/p_x
    
    #normalise pi_prime, A_prime, B_prime
    pi_prime = pi_prime/pi_prime.sum()
    a_sum = A_prime.sum(axis=1)
    A_prime = A_prime / a_sum[:, np.newaxis]

    b_sum = B_prime.sum(axis=1)
    B_prime = B_prime / b_sum[:, np.newaxis]
    
    return (pi_prime, A_prime, B_prime)
