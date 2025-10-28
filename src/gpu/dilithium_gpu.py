import numpy as np
import cupy as cp
from numba import cuda
import cupy.random as random
import time
import hashlib

# Dilithium parameters
DILITHIUM_MODES = {
    2: {"n": 256, "q": 8380417, "d": 13, "tau": 39, "gamma1": 2 ** 17, "gamma2": 95232, "k": 4, "l": 4, "eta": 2, 'omega': 80},
    3: {"n": 256, "q": 8380417, "d": 13, "tau": 49, "gamma1": 2 ** 19, "gamma2": 261888, "k": 6, "l": 5, "eta": 4, 'omega': 55},
    5: {"n": 256, "q": 8380417, "d": 13, "tau": 60, "gamma1": 2 ** 19, "gamma2": 261888, "k": 8, "l": 7, "eta": 2, 'omega': 75}
}

# Select Dilithium mode (2, 3, or 5)
MODE = 5
PARAMS = DILITHIUM_MODES[MODE]

# Constants
N = PARAMS["n"]
D = PARAMS['d']
Q = PARAMS["q"]
K = PARAMS["k"]
L = PARAMS["l"]
ETA = PARAMS["eta"]
GAMMA1 = PARAMS["gamma1"]
GAMMA2 = PARAMS["gamma2"]
TAU = PARAMS["tau"]
BETA = TAU * ETA
OMEGA = PARAMS['omega']
BATCH_SIZE = 10000

@cuda.jit
def ntt_kernel(a, twiddles):
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x

    shared_mem = cuda.shared.array(shape=(N,), dtype=np.int32)
    shared_mem[tid] = a[bid * N + tid]
    cuda.syncthreads()

    for s in range(1, N.bit_length()):
        m = 1 << s
        for k in range(0, N, m):
            if tid < m // 2:
                idx = k + tid
                t = twiddles[m // 2 + tid] * shared_mem[idx + m // 2] % Q
                u = shared_mem[idx]
                shared_mem[idx] = (u + t) % Q
                shared_mem[idx + m // 2] = (u - t) % Q
        cuda.syncthreads()

    a[bid * N + tid] = shared_mem[tid]

@cuda.jit
def poly_mul_kernel(a, b, c):
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x

    shared_a = cuda.shared.array(shape=(N,), dtype=np.int32)
    shared_b = cuda.shared.array(shape=(N,), dtype=np.int32)

    shared_a[tid] = a[bid * N + tid]
    shared_b[tid] = b[bid * N + tid]
    cuda.syncthreads()

    result = 0
    for j in range(N):
        result += shared_a[j] * shared_b[(tid - j) % N]
    c[bid * N + tid] = result % Q

def batch_key_gen(num_keys):
    start_time = time.time()

    A = random.randint(0, Q, (num_keys, K, L, N), dtype=np.int32)
    s1 = random.randint(-ETA, ETA, (num_keys, L, N), dtype=np.int32)
    s2 = random.randint(-ETA, ETA, (num_keys, K, N), dtype=np.int32)

    d_A = cp.asarray(A)
    d_s1 = cp.asarray(s1)
    d_s2 = cp.asarray(s2)

    t = cp.zeros((num_keys, K, N), dtype=np.int32)
    for i in range(K):
        for j in range(L):
            temp_result = cp.sum(d_A[:, i, j, :, None] * d_s1[:, j, None, :], axis=2)
            t[:, i] += temp_result
    t += d_s2
    t %= Q

    end_time = time.time()
    key_gen_time = end_time - start_time

    return d_A, d_s1, d_s2, t, key_gen_time

def compute_challenge(message, w1):
    w1_bytes = cp.asnumpy(w1).tobytes()
    challenge_bytes = hashlib.shake_256(message.encode() + w1_bytes).digest(32)
    return np.frombuffer(challenge_bytes, dtype=np.uint32)


def batch_sign(messages, A, s1, s2):
    start_time = time.time()
    num_messages = len(messages)

    y = cp.random.randint(-GAMMA1 + 1, GAMMA1, (num_messages, L, N), dtype=cp.int32)
    # Compute Ay
    Ay = cp.zeros((num_messages, K, N), dtype=cp.int32)
    for i in range(K):
        for j in range(L):
            Ay[:, i] += cp.sum(A[:, i, j, :, None] * y[:, j, None, :], axis=2)

    Ay %= Q
    w = cp.zeros((num_messages, K, N), dtype=cp.int32)
    for i in range(K):
        for j in range(L):
            w[:, i] += cp.sum(A[:, i, j, :, None] * y[:, j, None, :], axis=2)
    w %= Q

    w1 = (w + (Q - 1) // 2) // (Q // (2 ** D))

    c_np = np.array([compute_challenge(m, w1[i]) for i, m in enumerate(messages)])
    c = cp.array(c_np)

    z = y.copy()
    for i in range(c.shape[1]):
        z += cp.einsum('i,ijk->ijk', c[:, i], s1)

    # Apply modulo Q
    z %= Q

    # Center around 0
    z = cp.where(z >= Q // 2, z - Q, z)

    # Clip to the allowed range
    z = cp.clip(z, -GAMMA1 + BETA, GAMMA1 - BETA - 1)

    h = cp.zeros((num_messages, K, N), dtype=cp.int32)
    for i in range(K):
        r = w[:, i] - cp.einsum('ij,ijk->ik', c, s2[:, :, :])
        r %= Q

        r0 = r % (Q // (2 ** D))
        r1 = (r - r0) // (Q // (2 ** D))

        h[:, i] = cp.where(
            (r0 > Q // 2) | ((r0 == Q // 2) & (r1 % 2 == 1)),
            cp.ones_like(r, dtype=cp.int32),
            cp.zeros_like(r, dtype=cp.int32)
        )

    z_ok = cp.all((z >= -GAMMA1 + BETA) & (z < GAMMA1 - BETA), axis=(1, 2))
    h_ok = cp.sum(h, axis=(1, 2)) <= OMEGA


    valid_mask = z_ok & h_ok

    end_time = time.time()
    sign_time = end_time - start_time

    return z, c, h, valid_mask, sign_time


def batch_verify(messages, signatures, A, t):
    start_time = time.time()

    z, c, h = signatures
    num_messages = len(messages)

    # Compute Az
    Az = cp.zeros((num_messages, K, N), dtype=cp.int32)
    for i in range(K):
        for j in range(L):
            Az[:, i] += cp.sum(A[:, i, j, :, None] * z[:, j, None, :], axis=2)

    # Compute ct
    ct = cp.zeros((num_messages, K, N), dtype=cp.int32)
    for i in range(K):
        ct[:, i] = cp.sum(c[:, :, None] * t[:, i, None, :], axis=1)

    # Compute w as before
    w = (Az - ct) % Q
    w1_prime = (w + (Q - 1) // 2) // (Q // (2 ** D))

    c_prime_np = np.array([compute_challenge(m, w1_prime[i]) for i, m in enumerate(messages)])
    c_prime = cp.array(c_prime_np)

    z_ok = cp.all((z >= -GAMMA1 + BETA) & (z < GAMMA1 - BETA), axis=(1, 2))
    h_ok = cp.sum(h, axis=(1, 2)) <= OMEGA
    c_ok = cp.all(c == c_prime, axis=1)

    results = z_ok & h_ok & c_ok

    end_time = time.time()
    verify_time = end_time - start_time

    return results, verify_time, z_ok, h_ok, c_ok


def run():
    A, s1, s2, t, key_gen_time = batch_key_gen(BATCH_SIZE)

    messages = [f"Message {i}" for i in range(BATCH_SIZE)]

    z, c, h, valid_mask, sign_time = batch_sign(messages, A, s1, s2)
    results, verify_time, z_ok, h_ok, c_ok = batch_verify(messages, (z, c, h), A, t)
