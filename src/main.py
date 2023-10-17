import matplotlib.pyplot as plt
import numpy as np
import math
from copy import copy
from math import sin, cos, pi

def primefactors(N): #Разложение длины сигнала на простые множители
    n = []
    while N % 2 == 0:
        n.append(2)
        N = N / 2
    for i in range(3, int(math.sqrt(N)) + 1, 2):
        while (N % i == 0):
            n.append(i)
            N = N / i
    if N > 2:
        n.append(int(N))
    return n

def fault():
    print("Размер длины последовательности имеет делители, отличные от: 2, 3, 5.")
    exit()

def fft2(s): #БПФ по основанию 2
    return [s[0] + s[1], s[0] - s[1]]

def ifft2(s): #ОБПФ по основанию 2
    return [(s[0] + s[1]) / 2, (s[0] - s[1]) / 2]

def fft3(s): #БПФ по основанию 3
    return [s[2] + s[1] + s[0], -(s[2] + s[1]) / 2 + s[0] + 1j * sin(pi / 3) * (s[2] - s[1]), -(s[2] + s[1]) / 2 + s[0] - 1j * sin(pi / 3) * (s[2] - s[1])]

def ifft3(s): #ОБПФ по основанию 3
    return [(s[2] + s[1] + s[0]) / 3, (-(s[2] + s[1]) / 2 + s[0] - 1j * sin(pi / 3) * (s[2] - s[1])) / 3, (-(s[2] + s[1]) / 2 + s[0] + 1j * sin(pi / 3) * (s[2] - s[1])) / 3]

def fft5(s): #БПФ по основанию 5
    q0 = (s[1] + s[4] + s[2] + s[3]) * ((cos(2 * pi / 5) - cos(pi / 5)) / 2) + s[0] + (s[1] + s[4] - s[2] - s[3]) * ((cos(pi / 5) + cos(2 * pi / 5)) / 2)
    q1 = (s[1] + s[4] + s[2] + s[3]) * ((cos(2 * pi / 5) - cos(pi / 5)) / 2) + s[0] - (s[1] + s[4] - s[2] - s[3]) * ((cos(pi / 5) + cos(2 * pi / 5)) / 2)
    q2 = (s[1] - s[4] + s[2] - s[3]) * (-1j * sin(2 * pi / 5)) - (s[2] - s[3]) * (1j * (sin(pi / 5) - sin(2 * pi / 5)))
    q3 = (s[1] - s[4] + s[2] - s[3]) * (-1j * sin(2 * pi / 5)) - (s[1] - s[4]) * (-1j * (sin(pi / 5) + sin(2 * pi / 5)))
    return [s[0] + s[1] + s[4] + s[2] + s[3], q0 + q2, q1 - q3, q1 + q3, q0 - q2]

def ifft5(s): #ОБПФ по основанию 5
    q00 = (s[1] + s[4] + s[2] + s[3]) * ((cos(2 * pi / 5) - cos(pi / 5)) / 2) + s[0] + (s[1] + s[4] - s[2] - s[3]) * ((cos(pi / 5) + cos(2 * pi / 5)) / 2)
    q11 = (s[1] + s[4] + s[2] + s[3]) * ((cos(2 * pi / 5) - cos(pi / 5)) / 2) + s[0] - (s[1] + s[4] - s[2] - s[3]) * ((cos(pi / 5) + cos(2 * pi / 5)) / 2)
    q22 = (s[1] - s[4] + s[2] - s[3]) * (1j * sin(2 * pi / 5)) - (s[2] - s[3]) * (-1j * (sin(pi / 5) - sin(2 * pi / 5)))
    q33 = (s[1] - s[4] + s[2] - s[3]) * (1j * sin(2 * pi / 5)) - (s[1] - s[4]) * (1j * (sin(pi / 5) + sin(2 * pi / 5)))
    return [(s[0] + s[1] + s[4] + s[2] + s[3]) / 5, (q00 + q22) / 5, (q11 - q33) / 5, (q11 + q33) / 5, (q00 - q22) / 5]

def fft_compound_len(signal):
    N = len(signal)
    prime_seq = primefactors(N)

    for i in prime_seq:
        if i != 2 and i != 3 and i != 5: fault()
    L = prime_seq[0]
    M = int(N / L)
    # Заполняем матрицу W [L на M] поворотных коэффициентов составного FFT
    W = np.zeros((L, M), dtype=complex)
    for l in range(L):
        for m in range(M):
            W[l, m] = np.exp(-1j * 2 * np.pi * l * m / N)

    B = signal.reshape((L, M)).T  # Переводим входной сигнал в матрицу [M на L]
    for i in range(len(B)):
        col_len = len(B[i])
        if col_len % 2 == 0:
            B[i] = fft2(B[i])
        elif col_len % 3 == 0:
            B[i] = fft3(B[i])
        elif col_len % 5 == 0:
            B[i] = fft5(B[i])
    Z = B.T * W

    if (len(Z[0]) == 2 or len(Z[0]) == 3 or len(Z[0]) == 5):
        for i in range(len(Z)):
            col_len = len(Z[i])
            if col_len % 2 == 0:
                Z[i] = fft2(Z[i])
            elif col_len % 3 == 0:
                Z[i] = fft3(Z[i])
            elif col_len % 5 == 0:
                Z[i] = fft5(Z[i])
        Z = Z.T.reshape(M * L)
        return Z
    else:
        for i in range(len(Z)):
            Z[i] = fft_compound_len(Z[i])
    Z = Z.reshape(L * M, order='F')
    return Z

def ifft_compound_len(signal): #Алгоритм ОБПФ для составной длины(из простых множителей 2, 3, 5)
    N = len(signal)
    prime_seq = primefactors(N)

    for i in prime_seq:
        if i != 2 and i != 3 and i != 5: fault()
    L = prime_seq[0]
    M = int(N / L)
    # Заполняем матрицу W [L на M] поворотных коэффициентов составного FFT
    W = np.zeros((L, M), dtype=complex)
    for l in range(L):
        for m in range(M):
            W[l, m] = np.exp(1j * 2 * np.pi * l * m / N)

    B = signal.reshape((L, M)).T  # Переводим входной сигнал в матрицу [M на L]
    for i in range(len(B)):
        col_len = len(B[i])
        if col_len % 2 == 0:
            B[i] = ifft2(B[i])
        elif col_len % 3 == 0:
            B[i] = ifft3(B[i])
        elif col_len % 5 == 0:
            B[i] = ifft5(B[i])
    Z = B.T * W

    if (len(Z[0]) == 2 or len(Z[0]) == 3 or len(Z[0]) == 5):
        for i in range(len(Z)):
            col_len = len(Z[i])
            if col_len % 2 == 0:
                Z[i] = ifft2(Z[i])
            elif col_len % 3 == 0:
                Z[i] = ifft3(Z[i])
            elif col_len % 5 == 0:
                Z[i] = ifft5(Z[i])
        Z = Z.T.reshape(M * L)
        return Z
    else:
        for i in range(len(Z)):
            Z[i] = ifft_compound_len(Z[i])
    Z = Z.reshape(L * M, order='F')
    return Z

def fft(seq_len, signal):
    if seq_len == 1:
        return signal
    elif seq_len == 2:
        return fft2(signal)
    elif seq_len == 3:
        return fft3(signal)
    elif seq_len == 5:
        return fft5(signal)
    else:
        return fft_compound_len(signal)

def ifft(seq_len, signal):
    if seq_len == 1:
        return signal
    elif seq_len == 2:
        return ifft2(signal)
    elif seq_len == 3:
        return ifft3(signal)
    elif seq_len == 5:
        return ifft5(signal)
    else:
        return ifft_compound_len(signal.T)

# def verify():
#     for i in range(5):
#         for j in range(4):
#             for k in range(3):
#                 len = 2**i * 3**j * 5**k
#                 signal = np.random.randn(len) + 1j * np.random.randn(len)  # Случайный комплексный сигнал
#
#                 # Встроенная функции БПФ и ОБПФ для проверки моего алгоритма
#                 signal_np_fft = np.fft.fft(copy(signal))
#                 signal_np_ifft = np.fft.ifft(copy(signal_np_fft))
#
#                 signal_my_fft = fft(len, copy(signal))
#                 signal_my_ifft = ifft(len, copy(signal_my_fft))
#
#                 res = abs(signal - signal_my_ifft)
#                 for elem in res.T:
#                     if elem > 0.001:
#                         print(f"diff = {elem}")


def main():
    # verify()
    seq_len = int(input("Введите размерность БПФ, кратную только степеням 2, 3, 5: "))
    signal = np.random.randn(seq_len) + 1j * np.random.randn(seq_len)  # Случайный комплексный сигнал

    # Встроенная функции БПФ и ОБПФ для проверки моего алгоритма
    signal_np_fft = np.fft.fft(copy(signal))
    signal_np_ifft = np.fft.ifft(copy(signal_np_fft))

    signal_my_fft = fft(seq_len, copy(signal))
    signal_my_ifft = ifft(seq_len, copy(signal_my_fft))

    print("Reference signal ", signal)
    print("---------------")
    print("Mine IFFT ", signal_my_ifft)
    print("---------------")
    print("NumPy IFFT", signal_np_ifft)
    print("---------------")
    print("NumPy FFT", signal_np_fft)
    print("---------------")
    print("Mine FFT ", signal_my_fft)

    plt.title('Absolute error between input and output data ')
    plt.xlabel('Item number')
    plt.ylabel('Absolute error ')
    plt.plot(abs(signal-signal_my_ifft).T, 'bo')
    plt.show()

if __name__=="__main__":
    main()
