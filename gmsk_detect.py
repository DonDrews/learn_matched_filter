import numpy as np
import matplotlib.pyplot as plt
from commpy.filters import rrcosfilter

def convmtx(v, n):
    """Generates a convolution matrix
    
    Usage: X = convm(v,n)
    Given a vector v of length N, an N+n-1 by n convolution matrix is
    generated of the following form:
              |  v(0)  0      0     ...      0    |
              |  v(1) v(0)    0     ...      0    |
              |  v(2) v(1)   v(0)   ...      0    |
         X =  |   .    .      .              .    |
              |   .    .      .              .    |
              |   .    .      .              .    |
              |  v(N) v(N-1) v(N-2) ...  v(N-n+1) |
              |   0   v(N)   v(N-1) ...  v(N-n+2) |
              |   .    .      .              .    |
              |   .    .      .              .    |
              |   0    0      0     ...    v(N)   |
    And then it's trasposed to fit the MATLAB return value.     
    That is, v is assumed to be causal, and zero-valued after N.

    """
    N = len(v) + 2*n - 2
    xpad = np.concatenate([np.zeros(n-1), v[:], np.zeros(n-1)])
    X = np.zeros((len(v)+n-1, n))
    # Construct X column by column
    for i in range(n):
        X[:,i] = xpad[n-i-1:N-i]
    
    return X.transpose()

def get_matched_filter(n, sps):
    x = int((n-1)/4)
    #return np.concatenate([np.zeros(x), np.ones(x*2)/(x*2), np.zeros(x + 1)])
    domain = np.linspace(-(n-1)/sps, (n-1)/sps, n, endpoint=True)
    return np.sinc(domain)/np.sum(np.sinc(domain))
    #return np.array(rrcosfilter(n, alpha=1, Ts=1, Fs=sps)[1])/2

# Constants
NUM_BITS = int(1e6)
SPS = 8
FILT_LEN_SYMB = 6
FILT_LEN = SPS * FILT_LEN_SYMB + 1

# Generate a random set of BPSK symbols
bits = np.random.randint(0, 2, NUM_BITS)*2-1
print(bits)

# Upsample and apply RRC
upsamp = np.zeros((NUM_BITS, SPS))
upsamp[:,0] = bits
upsamp = upsamp.reshape((1,NUM_BITS * SPS))[0]
mf = get_matched_filter(FILT_LEN, SPS)
samps = np.convolve(upsamp, mf, mode='valid')

# Add noise in order to make the linear regression more stable
samps = samps + np.random.normal(scale=0.5, size=len(samps))
# plt.plot(mf)
# plt.figure()
# plt.plot(samps)
# plt.figure()
# plt.plot(np.fft.fftshift(10*np.log10(np.abs(np.fft.fft(samps)))))
# plt.show()

# Generate the convolution matrix of the samples with an
# unknown MF, then downsample to select only the symbols
cnv = convmtx(samps, FILT_LEN).transpose()[1::SPS]
cnv = cnv[FILT_LEN_SYMB-1:-FILT_LEN_SYMB+2]
comp = bits[FILT_LEN_SYMB-1:-FILT_LEN_SYMB+2]
plt.imshow(cnv)
plt.figure()
result = np.matmul(cnv,mf.transpose())
plt.plot(result)
plt.plot(comp)
plt.figure()
plt.show()

# Perform linear regression to find the optimal MF,
# showing that it is equal to RRC
x, _residuals, _rank, _s = np.linalg.lstsq(cnv, comp)
plt.plot(x/np.sum(np.abs(x)))
plt.plot(mf/np.sum(np.abs(mf)))
plt.legend(["Estimate", "Actual"])
plt.show()
