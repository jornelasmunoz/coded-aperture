import numpy as np
import matplotlib.pyplot as plt
# from scipy.fft import fft2, ifft2
import torch
from torch.fft import fft2, ifft2

# def load_data():
#     # Load encoded data 
#     filename_train = "../data/training_MNIST_mura"
#     filename_eval  = "../data/validation_MNIST_mura"
#     filename_test  = "../data/testing_MNIST_mura"

#     mura_train_data = torch.load(filename_train)
#     mura_eval_data = torch.load(filename_eval)
#     mura_test_data = torch.load(filename_test)
#     print(f"Number of elements in each dataset \nTraining: {len(mura_train_data)} \nValidation: {len(mura_eval_data)} \nTesting: {len(mura_test_data)}")
    
#     return mura_train_data, mura_eval_data, mura_test_data

def create_binary_aperture_arr(p):
    '''
    Inputs
        p: int. prime integer
        
    Output
        A: torch.Tensor. Binary aperture array 
    '''
    A = torch.zeros((p,p)) # binary aperture array
    # Aperture function p. 4350 in Gottesman and Fenimore (1989)
    for i in range(0,p):
        for j in range(0,p):
            C_i = legendre_symbol(i,p)
            C_j = legendre_symbol(j,p)

            if i == 0:
                A[i,j] = 0
            elif (j == 0 and i != 0):
                A[i,j] = 1
            elif (C_i * C_j) == 1:
                A[i,j] = 1
            else:
                A[i,j] = 0
    return torch.Tensor(A)

def create_decoding_arr(A):
    '''
    Inputs
        A: np.array. Binary aperture array
        
    Output
        G: np.array of same size as A. Decoding function 
    '''
    
    G = torch.zeros_like(A) # initialize decoding function
    p = G.shape[0]
    
    # Decoding function p. 4350 in Gottesman and Fenimore (1989)
    for i in range(0,p):
        for j in range(0,p):

            if (i + j) == 0:
                G[i,j] = 1
            elif (A[i,j] == 1 and (i + j) != 0):
                G[i,j] = 1
            elif (A[i,j] == 0 and (i + j) != 0):
                G[i,j] = -1
                
                
    return torch.Tensor(G)

def FFT_convolve(A, B, p=None):
    '''
    Compute convolution using FFT between A and B
    Inputs: 
        A,B: np.ndarrays
        
    Outputs:
        conv_AB: nd.array. Convolution between A and B
    '''
    # Check A and B are the same size
    if torch.Tensor(A).shape != torch.Tensor(B).shape:
        print(A.shape, B.shape)
        raise Exception("The arrays A and B are not the same shape")
    
    # Define p if it is not given already
    if p is None:
        p = A.shape[0]
    # Do convolution via FFT   
    fft_A = fft2(A)
    fft_B = fft2(B)
    conv_AB = torch.real(ifft2(torch.multiply(fft_A,fft_B)))
    conv_AB = torch.roll(conv_AB, [int((p-1)/2),int((p-1)/2)], dims=(0,1))
    # conv_AB = torch.roll(conv_AB, [int((p+1)/2),int((p+1)/2)], dims=(0,1))
    # conv_AB = torch.roll(conv_AB, [int((p)/2),int((p)/2)], dims=(0,1))
    return conv_AB

def add_Gaussian_noise(og_image, desired_snr):
    '''
    Given an image and a desired Signal to Noise Ratio (in decibels, dB)
    returns a noisy image
    
    Inputs:
        og_image: Tensor. Original (noiseless) image normalized to pixel values [0,1]
        desired_snr: Integer. Signal to noise ration in decibels
        
    Outputs: 
        noisy_image: Tensor. Noisy version of original image. Values are between 0 and 1.  
    '''
    # Calculate the variance of the image pixels
    signal_power = torch.var(og_image)

    # # Set the desired SNR
    # desired_snr = 30

    # Calculate the noise power
    noise_power = signal_power / (10**(desired_snr/10))

    # Generate random noise matrix
    noise = torch.normal(0,torch.sqrt(noise_power), size=og_image.shape)

    # Add the noise to the image
    noisy_image = og_image + noise

    noisy_image = torch.clip(noisy_image, 0, 1)#.astype(torch.float32)
    
    return noisy_image

def get_Gaussian_filter(image, mean=0, var=0.1):
    """
    Inputs:
        image: np.array of size [height, width]. Provides dimensions for filter
        mean:  Mean for Gaussian distribution
        var:   Variance for Gaussian distribution
        
    Outputs:
        noisy: Gaussian filter
    """
    
    row,col = image.shape
    
    # Calculate Gaussian filter
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col))
    gauss = gauss.reshape(row,col)
    
    return gauss    
    
# --------- HELPER FUNCTIONS
                
# copied from https://eli.thegreenplace.net/2009/03/07/computing-modular-square-roots-in-python on 10/05/22
def legendre_symbol(a, p):
    """ Compute the Legendre symbol a|p using
        Euler's criterion. p is a prime, a is
        relatively prime to p (if p divides
        a, then a|p = 0)

        Returns 1 if a has a square root modulo
        p, -1 otherwise.
    """
    ls = pow(a, (p - 1) // 2, p)
    return -1 if ls == p - 1 else ls


# normalize 
def normalize(data, a=0, b=1):
    '''
    Normalize data between [a,b]
    '''
    # data tensor
    normalized_data = a + ((b-a)*(data-data.min()))/(data.max()-data.min())
    return normalized_data

def get_D(x):
    return torch.unsqueeze(torch.tensor(FFT_convolve(np.squeeze(x.numpy()), A,p)), 0)