# Gabor feature extraction will not use integral image
import numpy as np
import math
import cmath
import matplotlib.pyplot as plt
import scipy.ndimage as sim
import os

class Gabor_Features():
    def __init__(self,image):
        # Extract size of image
        self.shape = image.shape
        self.image = image

    def next_power_of_2(self,x):
        return 1 if x == 0 else 2 ** (x - 1).bit_length()
    # Generate Kernel
    def gen_gabor_kernel(self,mu,nu):
        self.mu = mu
        self.nu = nu
        kmax = math.pi/2
        f = math.sqrt(2)
        angstep = math.pi/8
        th = .005
        sigma = 2*math.pi
        val1 = th*sigma**2
        val2 = kmax**2
        self.scaleXY = math.ceil(math.sqrt(-math.log(val1/val2)*2*sigma**2/val2))
        [X,Y] = np.meshgrid(np.linspace(-self.scaleXY,self.scaleXY,num=2*self.scaleXY+1),np.linspace(-self.scaleXY,self.scaleXY,num=2*self.scaleXY+1))
        realSz = self.next_power_of_2(max(self.shape)+self.scaleXY*2)
        self.kernel = np.zeros((len(nu), len(mu), realSz, realSz), dtype="complex128")
        for scale_idx in range(len(nu)):
            for angle_idx in range(len(mu)):
                phi = angstep*mu[angle_idx]
                k = kmax/f**nu[scale_idx]
                scale = (k/sigma)**2
                arr1 = np.exp(-k**2*(np.square(X)+np.square(Y))/2/sigma**2)
                arr2 = np.exp(1j*k*cmath.cos(phi)*X+k*cmath.sin(phi)*Y)
                # Apply fourier fast transform
                self.kernel[scale_idx, angle_idx, :, :] = np.fft.fft2(scale*arr1*arr2,[realSz,realSz])
                self.kernel[scale_idx, angle_idx, 0, 0] = 0
    def gabor_conv(self):
        # Multiply FFT of Image and kernel to get conv images
        f = np.pad(self.image,[self.scaleXY,self.scaleXY],'reflect')
        realSz = self.next_power_of_2(max(self.shape) + self.scaleXY * 2)
        fimg = np.fft.fft2(f, [realSz, realSz])
        self.Gimg = np.zeros((len(nu), len(mu), self.shape[0], self.shape[1]))
        for scale_iter in range(len(self.nu)):
            for orient_iter in range(len(self.mu)):
                filtered = np.fft.ifft2(fimg*self.kernel[scale_iter][orient_iter])
                cropped = filtered[self.scaleXY*2:self.shape[0]+self.scaleXY*2,self.scaleXY*2:self.shape[1]+self.scaleXY*2]
                self.Gimg[scale_iter,orient_iter,:,:] = np.absolute(cropped)

if __name__ == "__main__":
    pwd = os.getcwd()
    image = sim.imread(pwd+"\\data\\face\\01.jpg")
    gray_image = .21*image[:,:,0] + .72*image[:,:,1] + .07*image[:,:,2]
    gray_image = gray_image.astype(int)
    plt.imshow(gray_image,cmap='gray')
    plt.show()

    mu = [0, 1, 2, 3, 4, 5, 6, 7]
    nu = [0, 1, 2, 3, 4]
    feat = Gabor_Features(gray_image)
    feat.gen_gabor_kernel(mu,nu)
    feat.gabor_conv()
    plt.imshow(feat.Gimg[0][0],cmap="gray")
    plt.show()
    print("debug")