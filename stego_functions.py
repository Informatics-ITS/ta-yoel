import numpy as np
import random
import cv2
from typing import Tuple, List
import reedsolo
from random import randint
from math import log10, sqrt
import matplotlib.pyplot as plt

def map_string_to_numbers(text: str) -> list[int]:
    result = []
    for char in text:
        if char.isdigit():
            result.append(int(char))
        elif char.isalpha():
            # Konversi ke lowercase dan hitung offset dari 'a'
            result.append(ord(char.lower()) - ord('a') + 10)
    return result

def sort_revert(listn, indices_list = None):
    if indices_list is None:
        indices_list = list(range(len(listn)))
    sorted_indices = sorted(range(len(listn)), key=lambda i: listn[i])
    return (
        [listn[i] for i in sorted_indices],
        [indices_list[i] for i in sorted_indices]
    )

def revert_list(listn, indices_list):
    sorted_indices = sorted(range(len(indices_list)), key=lambda i: indices_list[i])
    sorted_first_list = [listn[i] for i in sorted_indices]
    return sorted_first_list

def to_binary(num, size):
    binary_list = []
    for _ in range(size):
        binary_list.append(0)

    index = size - 1
    while num > 0:
        binary_list[index] = num % 2
        num //= 2
        index -= 1

    return binary_list

def to_decimal(binary_list):
    binary_list = np.array(binary_list)
    decimal_value = 0
    for i, digit in enumerate(binary_list):
        if digit not in (0, 1):
            raise ValueError("Invalid binary digit: {}".format(digit))
        decimal_value += digit * (2 ** (len(binary_list) - i - 1))
    return decimal_value

def load_secret(secret):
    with open(secret, "r") as a:
        data = a.read()
        data = data.replace("\t", "")
        data = data.replace("\n", "")

    # secret = list(map(int, data))
    # print(len(secret), "{:.2f}kb".format(bitToKb(len(secret) * 2)))
    return map_string_to_numbers(data)

def flip_lsb(num):
    return num ^ 1

def spread_array(arr: list[int], n: int):
    seed = random.randint(0, 2**32 - 1)  # Tetapkan seed acak untuk reproducibility
    random.seed(seed)

    o_list = list(range(0, 2**n))
    random.shuffle(o_list)

    res = []
    for i in arr:
        res.append(o_list.pop(i % len(o_list)))

    return res, seed

def restore_array(spread_values, seed, n):
    random.seed(seed)

    o_list = list(range(0, 2**n))
    random.shuffle(o_list)

    restore = []
    for i in spread_values:
        ind = o_list.index(i)
        restore.append(ind)
        o_list.pop(ind)
        

    return restore

# def spread_array(arr: list[int], n: int):
#     """
#     Menyebarkan array dalam range 2^n
#     :param arr: List nilai asli
#     :param n: Pangkat dari 2^n
#     :return: List nilai tersebar dan seed untuk pengembalian
#     """
#     seed = random.randint(0, 2**32 - 1)  # Tetapkan seed acak untuk reproducibility
#     random.seed(seed)
    
#     max_range = 2**n
#     spread_values = [(a + random.randint(1, max_range - 1)) % max_range for a in arr]
#     return spread_values, seed

# def restore_array(spread_values, seed, n):
#     """
#     Mengembalikan array ke nilai asli
#     :param spread_values: List nilai yang sudah disebar
#     :param seed: Seed yang digunakan untuk penyebaran
#     :param original_length: Panjang array asli
#     :return: Array yang dikembalikan ke posisi asli (dummy nilai asli)
#     """
#     random.seed(seed)  # Set seed yang sama untuk reverse
#     max_range = 2**n
#     # Regenerate the same random numbers (this mimics the original array)
#     spread_random = np.array([random.randint(1, 2**n - 1) for i in range(len(spread_values))])
#     return list(map(lambda x: x[0] - x[1] + (max_range if x[0] < x[1] else 0), zip(spread_values, spread_random)))

def PSNR(original, compressed): 
    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                # Therefore PSNR have no importance. 
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr 

def MSE(original, compressed):
    mse = np.mean((original - compressed) ** 2) 
    return mse

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()  


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def image_to_byte(image):
    # Step 1: Extract the least significant bit (LSB) from each index
    lsb_bits = image.astype(np.uint8) & 1
    lsb_bits = (lsb_bits).astype(np.uint8)

    # Step 2: Convert the LSBs into an 8-bit byte array
    # Ensure the number of bits is a multiple of 8
    if len(lsb_bits) % 8 != 0:
        lsb_bits = np.pad(lsb_bits, (0, 8 - len(lsb_bits) % 8), mode='constant')

    # Convert the bits into bytes
    return np.packbits(lsb_bits).tobytes()


def embedding(cover: str, secret: str):
    cover_image = cv2.imread(cover)
    secret_data = load_secret(secret)

    cover_image = cv2.cvtColor(cover_image, cv2.COLOR_BGR2GRAY)

    n = int(np.ceil(np.log2(cover_image.shape[0] * cover_image.shape[1])))

    s_sort, s_ind = sort_revert(secret_data)
    s_sort = np.array(s_sort)

    _, q = np.unique(secret_data, return_counts=True)
    fa, seed = spread_array(q, n)
    fa_sort, fa_ind = sort_revert(fa)
    fa_sort = np.array(fa_sort)

    embedded_image = cover_image.copy().flatten().astype(np.uint8)

    for i in fa:
        embedded_image[i] = flip_lsb(embedded_image[i])

    MAX_PARITY = 128
    rs254 = reedsolo.RSCodec(MAX_PARITY)

    # Convert embedded_image to bytes
    cover_bytes = image_to_byte(cover_image)

    encoded_bytes = bytearray()
    parity_array = bytearray()
    for i in range(0, len(cover_bytes), MAX_PARITY//2):
        chunk = cover_bytes[i:i+MAX_PARITY//2]
        
        if len(chunk) < MAX_PARITY//2:
            rs = reedsolo.RSCodec(len(chunk) * 2)
            encoded_chunk = rs.encode(chunk)
            encoded_bytes.extend(encoded_chunk)
        else:
            encoded_chunk = rs254.encode(chunk)
            encoded_bytes.extend(encoded_chunk)

        parity_key = encoded_chunk[len(chunk):]
        parity_array.extend(parity_key)

    return [seed, s_ind, fa_ind, parity_array], embedded_image

def extraction(STEGO, KEY1, KEY2, KEY3, KEY4):
    MAX_PARITY = 128
    rs254 = reedsolo.RSCodec(MAX_PARITY)

    n = int(np.ceil(np.log2(STEGO.shape[0] * STEGO.shape[1])))
    # Convert embedded_image to bytes
    stego_bytes = image_to_byte(STEGO)

    reconstructed_cover_bytes = bytes()
    for i in range(0, len(KEY2), MAX_PARITY):
        parity = KEY4[i:i+MAX_PARITY]

        block = stego_bytes[i//2:i//2+MAX_PARITY//2] + parity

        if len(parity) < MAX_PARITY:
            rs = reedsolo.RSCodec(len(parity))
            decoded = rs.decode(block)[0]
            reconstructed_cover_bytes += decoded[:len(parity)//2]
        else:
            decoded = rs254.decode(block)[0]
            reconstructed_cover_bytes += decoded[:MAX_PARITY//2]
        
    stego = np.frombuffer(stego_bytes, dtype=np.uint8)
    r_cover = np.frombuffer(reconstructed_cover_bytes, dtype=np.uint8)

    # take the non-zero indices of the difference
    diff = np.bitwise_xor(stego, r_cover)
    non_zero_indices = np.where(diff != 0)[0]

    # convert it into binary
    secret_indices = np.array([])
    for i, v in enumerate(diff[non_zero_indices]):
        binary = np.array(to_binary(v, 8))
        idx = np.where(binary == 1)[0] + (non_zero_indices[i] * 8) # 8 is bit
        secret_indices = np.append(secret_indices, idx)

    recovered_cover = STEGO.copy().flatten().astype(np.uint8)

    for i in secret_indices:
        recovered_cover[i] = flip_lsb(recovered_cover[i])

    f_revert_ori= revert_list(secret_indices, KEY3)
    fa_restore = restore_array(f_revert_ori, KEY1, n)
    s_sorted_extract = np.concatenate([ [idx] * int(i) for idx, i in enumerate(fa_restore) ])

    recovered_secret_data = np.array(revert_list(s_sorted_extract, KEY2))

    return recovered_cover, recovered_secret_data
