import faiss
import torch
import torchhd
from torch.fft import fft, ifft

class HDCModel:
    def __init__(self, dim: int,  )