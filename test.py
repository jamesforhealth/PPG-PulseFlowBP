import vector_quantize_pytorch as vq
import torch

a = torch.FloatTensor([-0.1, 0.5, 0.2, 0.33, -0.6, 0.2]).view(1, 3, 2)
print('a=', a)

quantizer = vq.VectorQuantize(dim=2, codebook_size=6)

quantized, indices, loss = quantizer(a)
print('quantized', quantized)
print('indices', indices)
print('loss', loss)
