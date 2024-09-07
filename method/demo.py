import torch
import torch.nn as nn

if __name__ == '__main__':
    indices = torch.tensor([[4, 2, 1], [2, 0, 2]])
    values = torch.tensor([3, 4, 5], dtype=torch.float32)
    x = torch.sparse_coo_tensor(indices=indices, values=values, size=[5, 5])
    x_dense = x.to_dense()
    x_dense = x_dense.unsqueeze(1)
    multihead_attn = nn.MultiheadAttention(5, 1)
    attn_output, attn_output_weights = multihead_attn(x_dense, x_dense, x_dense)
    attn_output = attn_output.squeeze(1)
    attn_output = nn.Linear(5, 5)(attn_output)
    attn_output = nn.Softmax(dim=1)(attn_output)
    attn_output = attn_output.squeeze(1)
    mask = attn_output > 0.2
    indices = torch.nonzero(mask).T

    for i in range(indices.shape[1]):
        new_node = attn_output[indices[0, i]] + attn_output[indices[1, i]]

    print(x_dense.squeeze(1)[mask])


