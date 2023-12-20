import torch
import torch

def print_tensor_add(ts):
    for i in range(ts.size()[0]):
        for j in range(ts.size()[1]):
                print(ts[i][j].data_ptr(), end=' ')

tensor = torch.tensor([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print(f"id：{id(tensor)}")
print(tensor)
print(tensor.is_contiguous())
print_tensor_add(tensor)
print("\n")
tensor = tensor.transpose(1, 0)
print(f"id：{id(tensor)}")
print(tensor)
print(tensor.is_contiguous())
print_tensor_add(tensor)




