import torch
import sys
sys.path.append('..')

# def collate(batch):
#     """A custom collate function for dealing with batches of features that have a different number of associated targets
#     (action instances).
#     """
#     max_len = max([len(feat) for feat,_,_ in batch])
#
#     features = []
#     targets = []
#     idxs = []
#
#     for feat, label, idx in batch:
#         features.append(feat)
#         targets.append(label)
#         idxs.append(idx)
#
#     return torch.stack(features, 0), targets, idxs

def pad_sequence(sequences, max_len, batch_first=False, padding_value=0.0):
    # type: (List[Tensor], bool, float) -> Tensor

    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    # print('max_size is {}'.format(max_size))
    trailing_dims = max_size[1:]
    if max_len == 0:
        max_len = max([s.size(0) for s in sequences])
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims
    # print('out_dims {}'.format(out_dims))
    out_tensor = sequences[0].new_full(out_dims, padding_value)
    # print('out_tensor.size() {}'.format(out_tensor.size()))
    for i, tensor in enumerate(sequences):
        length = min(tensor.size(0), max_len)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor[:length]
        else:
            out_tensor[:length, i, ...] = tensor[:length]

    return out_tensor


def collate_fn_padd(batch, max_len=0):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''

    ## padd
    features = []
    targets = []
    idxs = []
    lengths = []
    # max_len = 0
    # print(batch)
    for l, t, label, idx in batch:
        features.append(torch.Tensor(t))
        targets.append(label)
        idxs.append(idx)
        if max_len == 0:
            lengths.append(l)
        else:
            lengths.append(min(l, max_len))

    batch = pad_sequence(features, max_len, True)
    ## compute mask
    mask = (batch != 0).float()
    return batch, torch.tensor(targets), idxs, torch.tensor(lengths), mask[..., 0:1]