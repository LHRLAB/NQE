import torch
import torch.nn as nn
import torch.nn.functional as F

# BetaE
class Regularizer():
    def __init__(self, base_add, min_val, max_val):
        self.base_add = base_add
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, entity_embedding):
        return torch.clamp(entity_embedding + self.base_add, self.min_val, self.max_val)


class SigmoidRegularizer(nn.Module):
    def __init__(self, vector_dim, dual=False):
        """
        :param dual: Split each embedding into 2 chunks.
                     The first chunk is property values and the second is property weight.
                     Do NOT sigmoid the second chunk.
        """
        super(SigmoidRegularizer, self).__init__()
        self.vector_dim = vector_dim
        # initialize weight as 8 and bias as -4, so that 0~1 input still mostly falls in 0~1
        self.weight = nn.Parameter(torch.Tensor([8]))
        self.bias = nn.Parameter(torch.Tensor([-4]))

        self.dual = dual

    def __call__(self, entity_embedding):
        if not self.dual:
            return torch.sigmoid(entity_embedding * self.weight + self.bias)
        else:
            # The first half is property values and the second is property weight.
            # Do NOT sigmoid the second chunk. The second chunk will be free parameters
            entity_vals, entity_val_weights = torch.chunk(entity_embedding, 2, dim=-1)
            entity_vals = torch.sigmoid(entity_vals * self.weight + self.bias)
            return torch.cat((entity_vals, entity_val_weights), dim=-1)


    def soft_discretize(self, entity_embedding, temperature=10):
        return torch.sigmoid((entity_embedding * self.weight + self.bias)*temperature)  # soft

    def hard_discretize(self, entity_embedding, temperature=10, thres=0.5):
        discrete = self.soft_discretize(entity_embedding, temperature)
        discrete[discrete>=thres] = 1
        discrete[discrete<thres] = 0
        return discrete


class MatrixSoftmaxRegularizer(nn.Module):
    def __init__(self, vector_dim, k):
        """
        :param k: split the vector into matrix, k elements per row. k has to be a factor of vector_dim
        """
        super(MatrixSoftmaxRegularizer, self).__init__()
        self.vector_dim = vector_dim
        self.k = k
        self.softmax = nn.Softmax(dim=-1)
        self.softmax_weight = nn.Parameter(torch.full((self.vector_dim,), fill_value=0.1))
        self.softmax_bias = nn.Parameter(torch.Tensor([0]))

    def __call__(self, entity_embedding):
        """
        :param entity_embedding: shape [batch_size, dim]
        """
        transformed = entity_embedding * self.softmax_weight + self.softmax_bias

        # reshape the last dimension into a matrix
        dims, last_dim = entity_embedding.size()[:-1], entity_embedding.size()[-1]
        n_row = last_dim//self.k
        n_col = self.k

        reshaped = transformed.view(*dims, n_row, n_col)
        reshaped = self.softmax(reshaped)  # softmax along the last dimension
        return reshaped.view(*dims, last_dim)  # change to original shape

    def reshape_to_matrix(self, entity_embedding):
        # reshape the last dimension into a matrix
        dims, last_dim = entity_embedding.size()[:-1], entity_embedding.size()[-1]
        n_row = last_dim//self.k
        n_col = self.k

        reshaped = entity_embedding.view(*dims, n_row, n_col)
        return reshaped

    def reshape_to_vector(self, entity_embedding_matrix):
        dims, n_row, n_col = entity_embedding_matrix.size()[:-2], entity_embedding_matrix.size()[-1], entity_embedding_matrix.size()[-2]
        last_dim = n_row*n_col
        return entity_embedding_matrix.view(*dims, last_dim)


class VectorSoftmaxRegularizer(nn.Module):
    def __init__(self, vector_dim):
        """
        :param k: split the vector into matrix, k elements per row. k has to be a factor of vector_dim
        """
        super(VectorSoftmaxRegularizer, self).__init__()
        self.vector_dim = vector_dim
        self.softmax = nn.Softmax(dim=-1)
        # self.softmax_weight = nn.Parameter(torch.Tensor([1]))
        # self.softmax_bias = nn.Parameter(torch.Tensor([0]))

        # a fixed small weight is much better than learnable weight
        self.softmax_weight = 0.01
        self.softmax_bias = 0

    def __call__(self, entity_embedding):
        """
        :param entity_embedding: shape [batch_size, dim]
        """
        # softmax along the last dimension
        entity_embedding = entity_embedding * self.softmax_weight + self.softmax_bias
        return self.softmax(entity_embedding)



class VectorSigmoidSumRegularizer(nn.Module):
    def __init__(self, vector_dim, neg_input_possible=False, use_layernorm=False):
        super(VectorSigmoidSumRegularizer, self).__init__()

        self.use_layernorm = use_layernorm
        if self.use_layernorm:
            # applies layernorm before sigmoid
            # initialize weight as 8 and bias as -4, so that 0~1 input still mostly falls in 0~1
            self.weight = nn.Parameter(torch.Tensor([5]), requires_grad=True)
            self.bias = nn.Parameter(torch.Tensor([0]), requires_grad=True)

            self.layernorm = nn.LayerNorm(vector_dim, elementwise_affine=False)

        else:
            # initialize weight as 8 and bias as -4, so that 0~1 input still mostly falls in 0~1
            self.weight = nn.Parameter(torch.Tensor([8]), requires_grad=False)
            self.bias = nn.Parameter(torch.Tensor([-4]), requires_grad=False)
            
        

    def forward(self, embeddings):
        """
        :param embeddings: shape [batch_size, dim]
        """
        x = embeddings
        if self.use_layernorm:
            x = self.layernorm(x)
        x = torch.sigmoid(x * self.weight + self.bias)  # shift to non-negative
        x = F.normalize(x, p=1, dim=-1)  # L1 normalize along the last dimension
        return x



class MatrixSumRegularizer(nn.Module):
    """
    For set representation (computation graph node representation)
    """
    def __init__(self, vector_dim, k, neg_input_possible=False):
        """
        :param k: split the vector into matrix, k elements per row. k has to be a factor of vector_dim
        """
        super(MatrixSumRegularizer, self).__init__()
        self.vector_dim = vector_dim
        self.k = k
        self.neg_input_possible = neg_input_possible  # True for entity regularizer, False for set regularizer

    def forward(self, embeddings):
        """
        :param embeddings: shape [batch_size, dim]
        """
        # reshape the last dimension into a matrix
        dims, last_dim = embeddings.size()[:-1], embeddings.size()[-1]
        n_row = last_dim//self.k
        n_col = self.k

        reshaped = embeddings.view(*dims, n_row, n_col)

        if self.neg_input_possible:
            # shift to non-negative
            reshaped = torch.relu(reshaped)

            # min_per_row, _ = torch.min(reshaped, dim=-1,keepdim=True)
            # min_per_row[min_per_row>=0] = 0  # if min_per_row is positive, no need to shift
            # reshaped -= min_per_row  # shift by the minimum negative value

        # L1 normalize
        reshaped = F.normalize(reshaped, p=1, dim=-1)  # L1 normalize along the last dimension
        reshaped = reshaped.view(*dims, last_dim)  # change to original shape
        return reshaped

    def reshape_to_matrix(self, embeddings):
        # reshape the last dimension into a matrix
        dims, last_dim = embeddings.size()[:-1], embeddings.size()[-1]
        n_row = last_dim//self.k
        n_col = self.k

        reshaped = embeddings.view(*dims, n_row, n_col)
        return reshaped

    def reshape_to_vector(self, embeddings_matrix):
        dims, n_row, n_col = embeddings_matrix.size()[:-2], embeddings_matrix.size()[-2], embeddings_matrix.size()[-1]
        last_dim = n_row*n_col
        return embeddings_matrix.view(*dims, last_dim)

    def hard_discretize(self, embeddings):
        """
        Discretize as a matrix. k entries per row => one '1' per row.
        No gradient.
        No normalization added. (not needed)
        :param embeddings: shape [batch_size, 1 or num_neg, entity_dim], 0<=embeddings[i]<=1
        :return y_hard: [batch_size, 1 or num_neg, entity_dim]
        """
        y = self.reshape_to_matrix(embeddings)
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)  # shape [*dims, entity_dim//k, k]
        y_hard = self.reshape_to_vector(y_hard)
        return y_hard

    def soft_discretize(self, embeddings, gumbel_temperature):
        """
        Discretize as a matrix. k entries per row => one '1' per row.
        Soft discretize using Gumbel softmax.
        No normalization added. (not needed)
        :param embeddings: shape [batch_size, 1 or num_neg, entity_dim], 0<=embeddings[i]<=1
        :param gumbel_temperature: max(0.5, exp(-rt)), r={1e-4, 1e-5}
        :return y_hard: [batch_size, 1 or num_neg, entity_dim]
        """
        y = self.reshape_to_matrix(embeddings)
        eps = 1e-5
        log_y = torch.log(y+eps)
        y_soft = F.gumbel_softmax(log_y, tau=gumbel_temperature, hard=False)
        y_soft = self.reshape_to_vector(y_soft)
        return y_soft

    def L1_normalize(self, embeddings):
        """
        :param embeddings: shape [batch_size, dim]
        :return: shape [batch_size, dim]
        """
        k = self.k
        # reshape the last dimension into a matrix
        dims, last_dim = embeddings.size()[:-1], embeddings.size()[-1]
        n_row = last_dim // k
        n_col = k

        reshaped = embeddings.view(*dims, n_row, n_col)

        # L1 normalize
        reshaped = F.normalize(reshaped, p=1, dim=-1)  # L1 normalize along the last dimension
        reshaped = reshaped.view(*dims, last_dim)  # change to original shape
        return reshaped

    def get_num_distributions(self):
        return self.vector_dim // self.k



class MatrixSigmoidSumRegularizer(MatrixSumRegularizer):
    def __init__(self, vector_dim, k, neg_input_possible=False):
        super(MatrixSigmoidSumRegularizer, self).__init__(vector_dim, k, neg_input_possible)
        # initialize weight as 8 and bias as -4, so that 0~1 input still mostly falls in 0~1
        self.weight = nn.Parameter(torch.Tensor([1]))
        self.bias = nn.Parameter(torch.Tensor([0]))

    def forward(self, embeddings):
        """
        :param embeddings: shape [batch_size, dim]
        """
        # reshape the last dimension into a matrix
        dims, last_dim = embeddings.size()[:-1], embeddings.size()[-1]
        n_row = last_dim//self.k
        n_col = self.k

        reshaped = embeddings.view(*dims, n_row, n_col)

        if self.neg_input_possible:  # for entity free parameters
            # shift to non-negative
            reshaped = torch.sigmoid(reshaped * self.weight + self.bias)

        # L1 normalize
        reshaped = F.normalize(reshaped, p=1, dim=-1)  # L1 normalize along the last dimension
        reshaped = reshaped.view(*dims, last_dim)  # change to original shape
        return reshaped
