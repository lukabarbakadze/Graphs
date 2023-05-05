import torch
from torch import nn
import torch.nn.functional as F

class GCNetwork(nn.Module):
    def __init__(self, 
                 in_features, 
                 out_features, 
                 n_classes, 
                 bias=True, 
                 num_layers=1):
        super().__init__()
        """
        in_features :   number of columns in input matrix X,
        out_features:   desired number of embedding/output variables,
        n_classes   :   number of target classes,
        bias        :   whether linear layers should have bias term or not,
        num_layers  :   if num_layers=2 information of neighbor's neighbors will be aggregated in a edges too
        """
        self.num_layers = num_layers
        self.W1 = nn.Linear(in_features, out_features, bias=bias)
        if num_layers==2:
            self.W2 = nn.Linear(out_features, out_features, bias=bias)
        self.FL = nn.Linear(out_features, n_classes)

    def block(self, x, A_tilda, W):
        """
        --input
        x       : feature matrix size of (m, n) where m is number of edges and n is number of features,
        A_tilda : regularization term which scales values of new embeddings based on number of connections,
        W       : linear layer on to project input x
        --output
        x       : features matrix with size (number_of_edges, out_features) or simply new embedding matrix
        """
        x = A_tilda @ x
        x = W(x)
        x = F.relu(x)
        return x
    
    def forward(self, x, A):
        """
        --input
        x   : feature matrix size of (m, n) where m is number of edges and n is number of features,
        A   : adjacency matrix with size (m, m) which captures connections between edges
        --output
        x   : last features space with size of (m, out_features)
        out : logits (without softmax) with size of (m, n_classes)
        """
        A_tilda = torch.diag(torch.rsqrt(A.sum(axis=0))) @ A @ torch.diag(torch.rsqrt(A.sum(axis=0))) # (number_of_edges, number_of_edges)
        x = self.block(x, A_tilda, self.W1) # (number_of_edges, out_features)
        if self.num_layers==2:
            x = self.block(x, A_tilda, self.W2) # (number_of_edges, out_features)
        out = self.FL(x) # (number_of_edges, n_classes)

        return x, out

class GANetwork(nn.Module):
    def __init__(self, 
                 in_features, 
                 out_features,
                 n_classes,
                 num_heads=4,
                 alpha=0.2,
                 bias=True):
        super().__init__()
        """
        in_features :   number of columns in input matrix X,
        out_features:   desired number of embedding/output variables,
        n_classes   :   number of target classes,
        num_heads   :   number of self-attention heads in a single attention layer,
        bias        :   whether linear layers should have bias term or not,
        alpha       :   negative slope coefficient for leaky_relu activation function
        """
        self.n_classes = n_classes
        self.num_heads = num_heads
        self.alpha = alpha

        self.W1 = nn.Linear(in_features, out_features * num_heads, bias=bias)
        self.a1 = nn.Parameter(torch.Tensor(num_heads, 2 * out_features))

        self.W2 = nn.Linear(out_features, out_features * num_heads)
        self.a2 = nn.Parameter(torch.Tensor(num_heads, 2 * out_features))

        self.W3 = nn.Linear(out_features * num_heads, out_features)
        self.FL = nn.Linear(out_features * num_heads, n_classes)

        nn.init.xavier_uniform_(self.a1.data, gain=1.414)
        nn.init.xavier_uniform_(self.a2.data, gain=1.414)
    
    def attention_block(self, h, A, W, a):
        """
        --input
        h  :   feature matrix size of (m, n) where m is number of edges and n is number of features,
        A  :   adjacency matrix with size (m, m) which captures connections between edges,
        W  :   linear layer on to project input (outputs (m, out_features)),
        a  :   weights for self-attention
        --output
        Wh :   new embeding matrix with size of (batch_size, num_nodes, out_features * num_heads)
        """
        # save batch size & number of features
        batch_size, num_nodes = h.size(0), h.size(1)

        # project X to Matrix W
        Wh = W(h)

        # separate each head via separate dimension
        Wh = Wh.view(batch_size, num_nodes, self.num_heads, -1)

        # get edge matrix from A
        edges = A.nonzero(as_tuple=False)

        # remove batch dimension
        Wh_flat_flat = Wh.view(batch_size * num_nodes, self.num_heads, -1)

        # select indices of connected nodes
        edge_indices_row = edges[:, 0] * num_nodes + edges[:, 1]
        edge_indices_col = edges[:, 0] * num_nodes + edges[:, 2]

        # select corresponding projected vectors from Wh_flat_flat Matrix (for connected pairs)
        a_input = torch.cat(
            [
                torch.index_select(input=Wh_flat_flat, index=edge_indices_row, dim=0),
                torch.index_select(input=Wh_flat_flat, index=edge_indices_col, dim=0),
            ],
            dim=-1,
        )
        # calculate attention logits & aplly Leaky_relu activation function
        attn_logits = torch.einsum("bhc,hc->bh", a_input, a)
        attn_logits = F.leaky_relu(attn_logits)

        # replace 0's by '-inf' as features with 0 value should not have attention power (softmax will nullify it)
        attn_matrix = attn_logits.new_zeros(A.shape + (self.num_heads,)).fill_(float('-inf'))
        attn_matrix[A.view(1, A.size(1), A.size(2), 1).repeat(1, 1, 1, self.num_heads) == 1] = attn_logits.reshape(-1)

        # apply softmax to calculate attention scores
        attn_probs = F.softmax(attn_matrix, dim=2)

        # use torch.einsum to get updated embeddings with size (batch_size, num_nodes, out_features * num_heads)
        Wh = torch.einsum("bijh,bjhc->bihc", attn_probs, Wh)
        Wh = Wh.reshape(batch_size, num_nodes, -1)
        return Wh

    def forward(self, h, A):
        """
        --input
        h   : feature matrix size of (b,m,n) where m is number of edges and n is number of features and b batch_size,
        A   : adjacency matrix with size (b,m,m) which captures connections between edges
        --output
        h   : last features space with size of (b,m,out_features*num_heads)
        out : logits (without softmax) with size of (b,m,n_classes)
        """
        h = self.attention_block(h, A, self.W1, self.a1) 
        h = self.W3(h)
        h = self.attention_block(h, A, self.W2, self.a2)
        out = self.FL(h)
        return h, out