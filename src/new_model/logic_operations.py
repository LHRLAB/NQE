import torch
import torch.nn as nn
import torch.nn.functional as F
from new_model.util import get_regularizer

class Projection(nn.Module):
    # def __init__(self, entity_dim, logic_type, regularizer_setting):
    def __init__(
            self,
            nrelation,
            entity_dim,
            logic_type,
            regularizer_setting,
            relation_dim,
            projection_dim,
            num_layers,
            projection_type,
            num_rel_base,  # for 'rtransform'
    ):
        super(Projection, self).__init__()
        self.logic = logic_type

        self.regularizer = get_regularizer(regularizer_setting, entity_dim, neg_input_possible=True)
        # for projection
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.projection_type = projection_type

        self.dual = regularizer_setting['dual']


        # mlp
        if projection_type == 'mlp':
            self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.entity_dim))  # same dim
            nn.init.uniform_(tensor=self.relation_embedding, a=0, b=1)

            # mlp
            self.hidden_dim = projection_dim
            self.num_layers = num_layers
            self.layer1 = nn.Linear(self.entity_dim + self.relation_dim, self.hidden_dim)  # 1st layer
            self.layer0 = nn.Linear(self.hidden_dim, self.entity_dim)  # final layer
            for nl in range(2, num_layers + 1):
                setattr(self, "layer{}".format(nl), nn.Linear(self.hidden_dim, self.hidden_dim))
            for nl in range(num_layers + 1):
                nn.init.xavier_uniform_(getattr(self, "layer{}".format(nl)).weight)
        elif projection_type == 'rtransform':
            n_base = num_rel_base
            if not self.dual:
                self.hidden_dim = entity_dim
                self.rel_base = nn.Parameter(torch.zeros(n_base, self.hidden_dim, self.hidden_dim))
                # nn.init.uniform_(self.rel_base, a=0, b=1e-2)
                self.rel_bias = nn.Parameter(torch.zeros(n_base, self.hidden_dim))
                self.rel_att = nn.Parameter(torch.zeros(nrelation, n_base))
                self.norm = nn.LayerNorm(self.hidden_dim, elementwise_affine=False)

                # new initialization
                torch.nn.init.orthogonal_(self.rel_base)
                torch.nn.init.xavier_normal_(self.rel_bias)
                torch.nn.init.xavier_normal_(self.rel_att)

            else:
                self.hidden_dim = entity_dim//2

                # for property vals
                self.rel_base1 = nn.Parameter(torch.randn(n_base, self.hidden_dim, self.hidden_dim))
                nn.init.uniform_(self.rel_base1, a=0, b=1e-2)
                self.rel_bias1 = nn.Parameter(torch.zeros(nrelation, self.hidden_dim))
                self.rel_att1 = nn.Parameter(torch.randn(nrelation, n_base))
                self.norm1 = nn.LayerNorm(self.hidden_dim, elementwise_affine=False)

                # for property weights
                self.rel_base2 = nn.Parameter(torch.randn(n_base, self.hidden_dim, self.hidden_dim))
                nn.init.uniform_(self.rel_base2, a=0, b=1e-2)
                self.rel_bias2 = nn.Parameter(torch.zeros(nrelation, self.hidden_dim))
                self.rel_att2 = nn.Parameter(torch.randn(nrelation, n_base))
                self.norm2 = nn.LayerNorm(self.hidden_dim, elementwise_affine=False)
        elif projection_type == 'transe':
            self.hidden_dim = entity_dim
            self.rel_trans = nn.Parameter(torch.zeros(nrelation, self.hidden_dim))
            self.rel_bias = nn.Parameter(torch.zeros(nrelation, self.hidden_dim))
            torch.nn.init.xavier_normal_(self.rel_trans)
            torch.nn.init.xavier_normal_(self.rel_bias)
            self.norm = nn.LayerNorm(self.hidden_dim, elementwise_affine=False)



    def forward(self, e_embedding, rid):
        if self.projection_type == 'mlp':
            r_embedding = torch.index_select(self.relation_embedding, dim=0, index=rid)
            x = torch.cat([e_embedding, r_embedding], dim=-1)
            for nl in range(1, self.num_layers + 1):
                x = F.relu(getattr(self, "layer{}".format(nl))(x))
            x = self.layer0(x)
            x = self.regularizer(x)
            return x

        if self.projection_type == 'rtransform':
            if not self.dual:
                project_r = torch.einsum('br,rio->bio', self.rel_att[rid], self.rel_base)
                if self.rel_bias.shape[0] == self.rel_base.shape[0]:
                    bias = torch.einsum('br,ri->bi', self.rel_att[rid], self.rel_bias)
                else:
                    bias = self.rel_bias[rid]
                output = torch.einsum('bio,bi->bo', project_r, e_embedding) + bias
                output = self.norm(output)
            else:
                e_embedding1, e_embedding2 = torch.chunk(e_embedding, 2, dim=-1)
                project_r1 = torch.einsum('br,rio->bio', self.rel_att1[rid], self.rel_base1)
                bias1 = self.rel_bias1[rid]
                output1 = torch.einsum('bio,bi->bo', project_r1, e_embedding1) + bias1
                output1 = self.norm1(output1)

                project_r2 = torch.einsum('br,rio->bio', self.rel_att2[rid], self.rel_base2)
                bias2 = self.rel_bias2[rid]
                output2 = torch.einsum('bio,bi->bo', project_r2, e_embedding2) + bias2
                output2 = self.norm1(output2)

                output = torch.cat((output1, output2), dim=-1)

            output = self.regularizer(output)
            return output
        
        if self.projection_type == 'transe':
            r_trans = torch.index_select(self.rel_trans, dim=0, index=rid)
            r_bias = torch.index_select(self.rel_bias, dim=0, index=rid)
            output = e_embedding * r_trans + r_bias
            
            output = self.norm(output)
            output = self.regularizer(output)
            return output



class Conjunction(nn.Module):
    def __init__(self, entity_dim, logic_type, regularizer_setting, use_attention='False', godel_gumbel_beta=0.01):
        super(Conjunction, self).__init__()
        self.logic = logic_type
        self.regularizer = get_regularizer(regularizer_setting, entity_dim)
        self.use_attention = use_attention
        self.entity_dim = entity_dim

        self.norm = nn.LayerNorm(entity_dim, elementwise_affine=False)

    def forward(self, embeddings):
        """
        :param embeddings: shape (# of sets, batch, dim).
        :return embeddings: shape (batch, dim)
        """
        if self.logic == 'godel':
            if self.logic == 'godel':
                # conjunction(x,y) = min{x,y}
                embeddings, _ = torch.min(embeddings, dim=0)
            elif self.logic == 'godel_gumbel':
                # soft way to compute min
                embeddings = -self.godel_gumbel_beta * torch.logsumexp(
                    -embeddings / self.godel_gumbel_beta,
                    0
                )
            return embeddings
        else:  # logic == product
            if self.logic == 'luka':
                # conjunction(x,y) = max{0, x+y-1}
                embeddings = torch.sum(embeddings, dim=0) - embeddings.shape[0] + 1
            elif self.logic == 'product':
                if not self.use_attention:
                    # conjunction(x,y) = xy
                    embeddings = torch.prod(embeddings, dim=0)
                else:
                    attention = self.get_conjunction_attention(embeddings)
                    # attention conjunction(x,y) = (x^p)*(y^q), p+q=1
                    # compute in log scale
                    epsilon = 1e-7  # avoid torch.log(0)
                    embeddings = torch.log(embeddings+epsilon)
                    embeddings = torch.exp(torch.sum(embeddings * attention, dim=0))
            embeddings = self.norm(embeddings)
            return self.regularizer(embeddings)

    def get_conjunction_attention(self, embeddings):
        layer1_act = F.relu(self.conjunction_layer1(embeddings))  # (num_conj, batch_size, 2 * dim)
        attention = F.softmax(self.conjunction_layer2(layer1_act)/torch.sqrt(self.entity_dim), dim=0)  # (num_conj, batch_size, 1)
        return attention


class Disjunction(nn.Module):
    def __init__(self, entity_dim, logic_type, regularizer_setting, godel_gumbel_beta=0.01):
        super(Disjunction, self).__init__()
        self.logic = logic_type
        self.regularizer = get_regularizer(regularizer_setting, entity_dim)


        self.norm = nn.LayerNorm(entity_dim, elementwise_affine=False)


    def forward(self, embeddings):
        """
        :param embeddings: shape (# of sets, batch, dim).
        :return embeddings: shape (batch, dim)
        """
        if self.logic == 'godel':
            if self.logic == 'godel':
                # disjunction(x,y) = max{x,y}
                embeddings, _ = torch.max(embeddings, dim=0)
                return embeddings
            elif self.logic == 'godel_gumbel':
                # soft way to compute max
                embeddings = self.godel_gumbel_beta * torch.logsumexp(
                    embeddings / self.godel_gumbel_beta,
                    0
                )
            return embeddings
        else:
            if self.logic == 'luka':
                # disjunction(x,y) = min{1, x+y}
                embeddings = torch.sum(embeddings, dim=0)
            else:  # self.logic == 'product'
                # disjunction(x,y) = x+y-xy
                embeddings = torch.sum(embeddings, dim=0) - torch.prod(embeddings, dim=0)
            return self.regularizer(embeddings)


class Negation(nn.Module):
    def __init__(self, entity_dim, logic_type, regularizer_setting):
        super(Negation, self).__init__()
        self.logic = logic_type
        self.regularizer = get_regularizer(regularizer_setting, entity_dim)

    def forward(self, embeddings):
        """
        :param embeddings: shape (# of sets, batch, dim).
        :return embeddings: shape (batch, dim)
        """
        # negation(x) = 1-x
        return 1 - embeddings


