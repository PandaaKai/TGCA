import torch
from torch import nn
import torch.nn.functional as F
from utils import *
import method.layers as layers
from .kan_layer_fourier import NaiveFourierKANLayer as fft
from .kan import KANLinear as effKan, KAN

def _coo_scipy2torch(adj):
    """
    convert a scipy sparse COO matrix to torch
    """
    values = adj.data
    indices = np.vstack((adj.row, adj.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    return torch.sparse.FloatTensor(i,v, torch.Size(adj.shape))


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y, temp=None):
        if temp is None:
            temp = self.temp
        return self.cos(x, y) / temp


def forward(x):
    return x


class GraphSAINT(nn.Module):
    def __init__(self, num_classes, arch_gcn, train_params, feat_full, label_full, cpu_eval=False):
        """
        Build the multi-layer GNN architecture.

        Inputs:
            num_classes         int, number of classes a node can belong to
            arch_gcn            dict, config for each GNN layer
            train_params        dict, training hyperparameters (e.g., learning rate)
            feat_full           np array of shape N x f, where N is the total num of
                                nodes and f is the dimension for input node feature
            label_full          np array, for single-class classification, the shape
                                is N x 1 and for multi-class classification, the
                                shape is N x c (where c = num_classes)
            cpu_eval            bool, if True, will put the model on CPU.

        Outputs:
            None
        """
        super(GraphSAINT,self).__init__()
        self.vocab_size = np.max(feat_full)+1
        self.mask = np.zeros_like(feat_full)
        self.mask[feat_full==0]=1
        self.mask = torch.from_numpy(self.mask.astype(np.bool_))


        # CNN sentence embedding

        # self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # dropout_prob = train_params["dropout"]
        # self.dropout = nn.Dropout(dropout_prob)

        # self.sentence_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=128, nhead=8), num_layers=1)
        self.use_cuda = (args_global.gpu >= 0)
        if cpu_eval:
            self.use_cuda=False
        if "attention" in arch_gcn:
            if "gated_attention" in arch_gcn:
                if arch_gcn['gated_attention']:
                    self.aggregator_cls = layers.GatedAttentionAggregator
                    self.mulhead = int(arch_gcn['attention'])
            else:
                self.aggregator_cls = layers.AttentionAggregator
                self.mulhead = int(arch_gcn['attention'])
        else:
            self.aggregator_cls = layers.HighOrderAggregator
            self.mulhead = 1
        self.num_layers = len(arch_gcn['arch'].split('-'))
        self.weight_decay = train_params['weight_decay']
        self.dropout = train_params['dropout']
        self.lr = train_params['lr']
        self.arch_gcn = arch_gcn
        self.sigmoid_loss = (arch_gcn['loss'] == 'sigmoid')
        self.feat_full = torch.from_numpy(feat_full.astype(np.float32))
        self.label_full = torch.from_numpy(label_full.astype(np.float32))
        self.sentence_embed_method = train_params["sentence_embed"]
        self.sentence_embedding_dim = self.set_sentence_embedding(train_params["sentence_embed"])
        if self.use_cuda:
            self.feat_full = self.feat_full.cuda()
            self.label_full = self.label_full.cuda()
            self.mask = self.mask.cuda()
        if not self.sigmoid_loss:
            self.label_full_cat = torch.from_numpy(label_full.argmax(axis=1).astype(np.int64))
            if self.use_cuda:
                self.label_full_cat = self.label_full_cat.cuda()
        self.num_classes = num_classes
        # _dims, self.order_layer, self.act_layer, self.bias_layer, self.aggr_layer \
        #                 = parse_layer_yml(arch_gcn, self.feat_full.shape[1])
        _dims, self.order_layer, self.act_layer, self.bias_layer, self.aggr_layer \
                        = parse_layer_yml(arch_gcn, self.sentence_embedding_dim)
        # _dims, self.order_layer, self.act_layer, self.bias_layer, self.aggr_layer \
        #                 = parse_layer_yml(arch_gcn, filter_num*len(filter_size))
        # get layer index for each conv layer, useful for jk net last layer aggregation
        self.set_idx_conv()
        self.set_dims(_dims)

        self.loss = 0
        self.opt_op = None

        # build the model below
        self.num_params = 0
        self.aggregators, num_param = self.get_aggregators()
        self.aggregators1, _  = self.get_aggregators()
        self.num_params += num_param
        self.conv_layers = nn.Sequential(*self.aggregators)
        self.g2 = nn.Sequential(*self.aggregators1)
        self.hidden_dim = train_params["hidden_dim"]
        self.no_graph = train_params["no_graph"]
        self.dropout_layer = nn.Dropout(self.dropout)
        if self.hidden_dim == -1:
            if self.no_graph:
                print("NO GRAPH")
                self.classifier = layers.HighOrderAggregator(self.sentence_embedding_dim, self.num_classes, act='I', order=0, dropout=self.dropout, bias='bias')
                self.num_params += self.classifier.num_param
            else:
                # self.classifier = layers.HighOrderAggregator(self.dims_feat[-1], self.num_classes,\
                #                      act='I', order=0, dropout=self.dropout, bias='bias')
                self.classifier = layers.HighOrderAggregator(self.dims_feat[-1]+self.sentence_embedding_dim, self.num_classes,\
                                     act='I', order=0, dropout=self.dropout, bias='bias')
                # self.classifier2 = layers.HighOrderAggregator(self.sentence_embedding_dim,
                #                                              self.num_classes, act='I', order=0, dropout=self.dropout, bias='bias')
                # self.num_params += self.classifier.num_param + self.classifier2.num_param
                self.num_params += self.classifier.num_param
        else:
            self.classifier_ = layers.HighOrderAggregator(self.dims_feat[-1]+self.sentence_embedding_dim, self.hidden_dim,\
                                act='relu', order=0, dropout=self.dropout, bias='norm-nn')
            self.classifier = nn.Linear(self.hidden_dim,self.num_classes)
            self.num_params += self.classifier_.num_param + self.num_classes*self.hidden_dim
        self.sentence_embed_norm = nn.BatchNorm1d(self.sentence_embedding_dim, eps=1e-9, track_running_stats=True)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)

        # new
        # self.insert_fc = effKan(self.sentence_embedding_dim, 2)
        # self.dims_weight[self.num_layers - 1][1]
        self.encoder = layers.Encoder(self.dims_weight[self.num_layers - 1][1], self.sentence_embedding_dim)
        self.att_struct = nn.MultiheadAttention(embed_dim=self.sentence_embedding_dim, num_heads=8, dropout=self.dropout)
        self.att_text = nn.MultiheadAttention(embed_dim=self.sentence_embedding_dim, num_heads=8,dropout=self.dropout)
        self.decoder = layers.Decoder(self.sentence_embedding_dim * 4, self.dims_weight[self.num_layers - 1][1])
        self.att = nn.MultiheadAttention(embed_dim=self.dims_weight[self.num_layers - 1][1], num_heads=8, dropout=self.dropout)
        # TransGNN
        self.encoder_layer1 = nn.TransformerEncoderLayer(d_model=self.sentence_embedding_dim, nhead=1)
        self.transformer_encoder1 = nn.TransformerEncoder(self.encoder_layer1, num_layers=1)
        self.gnns = layers.KanGNN(
            self.dims_weight[self.num_layers - 1][1],
            64,
            self.dims_weight[self.num_layers - 1][1],
            self.mulhead,
            self.num_layers,
            self.dropout
        )
        self.gnns1 = layers.KanGNN(
            self.dims_weight[self.num_layers - 1][1],
            64,
            self.dims_weight[self.num_layers - 1][1],
            self.mulhead,
            self.num_layers,
            self.dropout
        )
        # self.gnns = self.conv_layers
        # self.kan = fft(self.dims_weight[self.num_layers - 1][1], self.dims_weight[self.num_layers - 1][1])
        self.encoder_layer2 = nn.TransformerEncoderLayer(d_model=self.dims_weight[self.num_layers - 1][1], nhead=2)
        # self.encoder_layer2 = nn.TransformerEncoderLayer(d_model=self.sentence_embedding_dim, nhead=1)
        self.transformer_encoder2 = nn.TransformerEncoder(self.encoder_layer2, num_layers=1)
        # self.lin = nn.Linear(self.sentence_embedding_dim, 128)
        self.kans = nn.Sequential(
            effKan(self.dims_weight[self.num_layers - 1][1], self.dims_weight[self.num_layers - 1][1]),
            # nn.Sigmoid(),
            effKan(self.dims_weight[self.num_layers - 1][1], self.dims_weight[self.num_layers - 1][1]),
            effKan(self.dims_weight[self.num_layers - 1][1], self.dims_weight[self.num_layers - 1][1]),
            # nn.Sigmoid(),
            effKan(self.dims_weight[self.num_layers - 1][1], self.dims_weight[self.num_layers - 1][1]),
            # nn.Sigmoid(),
            effKan(self.dims_weight[self.num_layers - 1][1], self.dims_weight[self.num_layers - 1][1]),
        )
        try:
            self.kanClass = train_params["kanClass"] == "True"
        except:
            self.kanClass = False
        # self.kans = KAN(
        #     (self.sentence_embedding_dim, 192, 192, self.dims_weight[self.num_layers - 1][1])
        # )

        # self.kans1 = KAN(
        #     (self.sentence_embedding_dim, 192, 192, 192, self.dims_weight[self.num_layers - 1][1])
        # )

        # self.cla = nn.Sequential(
        #     effKan(self.sentence_embedding_dim + self.dims_weight[self.num_layers - 1][1], 128),
        #     nn.ReLU(),
        #     # nn.Sigmoid(),
        #     effKan(128, 128),
        #     nn.ReLU(),
        #     # nn.Sigmoid(),
        #     effKan(128, self.sentence_embedding_dim + self.dims_weight[self.num_layers - 1][1]),
        #     nn.ReLU(),
        #     # effKan(64, 32),
        #     # effKan(32, 2),
        #     # nn.Softmax()
        # )

    def set_dims(self, dims):
        """
        Set the feature dimension / weight dimension for each GNN or MLP layer.
        We will use the dimensions set here to initialize PyTorch layers.

        Inputs:
            dims        list, length of node feature for each hidden layer

        Outputs:
            None
        """
        self.dims_feat = [dims[0]] + [
            ((self.aggr_layer[l]=='concat') * self.order_layer[l] + 1) * dims[l+1]
            for l in range(len(dims) - 1)
        ]
        self.dims_weight = [(self.dims_feat[l],dims[l+1]) for l in range(len(dims)-1)]

    def set_idx_conv(self):
        """
        Set the index of GNN layers for the full neural net. For example, if
        the full NN is having 1-0-1-0 arch (1-hop graph conv, followed by 0-hop
        MLP, ...). Then the layer indices will be 0, 2.
        """
        idx_conv = np.where(np.array(self.order_layer) >= 1)[0]
        idx_conv = list(idx_conv[1:] - 1)
        idx_conv.append(len(self.order_layer) - 1)
        _o_arr = np.array(self.order_layer)[idx_conv]
        if np.prod(np.ediff1d(_o_arr)) == 0:
            self.idx_conv = idx_conv
        else:
            self.idx_conv = list(np.where(np.array(self.order_layer) == 1)[0])


    def cos_sim(self, x, is_training):
        if is_training:
            return nn.functional.cosine_similarity(x.unsqueeze(1), x.unsqueeze(0), dim=-1)
        B, L = x.shape
        sims = torch.zeros(size=(B,B))
        for i in range(B):
            sims[i,:] = nn.functional.cosine_similarity(x, x[i].unsqueeze(0))
        return sims.to(x.device)

    def top_sim(self, sims, topk=3):
        sims_, indices_ = sims.sort(descending=True)
        B,_ = sims.shape
        indices = torch.zeros(size=(2,B*topk))
        values = torch.zeros(size=(1,B*topk))
        for i,inds in enumerate(indices_):
            indices[0, i*topk:(i+1)*topk] = i*torch.ones(size=(1,topk))
            indices[1, i * topk:(i + 1) * topk] = inds[:topk]
            values[0, i * topk:(i + 1) * topk] = sims[i][:topk]
        return torch.sparse_coo_tensor(indices,values.squeeze(0),size=(B,B))
        # return indices, values

    def set_sentence_embedding(self, method="cnn"):
        if method=="cnn":
            embed_size = 128
            filter_size = [3, 4, 5]
            filter_num = 128
            self.embedding = nn.Embedding(self.vocab_size, embed_size)
            self.cnn_list = nn.ModuleList()
            for size in filter_size:
                self.cnn_list.append(nn.Conv1d(embed_size, filter_num, size))
            self.relu = nn.ReLU()
            self.max_pool = nn.AdaptiveMaxPool1d(1)
            self.sentence_embed=self.cnn_embed
            return len(filter_size) * filter_num
        '''
        one layer FCN equals only pool
        '''
        if method == "maxpool":
            self.embed_size = 128
            self.embed = nn.Embedding(self.vocab_size, self.embed_size)
            self.relu = nn.ReLU()
            self.embed_pool = nn.AdaptiveMaxPool1d(1)
            self.sentence_embed=self.pool_embed
            return self.embed_size
        if method == "avgpool":
            self.embed_size = 128
            self.embed = nn.Embedding(self.vocab_size, self.embed_size)
            self.relu = nn.ReLU()
            self.embed_pool = nn.AdaptiveAvgPool1d(1)
            self.sentence_embed = self.pool_embed
            return self.embed_size
        if method == "rnn":
            self.embed_size = 128
            hidden_dim = 64
            self.embed = nn.Embedding(self.vocab_size, self.embed_size)
            self.rnn = nn.LSTM(self.embed_size, hidden_dim, 1, dropout=self.dropout, bidirectional=True)
            self.sentence_embed=self.RNN_embed
            return 2 * hidden_dim
        if method == "lstm":
            self.embed_size = 128
            hidden_dim = 64
            self.embed = nn.Embedding(self.vocab_size, self.embed_size)
            self.rnn = nn.LSTM(self.embed_size, hidden_dim, 1, dropout=self.dropout, bidirectional=True)
            self.sentence_embed=self.LSTM_embed
            return 2 * hidden_dim * 2
        if method == "lstmatt":
            self.embed_size = 128
            hidden_dim = 64
            self.LSTMATT_DP=nn.Dropout(self.dropout)
            self.embed = nn.Embedding(self.vocab_size, self.embed_size)
            self.lstm = nn.LSTM(self.embed_size, hidden_dim, 1, dropout=self.dropout, bidirectional=True)
            self.attn = nn.MultiheadAttention(embed_dim=hidden_dim*2, num_heads=1, dropout=self.dropout)
            self.sentence_embed = self.LSTMATT_embed
            self.relu = nn.ReLU()
            self.embed_pool = nn.AdaptiveAvgPool1d(1)
            return hidden_dim*2
        if method == "Transformer":
            self.embed_size = 128
            self.embed = nn.Embedding(self.vocab_size, self.embed_size)
            self.Trans_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=128, nhead=8), num_layers=1)
            self.sentence_embed =self.Transformer_embed
            return self.embed_size

        if method == "LLM":
            self.embed_size = 128
            self.llm_block = 2
            self.embed = nn.Embedding(self.vocab_size, self.embed_size)
            self.llm_list = nn.ModuleList()
            for i in range(self.llm_block):
                self.llm_list.append(nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=128, nhead=8), num_layers=1))
            self.sentence_embed =self.LLM_embed
            return self.embed_size

        if method == "gnn":
            embed_size = 128
            self.edge_weight = nn.Embedding((self.vocab_size) * (self.vocab_size)+1, 1, padding_idx=0)
            self.node_embedding = nn.Embedding(self.vocab_size, embed_size, padding_idx=0)
            self.node_weight = nn.Embedding(self.vocab_size, 1, padding_idx=0)
            # nn.init.xavier_uniform_(self.edge_weight.weight)
            # nn.init.xavier_uniform_(self.node_weight.weight)
            self.sentence_embed = self.gnn_embed
            return embed_size
            # self.fc = nn.Sequential(
            #     nn.Linear(embed_size, 2),
            #     nn.ReLU(),
            #     nn.Dropout(self.dropout),
            #     nn.LogSoftmax(dim=1)
            # )


    def forward(self, node_subgraph, adj_subgraph, current_epoch=10, is_training=True):
        feat_subg = self.feat_full[node_subgraph]
        label_subg = self.label_full[node_subgraph]
        mask_subg = self.mask[node_subgraph]
        feat_subg = self.sentence_embed(tokens=feat_subg, padding_mask=mask_subg, is_training=is_training)
        try:
            label_subg_converted = label_subg if self.sigmoid_loss else self.label_full_cat[node_subgraph]
        except:
            print()
        if self.no_graph:
            pred_subg = self.classifier((None, feat_subg))[1]
        else:
            feat_subg_ = self.sentence_embed_norm(feat_subg)
            # sims = self.cos_sim(feat_subg_, is_training)
            # sims = self.top_sim(sims,topk=3).to(feat_subg_.device)
            # adj_subgraph += sims
            # adj_subgraph = adj_subgraph.to_dense()
            # # adj_subgraph = torch.ones_like(adj_subgraph).to(adj_subgraph.device) + adj_subgraph
            # adj_subgraph = adj_subgraph+sims*(sims>=0.85)
            # indices = adj_subgraph.to_sparse().indices()
            # values =  adj_subgraph.to_sparse().values()
            # adj_subgraph = torch.sparse_coo_tensor(indices,values,size=adj_subgraph.size())

            feat_subg_ = self.dropout_layer(feat_subg_)
            # sims = self.sim(feat_subg.unsqueeze(1), feat_subg.unsqueeze(0))

            # new idea
            # adj_subgraph, feat_subg_, label_subg_converted, label_subg, feat_subg \
            #     = self.upSample(adj_subgraph, feat_subg_, label_subg_converted, label_subg, feat_subg)
            # self.transGNN(adj_subgraph._indices(), feat_subg_)


            # feat_subg_sample_ = self.sentence_embed_norm(feat_subg_sample)
            # mask1 = torch.randint(0, feat_subg.shape[0], (50, )).cuda()
            # mask2 = torch.randint(0, feat_subg.shape[0], (50, )).cuda()
            # adj_sample1 = torch.mul(adj_subgraph.to_dense(), mask1).to(torch.int64)
            # adj_sample2 = torch.mul(adj_subgraph.to_dense(), mask2).to(torch.int64)
            # _, feat1 = self.conv_layers((adj_subgraph.to_dense()[adj_sample1].to_sparse(), feat_subg_[adj_sample1]))
            # _, feat2 = self.conv_layers((adj_subgraph.to_dense()[adj_sample2].to_sparse(), feat_subg_[adj_sample2]))
            if current_epoch >= 0:
                # TransGNN
                trans_embed = self.transformer_encoder1(feat_subg_.unsqueeze(0)).squeeze(0)
                _, emb_subg = self.conv_layers((adj_subgraph, trans_embed))
                # _, emb_subg = self.gnns(emb_subg, adj_subgraph)
                # lin = self.lin(feat_subg_)
                # emb_subg = self.kans(trans_embed)
                # emb_subg = self.kans1(emb_subg + feat_subg_)
                transGNN_embed = self.transformer_encoder2(emb_subg.unsqueeze(0)).squeeze(0)
                # _, emb_subg = self.gnns1(transGNN_embed, adj_subgraph)

                # transGNN信息维度变换
                emb_subg = self.encoder(transGNN_embed)
                # emb_subg = self.g2((adj_subgraph, emb_subg))
                # _, emb_subg = self.gnns(emb_subg, adj_subgraph)

                # emb_subg = transGNN_embed
                # attention再次提取结构化信息
                att_struct, _ = self.att_struct(emb_subg.unsqueeze(0), feat_subg_.unsqueeze(0), emb_subg.unsqueeze(0))
                # attention再次提取文本信息
                att_text, _ = self.att_text(feat_subg_.unsqueeze(0), emb_subg.unsqueeze(0), feat_subg_.unsqueeze(0))
                # 信息合并解码
                combine = torch.cat([emb_subg, feat_subg_, att_struct.squeeze(0), att_text.squeeze(0)], dim=1)
                emb_subg = self.decoder(combine)
                emb_subg, _ = self.att(emb_subg.unsqueeze(0), emb_subg.unsqueeze(0), emb_subg.unsqueeze(0))
                emb_subg = emb_subg.squeeze(0)
                # emb_subg = emb_subg + feat_subg_ + att_struct.squeeze(0) + att_text.squeeze(0)
                # emb_subg = self.decoder(emb_subg + feat_subg_)
                # emb_subg_norm = F.normalize(emb_subg, p=2, dim=1)

                if self.hidden_dim == -1:
                    # pred_subg = self.classifier((None, emb_subg_norm))[1]
                    # _, emb_subg_norm = self.gnns(emb_subg, adj_subgraph)
                    # emb_subg_norm = F.normalize(emb_subg_norm, p=2, dim=1)
                    # lin = self.lin(feat_subg_)
                    # if self.kanClass:
                    #     emb_subg_norm = self.kans(emb_subg)
                    # else:
                    #     _, emb_subg_norm = self.gnns(emb_subg, adj_subgraph)
                    # _, emb_subg_norm = self.g2((adj_subgraph, emb_subg))
                    pred_subg = self.classifier((None, torch.cat([emb_subg, feat_subg],dim=1)))[1]
                    # pred_subg = self.cla(torch.cat([emb_subg_norm, feat_subg], dim=1))
                    # pred_subg = self.classifier((None, pred_subg))[1]
                    # _, pred_subg = self.gnns(torch.cat([emb_subg_norm, feat_subg],dim=1), adj_subgraph)
                    # pred_subg = self.cla(torch.cat([emb_subg_norm, feat_subg], dim=1))
                else:
                    pred_subg = self.classifier_((None, torch.cat([emb_subg_norm, feat_subg], dim=1)))[1]
                    pred_subg = self.classifier(pred_subg)
            else:
                pred_subg = self.classifier2((None, feat_subg))[1]
        return pred_subg, label_subg, label_subg_converted, emb_subg, None


    def upSample(self, adj_subgraph, feat_subg_, label_subg_converted, label_subg, feat_subg):
        new_dices = None
        new_values = None
        old_dices = adj_subgraph._indices()
        for k in range(len(old_dices[0])):
            i = old_dices[0][k]
            j = old_dices[1][k]

            if i == j:
                continue
            prob = nn.Softmax(dim=0)(
                self.insert_fc(torch.cat([feat_subg_[i].unsqueeze(0), feat_subg_[j].unsqueeze(0)], dim=0)).sum(1))
            prob = prob.argmax()
            if prob == 1:
                new_node_feat = (feat_subg_[i] + feat_subg_[j])
                feat_subg = torch.cat((feat_subg, new_node_feat.unsqueeze(0)), dim=0)

                new_label = (label_subg_converted[i] + label_subg_converted[j]) % 2
                label_subg_converted = torch.cat((label_subg_converted, new_label.unsqueeze(0)), dim=0)
                if label_subg_converted[i] + label_subg_converted[j] == 1:
                    label_subg = torch.cat((label_subg, torch.tensor([0, 1]).unsqueeze(0).cuda()), dim=0)
                else:
                    label_subg = torch.cat((label_subg, torch.tensor([1, 0]).unsqueeze(0).cuda()), dim=0)

                if new_dices == None:
                    new_dices = torch.tensor([i, feat_subg_.shape[0] - 1]).unsqueeze(0).cuda()
                    new_values = torch.tensor(1).unsqueeze(0).cuda()
                    new_label_subg = torch.tensor([0, 1]).unsqueeze(0).cuda()

                    new_dices = torch.cat(
                        (new_dices, torch.tensor([feat_subg_.shape[0] - 1, j]).unsqueeze(0).cuda()), dim=0)
                    new_values = torch.cat((new_values, torch.tensor(1).unsqueeze(0).cuda()), dim=0)
                    new_label_subg = torch.cat((new_label_subg, torch.tensor([0, 1]).unsqueeze(0).cuda()), dim=0)
                else:
                    new_dices = torch.cat(
                        (new_dices, torch.tensor([i, feat_subg_.shape[0] - 1]).unsqueeze(0).cuda()), dim=0)
                    new_values = torch.cat((new_values, torch.tensor(1).unsqueeze(0).cuda()), dim=0)
                    new_label_subg = torch.cat((new_label_subg, torch.tensor([0, 1]).unsqueeze(0).cuda()), dim=0)

                    new_dices = torch.cat(
                        (new_dices, torch.tensor([feat_subg_.shape[0] - 1, j]).unsqueeze(0).cuda()), dim=0)
                    new_values = torch.cat((new_values, torch.tensor(1).unsqueeze(0).cuda()), dim=0)
                    new_label_subg = torch.cat((new_label_subg, torch.tensor([0, 1]).unsqueeze(0).cuda()),
                                               dim=0)
            else:
                if new_dices == None:
                    new_dices = torch.tensor([i, j]).unsqueeze(0).cuda()
                    new_values = torch.tensor(1).unsqueeze(0).cuda()
                    new_label_subg = torch.tensor([1, 0]).unsqueeze(0).cuda()

                else:
                    new_dices = torch.cat((new_dices, torch.tensor([i, j]).unsqueeze(0).cuda()), dim=0)
                    new_values = torch.cat((new_values, torch.tensor(1).unsqueeze(0).cuda()), dim=0)
                    new_label_subg = torch.cat((new_label_subg, torch.tensor([1, 0]).unsqueeze(0).cuda()),
                                               dim=0)

        # for i in adj_subgraph._indices()[0, :]:
        #     for j in adj_subgraph._indices()[1, :]:
        #         if i == j:
        #             continue
        #         prob = nn.Softmax(dim=0)(
        #             self.insert_fc(torch.cat([feat_subg_[i].unsqueeze(0), feat_subg_[j].unsqueeze(0)], dim=0)).sum(1))
        #         prob = prob.argmax()
        #         if prob == 1:
        #             new_node_feat = (feat_subg_[i] - feat_subg_[j])
        #             feat_subg = torch.cat((feat_subg, new_node_feat.unsqueeze(0)), dim=0)
        #
        #             new_label = (label_subg_converted[i] + label_subg_converted[j]) % 2
        #             label_subg_converted = torch.cat((label_subg_converted, new_label.unsqueeze(0)), dim=0)
        #             if label_subg_converted[i] + label_subg_converted[j] == 1:
        #                 label_subg = torch.cat((label_subg, torch.tensor([0, 1]).unsqueeze(0).cuda()), dim=0)
        #             else:
        #                 label_subg = torch.cat((label_subg, torch.tensor([1, 0]).unsqueeze(0).cuda()), dim=0)
        #
        #             if new_dices == None:
        #                 new_dices = torch.tensor([i, feat_subg_.shape[0] - 1]).unsqueeze(0).cuda()
        #                 new_values = torch.tensor(1).unsqueeze(0).cuda()
        #                 new_label_subg = torch.tensor([0, 1]).unsqueeze(0).cuda()
        #
        #                 new_dices = torch.cat(
        #                     (new_dices, torch.tensor([feat_subg_.shape[0] - 1, j]).unsqueeze(0).cuda()), dim=0)
        #                 new_values = torch.cat((new_values, torch.tensor(1).unsqueeze(0).cuda()), dim=0)
        #                 new_label_subg = torch.cat((new_label_subg, torch.tensor([0, 1]).unsqueeze(0).cuda()), dim=0)
        #             else:
        #                 new_dices = torch.cat(
        #                     (new_dices, torch.tensor([i, feat_subg_.shape[0] - 1]).unsqueeze(0).cuda()), dim=0)
        #                 new_values = torch.cat((new_values, torch.tensor(1).unsqueeze(0).cuda()), dim=0)
        #                 new_label_subg = torch.cat((new_label_subg, torch.tensor([0, 1]).unsqueeze(0).cuda()), dim=0)
        #
        #                 new_dices = torch.cat(
        #                     (new_dices, torch.tensor([feat_subg_.shape[0] - 1, j]).unsqueeze(0).cuda()), dim=0)
        #                 new_values = torch.cat((new_values, torch.tensor(1).unsqueeze(0).cuda()), dim=0)
        #                 new_label_subg = torch.cat((new_label_subg, torch.tensor([0, 1]).unsqueeze(0).cuda()),
        #                                            dim=0)
        #         else:
        #             if new_dices == None:
        #                 new_dices = torch.tensor([i, j]).unsqueeze(0).cuda()
        #                 new_values = torch.tensor(1).unsqueeze(0).cuda()
        #                 new_label_subg = torch.tensor([1, 0]).unsqueeze(0).cuda()
        #
        #             else:
        #                 new_dices = torch.cat((new_dices, torch.tensor([i, j]).unsqueeze(0).cuda()), dim=0)
        #                 new_values = torch.cat((new_values, torch.tensor(1).unsqueeze(0).cuda()), dim=0)
        #                 new_label_subg = torch.cat((new_label_subg, torch.tensor([1, 0]).unsqueeze(0).cuda()),
        #                                            dim=0)

        adj_subgraph = torch.sparse_coo_tensor(new_dices.T, new_values, size=(feat_subg.shape[0], feat_subg.shape[0]))
        feat_subg_ = self.sentence_embed_norm(feat_subg)
        feat_subg_ = self.dropout_layer(feat_subg_)

        return adj_subgraph, feat_subg_, label_subg_converted, label_subg, feat_subg

    def cnn_embed(self, tokens, padding_mask=None, is_training = None):
        '''
        CNN sentence embedding
        '''
        x = tokens.long()
        _ = self.embedding(x)
        _ = _.permute(0, 2, 1)
        result = []
        for self.cnn in self.cnn_list:
            __ = self.cnn(_)
            __ = self.max_pool(__)
            __ = self.relu(__)
            result.append(__.squeeze(dim=2))
        _ = torch.cat(result, dim=1)
        return _
        # return self.relu(self.avg_pool(self.embed(tokens.long()).permute(0,2,1)).squeeze())
        # return self.max_pool(self.sentence_encoder(self.embed(tokens.long()).permute(1,0,2), src_key_padding_mask=padding_mask).permute(1,0,2).permute(0,2,1)).squeeze()

    def pool_embed(self, tokens,padding_mask=None,is_training = None):
        return self.relu(self.embed_pool(self.embed(tokens.long()).permute(0, 2, 1)).squeeze())

    def RNN_embed(self, tokens, padding_mask=None,is_training = None):
        x = tokens.long()
        _ = self.embed(x)
        _ = _.permute(1, 0, 2)
        # h_out = self.rnn(_)
        hidden = self.rnn(_)
        hidden = hidden[1][-1].permute(1, 0, 2).reshape((-1, self.sentence_embedding_dim))
        return hidden

    def LSTM_embed(self, tokens, padding_mask=None,is_training = None):
        x = tokens.long()
        _ = self.embed(x)
        _ = _.permute(1, 0, 2)
        __, h_out = self.rnn(_)
        # if self._cell in ["lstm", "bi-lstm"]:
        #     h_out = torch.cat([h_out[0], h_out[1]], dim=2)
        h_out = torch.cat([h_out[0], h_out[1]], dim=2)
        h_out = h_out.permute(1, 0, 2)
        h_out = h_out.reshape(-1, h_out.shape[1] * h_out.shape[2])
        return h_out

    def LSTMATT_embed(self, tokens, padding_mask=None,is_training = None):
        input_lstm = self.embed(tokens.long())
        input_lstm = input_lstm.permute(1,0,2)
        output, _ = self.lstm(input_lstm)
        output = self.LSTMATT_DP(output)
        # sentence_output = torch.cat([_[0], _[1]], dim=2)
        # scc, _ = self.attn(output, output, output, key_padding_mask=padding_mask)
        # scc, _ = self.attn(sentence_output.mean(dim=0).unsqueeze(0), output, output, key_padding_mask=padding_mask)
        # return scc.squeeze()
        return self.embed_pool(self.attn(output, output, output, key_padding_mask=padding_mask)[0].permute(1,0,2).permute(0,2,1)).squeeze()
        # return self.attn(output, output, output, key_padding_mask=padding_mask)[0].mean(dim=0)

    def Transformer_embed(self, tokens, padding_mask=None,is_training = None):
        return self.Trans_encoder(self.embed(tokens.long()).permute(1, 0, 2),src_key_padding_mask=padding_mask).mean(dim=0)

    def LLM_embed(self, tokens, padding_mask=None,is_training = None):
        tokens = self.embed(tokens.long()).permute(1, 0, 2)
        for trans in self.llm_list:
            tokens = trans(tokens, src_key_padding_mask=padding_mask)
        return tokens.mean(dim=0)

    def gnn_embed(self, tokens, padding_mask=None,is_training = None):
        X = tokens.long()
        NX, EW = self.get_neighbors(X, nb_neighbor=2)
        NX = NX.long()
        EW = EW.long()
        # NX = input_ids
        # EW = input_ids
        Ra = self.node_embedding(NX)
        # edge weight  (bz, seq_len, neighbor_num, 1)
        Ean = self.edge_weight(EW)
        # neighbor representation  (bz, seq_len, embed_dim)
        if not is_training:
            B, L, N, E = Ra.shape
            # Mn = torch.zeros(size=(B,L,E)).to(Ra.device)
            y = torch.zeros(size=(B,E)).to(Ra.device)
            Rn = self.node_embedding(X)
            # self node weight  (bz, seq_len, 1)
            Nn = self.node_weight(X)
            # aggregate node features
            for i in range(B):
                tmp = (Ra[i]*Ean[i]).max(dim=1)[0]
                # Mn[i,:,:] = tmp
                y[i] = ((1-Nn[i])*tmp + Nn[i] * Rn[i]).sum(dim=0)
            return y
        Mn = (Ra * Ean).max(dim=2)[0]  # max pooling
        # self representation (bz, seq_len, embed_dim)
        Rn = self.node_embedding(X)
        # self node weight  (bz, seq_len, 1)
        Nn = self.node_weight(X)
        # aggregate node features
        y = (1 - Nn) * Mn + Nn * Rn
        return y.sum(dim=1)


    def get_neighbors(self, x_ids, nb_neighbor=2):
        B, L = x_ids.size()
        neighbours = torch.zeros(size=(L, B, 2 * nb_neighbor))
        ew_ids = torch.zeros(size=(L, B, 2 * nb_neighbor))
        # pad = [0] * nb_neighbor
        pad = torch.zeros(size=(B, nb_neighbor)).to(x_ids.device)
        # x_ids_ = pad + list(x_ids) + pad
        x_ids_ = torch.cat([pad, x_ids, pad], dim=-1)
        for i in range(nb_neighbor, L + nb_neighbor):
            # x = x_ids_[i - nb_neighbor: i] + x_ids_[i + 1: i + nb_neighbor + 1]
            neighbours[i - nb_neighbor, :, :] = torch.cat(
                [x_ids_[:, i - nb_neighbor: i], x_ids_[:, i + 1: i + nb_neighbor + 1]], dim=-1)
        # ew_ids[i-nb_neighbor,:,:] = (x_ids[i-nb_neighbor,:] -1) * self.vocab_size + nb_neighbor[i-nb_neighbor,:,:]
        neighbours = neighbours.permute(1, 0, 2).to(x_ids.device)
        ew_ids = ((x_ids) * (self.vocab_size)).reshape(B, L, 1) + neighbours
        ew_ids[neighbours == 0] = 0
        return neighbours, ew_ids


    def _loss(self, preds, labels, norm_loss, feat1, feat2):
        """
        The predictor performs sigmoid (for multi-class) or softmax (for single-class)
        """
        if self.sigmoid_loss:
            norm_loss = norm_loss.unsqueeze(1)
            return torch.nn.BCEWithLogitsLoss(weight=norm_loss,reduction='sum')(preds, labels)
        else:
            _ls = torch.nn.CrossEntropyLoss(reduction='none')(preds, labels)
            # sim = torch.zeros(size=(feat1.shape[0] // 100, feat1.shape[0] // 100)).to(feat1.device)
            # for i in range(feat1.shape[0] // 100):
            #     for j in range(feat1.shape[0] // 100):
            #         sim[i][j] = nn.functional.cosine_similarity(feat1[i], feat1[j], dim=0).abs()
            # pair = torch.zeros(size=(feat1.shape[0] // 100, feat1.shape[0] // 100)).to(feat1.device)
            # for i in range(feat1.shape[0] // 100):
            #     for j in range(feat1.shape[0] // 100):
            #         pair[i][j] = 1 if labels[i] == labels[j] else -1
            # sim = (sim - pair).abs()
            return (norm_loss*_ls).sum()
            # return (_ls).sum()




    def get_aggregator(self, in_size, out_size, l):
        num_param = 0
        aggr = self.aggregator_cls(
            dim_in=in_size,
            dim_out=out_size,
            dropout=self.dropout,
            act='relu',
            order=self.order_layer[l],
            aggr='concat',
            bias='norm',
            mulhead=self.mulhead,
        )
        num_param += aggr.num_param
        return aggr, num_param


    def get_aggregators(self):
        """
        Return a list of aggregator instances. to be used in self.build()
        """
        num_param = 0
        aggregators = []
        for l in range(self.num_layers):
            aggr = self.aggregator_cls(
                    *self.dims_weight[l],
                    dropout=self.dropout,
                    act=self.act_layer[l],
                    order=self.order_layer[l],
                    aggr=self.aggr_layer[l],
                    bias=self.bias_layer[l],
                    mulhead=self.mulhead,
            )
            num_param += aggr.num_param
            aggregators.append(aggr)
        return aggregators, num_param


    def predict(self, preds):
        return nn.Sigmoid()(preds) if self.sigmoid_loss else F.softmax(preds, dim=1)


    def train_step(self, node_subgraph, adj_subgraph, norm_loss_subgraph,current_epoch=0):
        """
        Forward and backward propagation
        """
        self.train()
        self.optimizer.zero_grad()
        preds, labels, labels_converted, feat1, feat2 = self(node_subgraph, adj_subgraph,current_epoch=current_epoch, is_training=True)
        loss = self._loss(preds, labels_converted, norm_loss_subgraph, feat1, feat2) # labels.squeeze()?
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.parameters(), 5)
        self.optimizer.step()
        return loss, self.predict(preds), labels


    def eval_step(self, node_subgraph, adj_subgraph, norm_loss_subgraph):
        """
        Forward propagation only
        """
        self.eval()
        with torch.no_grad():
            preds,labels,labels_converted, feat1, feat2 = self(node_subgraph, adj_subgraph,is_training=False)
            loss = self._loss(preds,labels_converted,norm_loss_subgraph, feat1, feat2)
        return loss, self.predict(preds), labels
