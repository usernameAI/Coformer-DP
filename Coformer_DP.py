import math
import torch
from torch import nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.utils import InputType


class MLP_ln_trans(nn.Module):
    def __init__(self, input_dim, embed_dim, output_dim, dropout):
        super(MLP_ln_trans, self).__init__()
        self.dropout = dropout
        self.ln = nn.LayerNorm(normalized_shape=input_dim, eps=1e-12)
        self.input_projection = nn.Linear(input_dim, embed_dim)
        self.output_projection = nn.Linear(embed_dim, output_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)
        self.ln1 = nn.LayerNorm(normalized_shape=embed_dim, eps=1e-12)

    def forward(self, x):
        origin = x
        x = self.ln(x)
        x = self.input_projection(x)
        x = self.activation(x)
        x = self.ln1(x)
        x = self.dropout(x)
        x = self.output_projection(x) + origin
        return x


class fusion_triple_feature(nn.Module):
    def __init__(self, emb_dim):
        super(fusion_triple_feature, self).__init__()
        self.emb_dim = emb_dim
        self.linear1 = nn.Linear(self.emb_dim, self.emb_dim)
        self.linear2 = nn.Linear(self.emb_dim, self.emb_dim)
        self.linear3 = nn.Linear(self.emb_dim, self.emb_dim)
        self.softmax = nn.Softmax(dim=1)
        self.linear_final = nn.Linear(self.emb_dim, self.emb_dim)

    def forward(self, seq_hidden, pos_emb, category_seq_hidden):
        seq_hidden = seq_hidden.unsqueeze(dim=1)
        pos_emb = pos_emb.unsqueeze(dim=1)
        category_seq_hidden = category_seq_hidden.unsqueeze(dim=1)
        seq_hidden = self.linear1(seq_hidden)
        pos_emb = self.linear2(pos_emb)
        category_seq_hidden = self.linear3(category_seq_hidden)
        fusion_feature = torch.cat((seq_hidden, pos_emb, category_seq_hidden), dim=1)
        attn_weight = self.softmax(fusion_feature)
        fusion_feature = torch.sum(attn_weight * fusion_feature, dim=1)
        fusion_feature = self.linear_final(fusion_feature)
        return fusion_feature


class seq_affinity_soft_attention(nn.Module):
    def __init__(self, emb_dim):
        super(seq_affinity_soft_attention, self).__init__()
        self.emb_dim = emb_dim
        self.linear_1 = nn.Linear(self.emb_dim, self.emb_dim)
        self.linear_2 = nn.Linear(self.emb_dim, self.emb_dim)
        self.linear_3 = nn.Linear(self.emb_dim, self.emb_dim)
        self.linear_alpha = nn.Linear(self.emb_dim, 1, bias=False)
        self.long_middle_short = fusion_triple_feature(self.emb_dim)

    def forward(self, mask, short_feature, seq_feature, affinity_feature):
        q1 = self.linear_1(seq_feature)
        q2 = self.linear_2(short_feature)
        q2_expand = q2.unsqueeze(1).expand_as(q1)
        q3 = self.linear_3(affinity_feature)
        q3_expand = q3.unsqueeze(1).expand_as(q1)
        alpha = self.linear_alpha(mask * torch.sigmoid(q1 + q2_expand + q3_expand))
        long_feature = torch.sum(alpha.expand_as(seq_feature) * seq_feature, 1)
        seq_output = self.long_middle_short(long_feature, short_feature, affinity_feature)
        return seq_output


class gate_mechanism(nn.Module):
    def __init__(self, emb_dim):
        super(gate_mechanism, self).__init__()
        self.emb_dim = emb_dim
        self.linear = nn.Linear(3 * self.emb_dim, self.emb_dim)
        self.act = nn.Sigmoid()

    def forward(self, tensor_a, tensor_b):
        alpha = torch.cat((tensor_a, tensor_b, tensor_a * tensor_b), dim=-1)
        alpha = self.linear(alpha)
        alpha = self.act(alpha)

        output = alpha * tensor_a + (1 - alpha) * tensor_b
        return output


class Dual_QK_mutlihead_attn(nn.Module):
    def __init__(
            self,
            n_heads,
            hidden_size,
            hidden_dropout_prob,
            attn_dropout_prob,
            layer_norm_eps,
    ):
        super(Dual_QK_mutlihead_attn, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)
        self.pre_ln_q = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.pre_ln_k = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.pre_ln_v = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.post_ln = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)
        self.query1 = nn.Linear(hidden_size, self.all_head_size)
        self.key1 = nn.Linear(hidden_size, self.all_head_size)
        self.pre_ln_q1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.pre_ln_k1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.attn_dropout1 = nn.Dropout(attn_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x

    def forward(self, query_tensor, key_tensor, query_tensor1, key_tensor1, value_tensor, attention_mask):
        origin_value = value_tensor
        # pre_ln
        query_tensor = self.pre_ln_q(query_tensor)
        key_tensor = self.pre_ln_k(key_tensor)
        query_tensor1 = self.pre_ln_q1(query_tensor1)
        key_tensor1 = self.pre_ln_k1(key_tensor1)
        value_tensor = self.pre_ln_v(value_tensor)
        mixed_query_layer = self.query(query_tensor)
        mixed_key_layer = self.key(key_tensor)
        mixed_query_layer1 = self.query1(query_tensor1)
        mixed_key_layer1 = self.key1(key_tensor1)
        mixed_value_layer = self.value(value_tensor)
        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)
        query_layer1 = self.transpose_for_scores(mixed_query_layer1).permute(0, 2, 1, 3)
        key_layer1 = self.transpose_for_scores(mixed_key_layer1).permute(0, 2, 3, 1)
        value_layer = self.transpose_for_scores(mixed_value_layer).permute(0, 2, 1, 3)
        attention_scores = torch.matmul(query_layer, key_layer)
        attention_scores1 = torch.matmul(query_layer1, key_layer1)
        attention_scores = attention_scores / self.sqrt_attention_head_size
        attention_scores1 = attention_scores1 / self.sqrt_attention_head_size
        attention_scores = attention_scores + attention_mask
        attention_scores1 = attention_scores1 + attention_mask
        attention_probs = self.softmax(attention_scores)
        attention_probs1 = self.softmax(attention_scores1)
        attention_probs = self.attn_dropout(attention_probs)
        attention_probs1 = self.attn_dropout(attention_probs1)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer1 = torch.matmul(attention_probs1, value_layer)
        context_layer = context_layer + context_layer1
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        context_layer = self.post_ln(context_layer)
        context_layer = self.out_dropout(context_layer)
        hidden_states = self.dense(context_layer) + origin_value
        return hidden_states


class Dual_QK_former(nn.Module):
    def __init__(self, n_heads, hidden_size, intermediate_size, hidden_dropout_prob, attn_dropout_prob,
                 n_layers, layer_norm_eps=1e-12):
        super(Dual_QK_former, self).__init__()
        self.n_layers = n_layers
        self.multi_head_attention_list = nn.ModuleList(Dual_QK_mutlihead_attn(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps
        )
                                                       for _ in range(self.n_layers))
        self.mlp_list = nn.ModuleList(MLP_ln_trans(input_dim=hidden_size, embed_dim=intermediate_size,
                                                   output_dim=hidden_size, dropout=hidden_dropout_prob)
                                      for _ in range(self.n_layers))

    def forward(self, query_tensor, key_tensor, query_tensor1, key_tensor1, value_tensor, attention_mask):
        feature_list = []
        for layer in range(self.n_layers):
            value_tensor = self.multi_head_attention_list[layer](query_tensor, key_tensor, query_tensor1,
                                                                 key_tensor1, value_tensor, attention_mask)
            value_tensor = self.mlp_list[layer](value_tensor)
            feature_list.append(value_tensor)
        value_tensor = sum(feature_list) / len(feature_list)
        return value_tensor


class Coformer_DP(SequentialRecommender):
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(Coformer_DP, self).__init__(config, dataset)
        self.dropout = config['dropout']
        self.embedding_size = config['embedding_size']
        self.device = config['device']
        self.max_seq_length = dataset.field2seqlen[self.ITEM_SEQ]
        self.mask_token = self.n_items
        self.time_layer_num = config['time_layer_num']
        self.time_head_num = config['time_head_num']
        self.position_embedding = nn.Embedding(self.max_seq_length, self.embedding_size)
        self.temperature_parameter = config['temperature_parameter']
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        self.time_trans = Dual_QK_former(
            n_layers=self.time_layer_num,
            n_heads=self.time_head_num,
            hidden_size=self.embedding_size,
            intermediate_size=4 * self.embedding_size,
            hidden_dropout_prob=self.dropout,
            attn_dropout_prob=self.dropout,
        )
        self.rnn_pos = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.embedding_size,
            num_layers=1,
            bias=False,
            batch_first=True,
        )
        self.rnn_pos_linear = nn.Linear(self.embedding_size, self.embedding_size)
        self.time_soft_attention = seq_affinity_soft_attention(emb_dim=self.embedding_size)
        self.dropout_layer = nn.ModuleList(nn.Dropout(p=self.dropout) for _ in range(4))
        self.gate = nn.ModuleList(gate_mechanism(self.embedding_size) for _ in range(2))
        self.ce_loss = nn.CrossEntropyLoss()

    def get_attention_mask(self, item_seq):
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)
        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, item_seq_len, item_seq):
        time_input = self.item_embedding(item_seq)
        time_input = self.dropout_layer[0](time_input)
        rnn_pos, _ = self.rnn_pos(time_input)
        rnn_pos = self.rnn_pos_linear(rnn_pos)
        rnn_pos = torch.tanh(rnn_pos)
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        init_pos = self.position_embedding(position_ids)
        time_init_pos = self.gate[0](time_input, init_pos)
        time_rnn_pos = self.gate[1](time_input, rnn_pos)
        time_attention_mask = self.get_attention_mask(item_seq)
        time_seq = self.time_trans(
            query_tensor=time_init_pos,
            key_tensor=time_init_pos,
            query_tensor1=time_rnn_pos,
            key_tensor1=time_rnn_pos,
            value_tensor=time_input,
            attention_mask=time_attention_mask
        )
        time_seq_mask = item_seq.gt(0).unsqueeze(2).expand_as(time_seq)
        time_seq_mean = torch.mean(time_seq_mask * time_seq, dim=1)
        time_short = self.gather_indexes(time_seq, item_seq_len - 1)
        time_session = self.time_soft_attention(
            mask=time_seq_mask,
            short_feature=time_short,
            seq_feature=time_seq,
            affinity_feature=time_seq_mean
        )
        time_session = F.normalize(time_session, dim=-1)
        return time_session

    def calculate_loss(self, interaction):
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_seq = interaction[self.ITEM_SEQ]
        seq_output = self.forward(item_seq_len, item_seq)
        pos_items = interaction[self.POS_ITEM_ID]
        test_item_emb = self.item_embedding.weight
        test_item_emb = self.dropout_layer[1](test_item_emb)
        test_item_emb = F.normalize(test_item_emb, dim=-1)
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1)) / self.temperature_parameter
        ce_loss = self.ce_loss(logits, pos_items)
        total_loss = ce_loss
        return total_loss

    def predict(self, interaction):
        test_item = interaction[self.ITEM_ID]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_seq = interaction[self.ITEM_SEQ]
        seq_output = self.forward(item_seq_len, item_seq)
        test_item_emb = self.item_embedding(test_item)
        test_item_emb = F.normalize(test_item_emb, dim=-1)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1) / self.temperature_parameter
        return scores

    def full_sort_predict(self, interaction):
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_seq = interaction[self.ITEM_SEQ]
        seq_output = self.forward(item_seq_len, item_seq)
        test_items_emb = self.item_embedding.weight
        test_items_emb = F.normalize(test_items_emb, dim=-1)
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1)) / self.temperature_parameter
        return scores
