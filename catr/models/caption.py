import torch
from torch import nn
import torch.nn.functional as F

from .utils import NestedTensor, nested_tensor_from_tensor_list
from .backbone import build_backbone
from .transformer import build_transformer, generate_square_subsequent_mask


class Caption(nn.Module):
    def __init__(self, backbone, transformer, hidden_dim, vocab_size):
        super().__init__()
        self.backbone = backbone
        self.input_proj = nn.Conv2d(
            backbone.num_channels, hidden_dim, kernel_size=1)
        self.transformer = transformer
        self.mlp = MLP(hidden_dim, 512, vocab_size, 3)
        self.hidden_dim = hidden_dim

    def forward(self, samples, target, target_mask, max_tgt_len=24, sample_k=5,end_token_id=102,pad_token_id=0,is_test=False):
        if is_test is True:
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        device = target.device
        src, mask = features[-1].decompose()
        bs = src.shape[0]  # batch_size
        assert mask is not None
        hs, memory, pos_embed, query_embed = self.transformer(self.input_proj(src), mask,
                                                              pos[-1], target, target_mask)
        output = self.mlp(hs.permute(1, 0, 2))  # bs * max_len * vocab_size
        out = output[:, 0, :]  # get the first output  -> shape: bs * vocab_size

        # ========== Recurrent Decoder ===========
        # decoder original input embeds
        input_embeds = self.transformer.embeddings.word_embeddings(target)
        # decoder input mask
        mask = mask.flatten(1)  # memory_padding_mask
        # finished flag settings
        finished = [0 for _ in range(bs)]  # 1 if finish generating caption
        end_token_embedding = self.transformer.embeddings.word_embeddings(
                torch.tensor(end_token_id, device=device))
        pad_token_embedding = self.transformer.embeddings.word_embeddings(
                torch.tensor(pad_token_id, device=device))
        # recurrent decode process: output 1~max_len
        for i in range(1, max_tgt_len):
            # get out token id
            max_out_id = torch.argmax(out, dim=-1)
            sorted_out_id = torch.argsort(out, dim=-1, descending=True)  # shape: bs*vocab_size
            # ========== top-k sampling ==========
            # 1: get all the outputs of tokens in of top-k
            sample_out_id = sorted_out_id[:, :sample_k]
            sample_out = torch.zeros((bs, sample_k), dtype=torch.float, device=device)
            for bi in range(bs):
              sample_out[bi] = out[bi, sample_out_id[bi]]
            # 2: use 'softmax' to get top-k tokens' probability , other:0
            out = torch.softmax(sample_out, dim=-1)
            out = out.unsqueeze(2)    # bs*k*1
            # 3: get top-k tokens embedding and do weighted average(The weights are the outputs of softmax)
            sample_token_embedding = self.transformer.embeddings.word_embeddings(sample_out_id)  # bs*k*hidden_dim
            sample_token_embedding = (out*sample_token_embedding).sum(dim=1) # element-wise product: broadcast
            # print(sample_token_embedding.shape)   # bs * hidden_dim
            # 4: check if finished
            for j in range(bs):
                if finished[j]:  # already finished
                    sample_token_embedding[j] = pad_token_embedding
                    target_mask[j][i] = True  # mask tokens i
                elif max_out_id[j] == end_token_id:  # output_end_token
                    finished[j] = i-1
                    sample_token_embedding[j] = end_token_embedding
                    target_mask[j][i] = False
                else:
                    target_mask[j][i] = False
            if min(finished) > 0:
                # if all batches finished, then break and return
                break
            # 5 get new input_embeds
            input_embeds[:, i, :] = sample_token_embedding  # bs*128*hidden_dim
            # ========== transformer decoder ============
            # put sample embeddings into decoder embedding
            tgt = self.transformer.embeddings(target, input_embeds=input_embeds).permute(1, 0, 2)
            tgt_mask = generate_square_subsequent_mask(len(tgt)).to(tgt.device)
            hs = self.transformer.decoder(tgt, memory,
                                          memory_key_padding_mask=mask,
                                          tgt_key_padding_mask=target_mask,
                                          pos=pos_embed,
                                          query_pos=query_embed,
                                          tgt_mask=tgt_mask)
            output = self.mlp(hs.permute(1, 0, 2))
            out = output[:, i, :]  # the i-th output

        # check finished
        if min(finished) == 0:
            for i in range(bs):
                if not finished[i]:
                    target_mask[i][max_tgt_len] = False
                    max_out_id = torch.argmax(out, dim=-1)
                    if max_out_id[i] == end_token_id:
                        finished[i] = max_tgt_len-1

        # return
        return memory, output[:, :max_tgt_len], hs[:max_tgt_len], target_mask[:, 1:max_tgt_len+1], finished


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_model(config):
    backbone = build_backbone(config)
    transformer = build_transformer(config)

    model = Caption(backbone, transformer, config.hidden_dim, config.vocab_size)
    criterion = torch.nn.CrossEntropyLoss()

    return model, criterion
