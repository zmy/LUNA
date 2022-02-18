import torch
import torch.nn as nn
import torch.nn.functional as F
from number_encoder import NumBed


class TabFormerConcatEmbeddings(nn.Module):
    """TabFormerConcatEmbeddings: Embeds tabular data of categorical variables

        Notes: - All column entries must be integer indices in a vocabolary that is common across columns
               - `sparse=True` in `nn.Embedding` speeds up gradient computation for large vocabs

        Args:
            config.ncols
            config.vocab_size
            config.hidden_size

        Inputs:
            - **input_ids** (batch, seq_len, ncols): tensor of batch of sequences of rows

        Outputs:
            - **output'**: (batch, seq_len, hidden_size): tensor of embedded rows
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.field_hidden_size,
                                            padding_idx=getattr(config, 'pad_token_id', 0), sparse=False)
        self.lin_proj = nn.Linear(config.field_hidden_size * config.ncols, config.hidden_size)

        self.hidden_size = config.hidden_size
        self.field_hidden_size = config.field_hidden_size

    def forward(self, input_ids):
        input_shape = input_ids.size()

        embeds_sz = list(input_shape[:-1]) + [input_shape[-1] * self.field_hidden_size]
        inputs_embeds = self.lin_proj(self.word_embeddings(input_ids).view(embeds_sz))

        return inputs_embeds


class TabFormerEmbeddings(nn.Module):
    """TabFormerEmbeddings: Embeds tabular data of categorical variables

        Notes: - All column entries must be integer indices in a vocabolary that is common across columns

        Args:
            config.ncols
            config.num_layers (int): Number of transformer layers
            config.vocab_size
            config.hidden_size
            config.field_hidden_size

        Inputs:
            - **input** (batch, seq_len, ncols): tensor of batch of sequences of rows

        Outputs:
            - **output**: (batch, seq_len, hidden_size): tensor of embedded rows
    """

    def __init__(self, config):
        super().__init__()

        if not hasattr(config, 'num_layers'):
            config.num_layers = 1
        if not hasattr(config, 'nhead'):
            config.nhead = 8

        self.word_embeddings = nn.Embedding(config.vocab_size, config.field_hidden_size,
                                            padding_idx=getattr(config, 'pad_token_id', 0), sparse=False)

        encoder_layer = nn.TransformerEncoderLayer(d_model=config.field_hidden_size, nhead=config.nhead,
                                                   dim_feedforward=config.field_hidden_size)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        self.lin_proj = nn.Linear(config.field_hidden_size * config.ncols, config.hidden_size)

    def forward(self, input_ids):
        inputs_embeds = self.word_embeddings(input_ids)
        embeds_shape = list(inputs_embeds.size())

        inputs_embeds = inputs_embeds.view([-1] + embeds_shape[-2:])
        inputs_embeds = inputs_embeds.permute(1, 0, 2)
        inputs_embeds = self.transformer_encoder(inputs_embeds)
        inputs_embeds = inputs_embeds.permute(1, 0, 2)
        inputs_embeds = inputs_embeds.contiguous().view(embeds_shape[0:2] + [-1])

        inputs_embeds = self.lin_proj(inputs_embeds)

        return inputs_embeds


class NumNetTabFormerEmbeddings(nn.Module):
    """TabFormerEmbeddings: Embeds tabular data of categorical variables

        Notes: - All column entries must be integer indices in a vocabolary that is common across columns

        Args:
            config.ncols
            config.num_layers (int): Number of transformer layers
            config.vocab_size
            config.hidden_size
            config.field_hidden_size

        Inputs:
            - **input** (batch, seq_len, ncols): tensor of batch of sequences of rows

        Outputs:
            - **output**: (batch, seq_len, hidden_size): tensor of embedded rows
    """

    def __init__(self, config, number_model_config, tokenizer, mlm_probability=0.15):
        super().__init__()

        if not hasattr(config, 'num_layers'):
            config.num_layers = 1
        if not hasattr(config, 'nhead'):
            config.nhead = 8

        self.data_type = config.data_type

        # For masking.
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability

        self.word_embeddings = nn.Embedding(config.vocab_size + 1, config.field_hidden_size,
                                            padding_idx=getattr(config, 'pad_token_id', 0), sparse=False)
        self.use_replace = config.use_replace
        self.num_id = config.vocab_size
        self.number_model = NumBed(number_model_config)

        encoder_layer = nn.TransformerEncoderLayer(d_model=config.field_hidden_size, nhead=config.nhead,
                                                   dim_feedforward=config.field_hidden_size)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        self.lin_proj = nn.Linear(config.field_hidden_size * config.ncols, config.hidden_size)

    def forward(self, input_ids, tokenized_number, mlm=True):
        """For Fraudulent Dataset, we extract raw numbers in columns 'Time', 'Amount', 'Zip', 'MCC'.
        For PRSA Dataset, we extract raw numbers in columns 'Time', 'SO2', 'NO2', 'CO', 'O3', 'TEMP',
        'PRES', 'DEWP', 'RAIN', 'WSPM'.
        """
        if self.use_replace:
            new_input_ids = input_ids.clone()
            if self.data_type == 'card':
                new_input_ids[:, :, [2, 3, 8, 9]] = self.num_id
            elif self.data_type == 'prsa':
                new_input_ids[:, :, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]] = self.num_id
            else:
                raise NotImplementedError
        else:
            new_input_ids = input_ids
        inputs_embeds = self.word_embeddings(new_input_ids)
        number_embeds = self.number_model(tokenized_number)
        num_id = torch.zeros_like(input_ids).long()

        if self.data_type == 'card':
            num_id[:, :, [2, 3, 8, 9]] = torch.arange(1, number_embeds.shape[0] + 1).reshape(
                input_ids.shape[0], input_ids.shape[1], -1).cuda()
        elif self.data_type == 'prsa':
            num_id[:, :, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]] = torch.arange(1, number_embeds.shape[0] + 1).reshape(
                input_ids.shape[0], input_ids.shape[1], -1).cuda()
        else:
            raise NotImplementedError

        number_embeds = F.embedding(num_id, torch.cat((torch.zeros(1, number_embeds.shape[1]).cuda(), number_embeds)),
                                    padding_idx=0)
        # Add number embeddings.
        inputs_embeds = inputs_embeds + number_embeds

        # Mask language model.
        if mlm:
            inputs_embeds, masked_indices = self.mask(input_ids, inputs_embeds, self.word_embeddings)

        embeds_shape = list(inputs_embeds.size())

        inputs_embeds = inputs_embeds.view([-1] + embeds_shape[-2:])
        inputs_embeds = inputs_embeds.permute(1, 0, 2)
        inputs_embeds = self.transformer_encoder(inputs_embeds)
        inputs_embeds = inputs_embeds.permute(1, 0, 2)
        inputs_embeds = inputs_embeds.contiguous().view(embeds_shape[0:2] + [-1])

        inputs_embeds = self.lin_proj(inputs_embeds)

        if mlm:
            return inputs_embeds, masked_indices
        else:
            return inputs_embeds

    def mask(self, input_ids, input_embeddings, embeddings_table):
        probability_matrix = torch.full(input_ids.shape, self.mlm_probability).to(input_embeddings)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        if self.tokenizer._pad_token is not None:
            masked_indices[input_ids == self.tokenizer.pad_token_id] = False

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(
            torch.full(input_ids.shape, 0.8).to(input_embeddings)).bool() & masked_indices

        input_embeddings = torch.where(indices_replaced.unsqueeze(-1),
                                       embeddings_table(torch.tensor(
                                           self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)).to(
                                           input_ids)), input_embeddings)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(
            torch.full(input_ids.shape, 0.5).to(input_embeddings)).bool() & masked_indices & ~indices_replaced

        random_words = torch.randint(len(self.tokenizer), input_ids.shape).to(input_ids)

        input_embeddings = torch.where(indices_random.unsqueeze(-1), embeddings_table(random_words), input_embeddings)

        labels = input_ids.clone()
        labels[~masked_indices] = -100

        return input_embeddings, input_ids
