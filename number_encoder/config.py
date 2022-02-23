ATTR_CONFIGS = {

    # CharLSTM
    'CharLSTM_base': {
        'model_name': 'CharLSTM',
        'out_emb_size': 768,
        'hidden_size': 128,
        'lstm_num_layers': 1,
        'bidrectional': True
    },
    'CharLSTM_large': {
        'model_name': 'CharLSTM',
        'out_emb_size': 1024,
        'hidden_size': 128,
        'lstm_num_layers': 1,
        'bidrectional': True
    },
    'CharLSTM_1M_base': {
        'model_name': 'CharLSTM',
        'hidden_size': 128,
        'lstm_num_layers': 3,
        'bidrectional': True
    },
    'CharLSTM_1M_large': {
        'model_name': 'CharLSTM',
        'hidden_size': 128,
        'lstm_num_layers': 3,
        'bidrectional': True
    },
    'CharLSTM_9M_base': {
        'model_name': 'CharLSTM',
        'hidden_size': 208,
        'lstm_num_layers': 9,
        'bidrectional': True
    },
    'CharLSTM_9M_large': {
        'model_name': 'CharLSTM',
        'hidden_size': 208,
        'lstm_num_layers': 9,
        'bidrectional': True
    },
    'CharLSTM_100K_base': {
        'model_name': 'CharLSTM',
        'hidden_size': 32,
        'lstm_num_layers': 3,
        'bidrectional': True
    },
    'CharLSTM_100K_large': {
        'model_name': 'CharLSTM',
        'hidden_size': 32,
        'lstm_num_layers': 3,
        'bidrectional': True
    },

    # TransPos
    'TransPos': {
        'model_name': 'TransPos',
        'hidden_size': 32,
        'transformer_num_layers': 3,
        'transformer_nhead': 4,
        'direct_average': False
    },
    'TransPosAvg': {
        'model_name': 'TransPos',
        'hidden_size': 32,
        'transformer_num_layers': 3,
        'transformer_nhead': 4,
        'direct_average': True
    },
    'TransPos_1M_base': {
        'model_name': 'TransPos',
        'hidden_size': 64,
        'transformer_num_layers': 3,
        'transformer_nhead': 4,
        'direct_average': False
    },
    'TransPos_1M_large': {
        'model_name': 'TransPos',
        'hidden_size': 64,
        'transformer_num_layers': 3,
        'transformer_nhead': 4,
        'direct_average': False
    },
    'TransPos_9M_base': {
        'model_name': 'TransPos',
        'hidden_size': 128,
        'transformer_num_layers': 9,
        'transformer_nhead': 4,
        'direct_average': False
    },
    'TransPos_9M_large': {
        'model_name': 'TransPos',
        'hidden_size': 128,
        'transformer_num_layers': 9,
        'transformer_nhead': 4,
        'direct_average': False
    },
    'TransPos_100K_base': {
        'model_name': 'TransPos',
        'hidden_size': 8,
        'transformer_num_layers': 3,
        'transformer_nhead': 4,
        'direct_average': False
    },
    'TransPos_100K_large': {
        'model_name': 'TransPos',
        'hidden_size': 8,
        'transformer_num_layers': 3,
        'transformer_nhead': 4,
        'direct_average': False
    },

    # Streal
    'Streal': {
        'model_name': 'Streal',
        'hidden_size': 32,
        'transformer_num_layers': 3,
        'transformer_nhead': 4,
    },
    'Streal_1M_base': {
        'model_name': 'Streal',
        'hidden_size': 64,
        'transformer_num_layers': 3,
        'transformer_nhead': 4,
    },
    'Streal_1M_large': {
        'model_name': 'Streal',
        'hidden_size': 64,
        'transformer_num_layers': 3,
        'transformer_nhead': 4,
    },
    'Streal_9M_base': {
        'model_name': 'Streal',
        'hidden_size': 128,
        'transformer_num_layers': 9,
        'transformer_nhead': 4,
    },
    'Streal_9M_large': {
        'model_name': 'Streal',
        'hidden_size': 128,
        'transformer_num_layers': 9,
        'transformer_nhead': 4,
    },


    # SemLit
    'SemLit': {
        'model_name': 'SemLit',
        'hidden_size': 128,
        'transformer_num_layers': 3,
        'transformer_nhead': 4,
    },
    'SemLit_1M_base': {
        'model_name': 'SemLit',
        'hidden_size': 64,
        'transformer_num_layers': 3,
        'transformer_nhead': 4,
    },
    'SemLit_1M_large': {
        'hidden_size': 64,
        'transformer_num_layers': 3,
        'transformer_nhead': 4,
    },
    'SemLit_9M_base': {
        'model_name': 'SemLit',
        'hidden_size': 128,
        'transformer_num_layers': 9,
        'transformer_nhead': 4,
    },
    'SemLit_9M_large': {
        'hidden_size': 128,
        'transformer_num_layers': 9,
        'transformer_nhead': 4,
    },


    # TutaFeat
    'TutaFeat_1M_base': {
        'model_name': 'TutaFeat',
        'hidden_size': 546
    },
    'TutaFeat_1M_large': {
        'model_name': 'TutaFeat',
        'hidden_size': 448
    },
    'TutaFeat_9M_base': {
        'model_name': 'TutaFeat',
        'hidden_size': 2304
    },
    'TutaFeat_9M_large': {
        'model_name': 'TutaFeat',
        'hidden_size': 2176
    }
}

MODEL_NAMES = ATTR_CONFIGS.keys()


class NumBedConfig:
    def __init__(self, model_name: str,
                 encoder_name: str = 'TaPas',
                 model_suffix: str = 'Bi',
                 preprocess_type: str = 'trivial',
                 mode: str = 'sigexp',
                 lstm_num_layers: int = 3,
                 transformer_num_layers: int = 3,
                 transformer_nhead: int = 4,
                 emb_size: int = 768,
                 hidden_size: int = 128,
                 direct_expand: bool = False,
                 direct_average: bool = False,
                 checkpoint_path: str = '',
                 value_ratio: float = 0.25,
                 decoding_exp_class: int = 10,
                 addition_exp_class: int = 10,
                 listmax_class: int = 3,
                 format_frac_digit_class: int = 4,
                 format_in01_class: int = 2,
                 format_in0100_class: int = 2,
                 cp_class: int = 3,
                 cs_class: int = 3,
                 mix: str = 'cat',
                 aligned: bool = False,
                 use_layer_norm: bool = False,
                 align_with_orig: bool = False,
                 out_emb_size: int = 768):
        """[summary]

        :param model_name: [description]
        :type model_name: str
        :param encoder_name: [description]
        :type encoder_name: str
        :param model_suffix: [description], defaults to 'Bi'
        :type model_suffix: str, optional
        :param preprocess_type: [description], defaults to 'trivial'
        :type preprocess_type: str, optional
        :param mode: [description], defaults to 'sigexp'
        :type mode: str, optional
        :param lstm_num_layers: [description], defaults to 3
        :type lstm_num_layers: int, optional
        :param transformer_num_layers: [description], defaults to 3
        :type transformer_num_layers: int, optional
        :param transformer_nhead: [description], defaults to 3
        :type transformer_nhead: int, optional
        :param emb_size: [description], defaults to 768
        :type emb_size: int, optional
        :param hidden_size: [description], defaults to 128
        :type hidden_size: int, optional
        :param direct_expand: [description], defaults to False
        :type direct_expand: bool, optional
        :param direct_average: [description], defaults to False
        :type direct_average: bool, optional
        :param checkpoint_path: [description], defaults to ''
        :type checkpoint_path: str, optional
        :param value_ratio: [description], defaults to 0.25
        :type value_ratio: float, optional
        :param decoding_exp_class: [description], defaults to 10
        :type decoding_exp_class: int, optional
        :param addition_exp_class: [description], defaults to 10
        :type addition_exp_class: int, optional
        :param listmax_class: [description], defaults to 3
        :type listmax_class: int, optional
        :param format_frac_digit_class: [description], defaults to 4
        :type format_frac_digit_class: int, optional
        :param format_in01_class: [description], defaults to 2
        :type format_in01_class: int, optional
        :param format_in0100_class: [description], defaults to 2
        :type format_in0100_class: int, optional
        :param cp_class: [description], defaults to 3
        :type cp_class: int, optional
        :param cs_class: [description], defaults to 3
        :type cs_class: int, optional
        :param mix: [description], defaults to 'cat'
        :type mix: str, optional
        :param aligned: [description], defaults to False
        :type aligned: bool, optional
        :param use_layer_norm: [description], defaults to False
        :type use_layer_norm: bool, optional
        :param align_with_orig: [description], defaults to False
        :type align_with_orig: bool, optional
        """
        self.model_name = model_name
        self.encoder_name = encoder_name
        self.model_suffix = model_suffix
        self.preprocess_type = preprocess_type
        self.mode = mode
        self.lstm_num_layers = lstm_num_layers
        self.transformer_num_layers = transformer_num_layers
        self.transformer_nhead = transformer_nhead
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.bidirectional = True if 'Bi' in model_suffix.split('_') else False
        self.direct_expand = direct_expand
        self.direct_average = direct_average
        self.checkpoint_path = checkpoint_path
        self.value_ratio = value_ratio
        self.decoding_exp_class = decoding_exp_class
        self.addition_exp_class = addition_exp_class
        self.listmax_class = listmax_class
        self.format_frac_digit_class = format_frac_digit_class
        self.format_in01_class = format_in01_class
        self.format_in0100_class = format_in0100_class
        self.cp_class = cp_class
        self.cs_class = cs_class
        self.mix = mix
        self.aligned = aligned
        self.use_layer_norm = use_layer_norm
        self.align_with_orig = align_with_orig
        self.hidden_size = hidden_size
        self.out_emb_size = out_emb_size

        if model_name in MODEL_NAMES:
            self.from_existing(model_name)

    def from_existing(self, model_name: str = 'TransPos'):

        if model_name.split('_')[-1] == 'base':
            assert self.encoder_name in ['TaPas', 'BERT']
        if model_name.split('_')[-1] == 'large':
            assert self.encoder_name in ['RoBERTa']

        # Set up all the parameters for the specific model_name
        for k, v in ATTR_CONFIGS[model_name].items():
            setattr(self, k, v)

        # Set the output embedding size (determined by encoder)
        if self.encoder_name in ['TaPas', 'BERT']:
            self.out_emb_size = 768
        elif self.encoder_name in ['RoBERTa']:
            self.out_emb_size = 1024

    def get_model_id(self):
        if self.model_name in ['CharLSTM', 'Hybrid']:
            return '_'.join([self.model_name, self.model_suffix,
                             self.preprocess_type, self.mode])
        if self.model_name in ['RoBERTa', 'ValueEmbedding', 'Dice']:
            return '_'.join([self.model_name, self.preprocess_type])
