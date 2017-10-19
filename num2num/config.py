import attr
import argparse
import datetime as dt
import pathlib
import torch

from .utils import argparse_attr, update_parser, read_parser


@attr.s
class Configuration:

    # Data
    train_data_path = argparse_attr(
        default=None, type=str,
        help="Path to training data (.csv)")
    val_data_path = argparse_attr(
        default=None, type=str,
        help="Path to val data (.csv)")
    tokens_path = argparse_attr(
        default=(
            pathlib.Path(__file__).parents[1] / "datafiles" /
            "word2num_tokens.json"
        ), type=str,
        help="Path to tokens (.json)")
    model_path = argparse_attr(
        default=None,
        help="Pre-trained model",
    )

    # Problem definition
    word_to_num = argparse_attr(
        default=False, type=bool,
        help="True: Translate from words to number. False: Vice-versa")
    char_by_char = argparse_attr(
        default=False, type=bool,
        help="True: char-to-char translation")

    # Training hyper-parameters
    cuda = argparse_attr(
        default=torch.has_cudnn, type=bool,
        help="Whether to use CUDA")
    batch_size = argparse_attr(
        default=256, type=int,
        help="Batch size in training/evaluation")
    learning_rate = argparse_attr(
        default=0.001, type=float,
        help="Learning Rate")
    use_teacher_forcing_perc = argparse_attr(
        default=0.5, type=float,
        help="Teaching forcing ratio. 1.0 = always use teaching forcing in"
             " training")
    epochs = argparse_attr(
        default=1000, type=int,
        help="Number of training epochs")
    inspect_every = argparse_attr(
        default=500, type=int,
        help="Inspect model (training batch loss, plot attention, etc) "
             "every N iterations")
    validate_every = argparse_attr(
        default=500, type=int,
        help="Run model against validation set every N iterations")
    seed = argparse_attr(
        default=1, type=int,
        help="Global random seed")

    # Model hyper-parameters
    num_layers = argparse_attr(
        default=2, type=int,
        help="Number of layers in RNN")
    rnn_size = argparse_attr(
        default=200, type=int,
        help="Number of nodes in RNN layer")
    bidirectional_encoder = argparse_attr(
        default=True, type=bool,
        help="Whether to use bidirectional RNN for encoder")
    attn_decoder = argparse_attr(
        default=True, type=bool,
        help="Whether to use  an attentional decoder")
    attn_method = argparse_attr(
        default="concat", type=str,
        help="Type of attention model.",
        choices={"concat", "dot", "global"},
    )

    # Output configuration
    verbosity = argparse_attr(
        default=4, type=int,
        help="Verbosity of output, higher=more verbose")
    model_save_path = argparse_attr(
        default=None,
        help="Where to save models")
    model_save_every = argparse_attr(
        default=500, type=int,
        help="Save model every N iterations")
    plot_attn = argparse_attr(
        default=True, type=bool,
        help="Whether to plot attention (randomly sampled)")
    plot_attn_show = argparse_attr(
        default=True, type=bool,
        help="Whether to show attention plots in session")
    plot_attn_save_path = argparse_attr(
        default=None,
        help="Where to save attention plots")

    # batch_size = 1024
    # num_layers = 1
    # rnn_size = 64

    # Set post-processing
    input_vocab_size = attr.attr(default=None)
    output_vocab_size = attr.attr(default=None)
    initialized_at = attr.attr(default=None)

    def __attrs_post_init__(self):
        self.initialized_at = dt.datetime.now()

    @classmethod
    def parse_configuration(cls, prog=None, description=None):
        parser = argparse.ArgumentParser(
            prog=prog,
            description=description,
        )
        update_parser(
            parser=parser,
            class_with_attributes=cls,
        )
        return read_parser(
            parser=parser,
            class_with_attributes=cls,
        )
