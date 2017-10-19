import attr
import datetime as dt
import numpy as np
import pathlib

import torch
from torch.autograd import Variable


def set_seed(config, seed=None):
    """Set seed for Torch, CUDA and NumPy"""
    if seed is None:
        seed = config.seed
    torch.manual_seed(seed)
    if config.cuda:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def random_draw_use_teaching_forcing(use_teacher_forcing_perc):
    return np.random.uniform(0, 1) < use_teacher_forcing_perc


def maybe_cuda(x, cuda):
    """Helper for converting to a Variable"""
    if cuda:
        x = x.cuda()
    return x


def save_model(encoder, decoder, model_path):
    if not isinstance(model_path, pathlib.Path):
        model_path = pathlib.Path(model_path)
    model_path.mkdir(exist_ok=True, parents=True)
    torch.save(encoder.state_dict(), str(model_path / "encoder.ckpt"))
    torch.save(decoder.state_dict(), str(model_path / "decoder.ckpt"))


def load_model(encoder, decoder, model_path):
    if not isinstance(model_path, pathlib.Path):
        model_path = pathlib.Path(model_path)
    encoder.load_state_dict(torch.load(str(model_path / "encoder.ckpt")))
    decoder.load_state_dict(torch.load(str(model_path / "decoder.ckpt")))


def checkpoint_filename(config, epoch_and_batch_i=None, prefix="", suffix=""):
    if epoch_and_batch_i:
        epoch_and_batch_i_string = "{}x{}".format(
            *epoch_and_batch_i)
    else:
        epoch_and_batch_i_string = ""

    # Python doesn't support milliseconds in format...
    return "{}{}___{}___{}{}".format(
        prefix,
        config.initialized_at.strftime("%Y.%m.%d.%H.%M.%S.%f")[:-3],
        epoch_and_batch_i_string,
        dt.datetime.now().strftime("%Y.%m.%d.%H.%M.%S.%f")[:-3],
        suffix,
    )


def _is_true(x):
    return x == "True"


def argparse_attr(default=attr.NOTHING, validator=None,
                  repr=True, cmp=True, hash=True, init=True,
                  convert=None, opt_string=None,
                  **argparse_kwargs):
    if opt_string is None:
        opt_string_ls = []
    elif isinstance(opt_string, str):
        opt_string_ls = [opt_string]
    else:
        opt_string_ls = opt_string

    if argparse_kwargs.get("type", None) is bool:
        argparse_kwargs["choices"] = {True, False}
        argparse_kwargs["type"] = _is_true

    return attr.attr(
        default=default,
        validator=validator,
        repr=repr,
        cmp=cmp,
        hash=hash,
        init=init,
        convert=convert,
        metadata={
            "opt_string_ls": opt_string_ls,
            "argparse_kwargs": argparse_kwargs,
        }
    )


def update_parser(parser, class_with_attributes):
    for attribute in class_with_attributes.__attrs_attrs__:
        if "argparse_kwargs" in attribute.metadata:
            argparse_kwargs = attribute.metadata["argparse_kwargs"]
            opt_string_ls = attribute.metadata["opt_string_ls"]
            if attribute.default is attr.NOTHING:
                argparse_kwargs = argparse_kwargs.copy()
                argparse_kwargs["required"] = True
            else:
                argparse_kwargs["default"] = attribute.default
            parser.add_argument(
                f"--{attribute.name}", *opt_string_ls,
                **argparse_kwargs
            )


def read_parser(parser, class_with_attributes, skip_non_class_attributes=False):
    attribute_name_set = {
        attribute.name
        for attribute in class_with_attributes.__attrs_attrs__
    }

    kwargs = dict()
    leftover_kwargs = dict()

    for k, v in vars(parser.parse_args()).items():
        if k in attribute_name_set:
            kwargs[k] = v
        else:
            if not skip_non_class_attributes:
                raise RuntimeError(f"Unknown attribute {k}")
            leftover_kwargs[k] = v

    instance = class_with_attributes(**kwargs)
    if skip_non_class_attributes:
        return instance, leftover_kwargs
    else:
        return instance

