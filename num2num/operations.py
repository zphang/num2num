import numpy as np
import pathlib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from .data import NOTHING_IDX
from .model import Encoder, Decoder, AttnDecoder
from .utils import maybe_cuda, random_draw_use_teaching_forcing, \
    checkpoint_filename, save_model, load_model
from .plotting import plot_attn_weights


def get_model(config):
    encoder = maybe_cuda(Encoder(config), cuda=config.cuda)
    if config.attn_decoder:
        decoder = maybe_cuda(AttnDecoder(config), cuda=config.cuda)
    else:
        decoder = maybe_cuda(Decoder(config), cuda=config.cuda)

    if config.model_path:
        load_model(encoder, decoder, model_path=config.model_path)
    return encoder, decoder


def get_optimizers(encoder, decoder, config):
    encoder_optimizer = optim.Adam(
        encoder.parameters(), lr=config.learning_rate)
    decoder_optimizer = optim.Adam(
        decoder.parameters(), lr=config.learning_rate)
    return encoder_optimizer, decoder_optimizer


def padded_criterion(config):
    weight = torch.ones(config.output_vocab_size)
    weight[NOTHING_IDX] = 0
    crit = maybe_cuda(nn.NLLLoss(weight), cuda=config.cuda)
    return crit


def run_model_on_batch(encoder, decoder,
                       config, use_teacher_forcing,
                       batch_data):
    (x, x_len), (y, y_len) = batch_data

    encoder_output, encoder_hidden = encoder(
        x=maybe_cuda(Variable(x, requires_grad=False), cuda=config.cuda),
        x_len=x_len,
        hidden=None,
    )

    y_var = maybe_cuda(Variable(y, requires_grad=False), cuda=config.cuda)
    target_length = y_var.size()[1]

    decoder_input = maybe_cuda(
        Variable(torch.LongTensor([[0]] * x.size()[0])), cuda=config.cuda)
    decoder_hidden = encoder_hidden

    decoder_output_ls = []
    attn_ls = []
    for di in range(target_length):
        decoder_output, decoder_hidden, other_dict = \
            decoder(decoder_input, decoder_hidden, encoder_output)
        decoder_output_ls.append(decoder_output)
        if "attn" in other_dict:
            attn_ls.append(other_dict["attn"])

        if use_teacher_forcing:
            decoder_input = y_var[:, di].unsqueeze(1)
        else:
            topv, topi = decoder_output.data.topk(1)
            decoder_input = maybe_cuda(
                Variable(topi.squeeze(1)), cuda=config.cuda)

    full_decoder_output = torch.cat(decoder_output_ls, dim=1)
    return full_decoder_output, attn_ls


def train(encoder, encoder_optimizer, decoder, decoder_optimizer, criterion,
          config, dataloader,
          val_dataloader=None, epoch=None):
    summed_batch_loss = 0.
    for batch_i, batch_data in enumerate(dataloader):
        (x, x_len), (y, y_len) = batch_data

        use_teacher_forcing = random_draw_use_teaching_forcing(
            config.use_teacher_forcing_perc)

        encoder.train()
        decoder.train()
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        full_decoder_output, _ = run_model_on_batch(
            encoder=encoder,
            decoder=decoder,
            config=config,
            use_teacher_forcing=use_teacher_forcing,
            batch_data=batch_data,
        )

        y_var = maybe_cuda(Variable(y, requires_grad=False), cuda=config.cuda)

        batch_loss = criterion(
            full_decoder_output.view(-1, full_decoder_output.size()[2]),
            y_var.view(-1),
        )
        batch_loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        summed_batch_loss += batch_loss.data[0] * len(x)

        if config.verbosity > 2:
            if batch_i % config.inspect_every == 0 and config.verbosity > 2:
                inspect_dataloader = val_dataloader \
                    if val_dataloader else dataloader
                print(" ----- Inspect ----- ")
                print(f"Iter: {(batch_i+1) * config.batch_size}/"
                      f"{len(dataloader.dataset)} (Batch: {batch_i})")
                print(f"Batch train loss: {batch_loss.data[0]}")
                inspect_model(
                    encoder=encoder, decoder=decoder,
                    batch_data=next(iter(inspect_dataloader)),
                    dataset=dataloader.dataset,
                    config=config,
                    epoch_and_batch_i=(epoch, batch_i),
                )
                print("")

            if batch_i % config.validate_every == 0:
                if not val_dataloader:
                    raise RuntimeError("No validation dataloader configured")
                val_loss = compute_loss(
                    encoder=encoder, decoder=decoder,
                    dataloader=val_dataloader,
                    config=config,
                    criterion=criterion,
                )
                print(" ----- Validate ----- ")
                print(f"Iter: {(batch_i+1) * config.batch_size}/"
                      f"{len(dataloader.dataset)} (Batch: {batch_i})")
                print(f"Validation loss: {val_loss}")
                print("")

            if config.model_save_path and \
                    batch_i % config.model_save_every == 0:
                model_save_fol = (
                    pathlib.Path(config.model_save_path) /
                    checkpoint_filename(
                        config=config,
                        epoch_and_batch_i=(epoch, batch_i),
                        prefix="model_",
                    )
                )
                save_model(
                    encoder=encoder,
                    decoder=decoder,
                    model_path=model_save_fol,
                )
                print("Model saved to {}".format(model_save_fol))

    return summed_batch_loss / len(dataloader.dataset)


def compute_loss(encoder, decoder, dataloader, config, criterion):
    encoder.eval()
    decoder.eval()
    summed_batch_loss = 0.
    for batch_i, batch_data in enumerate(dataloader):

        (x, x_len), (y, y_len) = batch_data
        full_decoder_output, _ = run_model_on_batch(
            encoder=encoder,
            decoder=decoder,
            config=config,
            use_teacher_forcing=False,
            batch_data=batch_data,
        )

        y_var = maybe_cuda(Variable(y, requires_grad=False), cuda=config.cuda)

        batch_loss = criterion(
            full_decoder_output.view(-1, full_decoder_output.size()[2]),
            y_var.view(-1),
        )
        summed_batch_loss += batch_loss.data[0] * len(x)
    return summed_batch_loss / len(dataloader.dataset)


def inspect_model(encoder, decoder, batch_data, dataset, config,
                  epoch_and_batch_i=None, num=0):
    (x, x_len), (y, y_len) = batch_data

    # We only inspect one element (default: first) of the batch
    single_elem_batch_data = (
        (x[num:num+1], x_len[num:num+1]),
        (y[num:num + 1], y_len[num:num + 1]),
    )

    encoder.eval()
    decoder.eval()

    full_decoder_output, attn_ls = run_model_on_batch(
        encoder=encoder, decoder=decoder,
        config=config, use_teacher_forcing=False,
        batch_data=single_elem_batch_data,
    )

    topv, topi = full_decoder_output.data.topk(1)
    pred_y = topi.squeeze(2)

    if config.verbosity > 3:
        print("Input string:\n    {}\n".format(
            dataset.input_lang.batch_tensor_to_string(x)[0]))
        print("Expected output:\n    {}\n".format(
            dataset.output_lang.batch_tensor_to_string(y)[0]))
        print("Predicted output:\n    {}\n".format(
            dataset.output_lang.batch_tensor_to_string(pred_y)[0]))

    if config.plot_attn:
        attn_weights = torch.stack(attn_ls)  # O*I*B
        attn_weights_mat = attn_weights[:, :, 0].cpu().data.numpy()

        if config.plot_attn_save_path:
            save_fol_path = pathlib.Path(config.plot_attn_save_path)
            save_fol_path.mkdir(exist_ok=True, parents=True)
            save_to = str(
                save_fol_path /
                checkpoint_filename(
                    config=config,
                    epoch_and_batch_i=epoch_and_batch_i,
                    prefix="Attn_",
                    suffix=".png"
                )
            )
            if config.verbosity > 3:
                print("Attention plot saved to {}".format(save_to))
        else:
            save_to = None

        plot_attn_weights(
            attn_weights_mat=attn_weights_mat,
            x=x,
            pred_y=pred_y,
            dataset=dataset,
            show=config.plot_attn_show,
            save_to=save_to,
        )


def sample(encoder, decoder, dataloader, config, criterion=None):
    encoder.eval()
    decoder.eval()

    sample_result_ls = []
    summed_batch_loss = 0.
    for batch_i, batch_data in enumerate(dataloader):
        (x, x_len), (y, y_len) = batch_data
        full_decoder_output, _ = run_model_on_batch(
            encoder=encoder,
            decoder=decoder,
            config=config,
            use_teacher_forcing=False,
            batch_data=batch_data,
        )
        topv, topi = full_decoder_output.data.topk(1)
        pred_y = topi.squeeze(2)

        sample_result_batch = list(zip(
            dataloader.dataset.input_lang.batch_tensor_to_string(x),
            dataloader.dataset.output_lang.batch_tensor_to_string(y),
            dataloader.dataset.output_lang.batch_tensor_to_string(pred_y),
        ))

        for x_text, y_text, pred_y_text in sample_result_batch:
            if config.verbosity > 3:
                print("Input string:\n    {}\n".format(x_text))
                print("Expected output:\n    {}\n".format(y_text))
                print("Predicted output:\n    {}\n".format(pred_y_text))

        if criterion:
            y_var = maybe_cuda(Variable(y, requires_grad=False),
                               cuda=config.cuda)
            batch_loss = criterion(
                full_decoder_output.view(-1, full_decoder_output.size()[2]),
                y_var.view(-1),
            )
            summed_batch_loss += batch_loss.data[0] * len(x)

        sample_result_ls += sample_result_batch
    if criterion:
        loss = summed_batch_loss / len(dataloader.dataset)
    else:
        loss = None
    return sample_result_ls, loss
