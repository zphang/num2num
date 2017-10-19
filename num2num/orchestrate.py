import pathlib
import warnings

from num2num.data import get_data_loader
from num2num.utils import set_seed, save_model, checkpoint_filename
from num2num.operations import padded_criterion, train, compute_loss, \
    get_model, get_optimizers, sample


def run_train(config):
    print(config)
    set_seed(config)

    # --- Setup data loaders --- #
    train_dataloader = get_data_loader(
        data_path=config.train_data_path,
        tokens_path=config.tokens_path,
        config=config,
    )
    val_dataloader = get_data_loader(
        data_path=config.val_data_path,
        tokens_path=config.tokens_path,
        config=config,
    )

    # --- Setup models/optimizers/criterion --- #
    encoder, decoder = get_model(config)
    encoder_optimizer, decoder_optimizer = \
        get_optimizers(encoder, decoder, config)
    criterion = padded_criterion(config=config)

    # --- Train-loop --- #
    train_epoch_loss_ls = []
    val_epoch_loss_ls = []

    for epoch in range(config.epochs):
        if config.verbosity > 1:
            epoch_str = f"EPOCH: {epoch}"
            print("="*len(epoch_str))
            print(epoch_str)
            print("=" * len(epoch_str))

        train_loss = train(
            encoder=encoder, encoder_optimizer=encoder_optimizer,
            decoder=decoder, decoder_optimizer=decoder_optimizer,
            criterion=criterion,
            config=config,
            dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            epoch=epoch,
        )
        val_loss = compute_loss(
            encoder=encoder,
            decoder=decoder,
            criterion=criterion,
            config=config, dataloader=val_dataloader,
        )

        train_epoch_loss = train_loss
        val_epoch_loss = val_loss
        train_epoch_loss_ls.append(train_epoch_loss)
        val_epoch_loss_ls.append(val_epoch_loss)

        if config.verbosity > 2:
            print(f" ----- End of Epoch {epoch} ----- ")
            print(f"Training loss:   {train_epoch_loss}")
            print(f"Validation loss: {val_loss}")
            print("")

    if config.model_save_path:
        model_save_fol = (
            pathlib.Path(config.model_save_path) /
            checkpoint_filename(
                config=config,
                prefix="model_",
                suffix="final"
            )
        )
        save_model(
            encoder=encoder,
            decoder=decoder,
            model_path=model_save_fol,
        )
    return (
        (encoder, decoder),
        train_epoch_loss_ls,
        val_epoch_loss_ls,
    )


def run_sample(config):
    print(config)
    set_seed(config)

    # --- Setup data loaders --- #
    val_dataloader = get_data_loader(
        data_path=config.val_data_path,
        tokens_path=config.tokens_path,
        config=config,
    )

    # --- Setup models/optimizers/criterion --- #
    encoder, decoder = get_model(config)
    if not config.model_path:
        warnings.warn("Sampling with untrained model")
    criterion = padded_criterion(config)

    # --- Sample --- #
    sample_result_ls, loss = sample(
        encoder=encoder,
        decoder=decoder,
        dataloader=val_dataloader,
        config=config,
        criterion=criterion,
    )
    return sample_result_ls, loss
