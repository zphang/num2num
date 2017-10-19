# Home

An example (Attentional) Encoder-Decoder RNN in PyTorch, applied to a toy data set: translating back and forth between numbers-as-text ("one thousand and two") and numbers-as-digits ("1002").

### Requirements
- Python 3.6
- [PyTorch 0.2.0](http://pytorch.org/)
- (For generating data) `num2words`, `tqdm`
- `pandas`, `argparse`, `matplotlib`, `seaborn`

### Data

Using `num2words`, we can generate random numbers and get the "ground-truth" text versions of those numbers. Commas are removed, and the remaining tokens have been pre-computed and saved in (`datafiles/word2num_tokens`)[datafiles/word2num_tokens]. 

Note that `num2num` supports both word-level and character-level modeling, so both tokenizations have been pre-computed.

### Usage

1. Git clone this repository.
2. (Optional) Generate training and validation datasets. A very small sample training and validation dataset is included with the project. 
    * E.g. 
        ```bash
        python gendata.py \
            --output=datafiles/train_numbers.csv \
            --size=100000

        python gendata.py \
            --output=datafiles/val_numbers.csv \ 
            --size=10000
        ```
    * Run `python gendata.py -h` or see `gendata.py` for details and more options.
3. Train the model:
    * E.g.
        ```bash
        python run_train.py \
          --train_data_path=datafiles/train_numbers.csv \
          --val_data_path=datafiles/val_numbers.csv \
          --plot_attn_show=False \
          --plot_attn_save_path="output/attn_plots" \
          --model_save_path="output/models
        ```
    * Run `python run_train.py -h` or see `orchestrate.py` and `num2num/config.py` for details and more options
4. Sample from model:
    * E.g.
        ```bash
        python run_sample.py \
          --val_data_path=datafiles/val_numbers.csv \
          --model_save_path="output/models/my_favorite_model
        ```
    * Run `python run_sample.py -h` or see `orchestrate.py` and `num2num/config.py` for details and more options
    
### Road map

- [ ] Documentation
- [ ] Tests