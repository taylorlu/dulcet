import argparse

import tensorflow as tf
import numpy as np
from tqdm import tqdm

from utils.config_manager import Config
from data.datasets import ASRDataset
from utils.scripts_utils import dynamic_memory_allocation, basic_train_parser
from ctc_segmentation import ctc_segmentation, determine_utterance_segments
from ctc_segmentation import CtcSegmentationParameters
from ctc_segmentation import prepare_token_list

np.random.seed(42)
tf.random.set_seed(42)
dynamic_memory_allocation()

# consuming CLI, creating paths and directories, load data
parser = basic_train_parser()
args = parser.parse_args()

config = Config(config_path=args.config, asr=True)
config_dict = config.config
config.create_remove_dirs(clear_dir=args.clear_dir,
                          clear_logs=args.clear_logs,
                          clear_weights=args.clear_weights)
config.dump_config()
config.print_config()

model = config.get_model()
config.compile_model(model)

data_handler = ASRDataset.from_config(config,
                                      tokenizer=model.text_pipeline.tokenizer,
                                      kind='phonemized')
dataset = data_handler.get_dataset(bucket_batch_sizes=config_dict['bucket_batch_sizes'],
                                   bucket_boundaries=config_dict['bucket_boundaries'],
                                   shuffle=True)

# create logger and checkpointer and restore latest model
checkpoint = tf.train.Checkpoint(step=tf.Variable(1),
                                 optimizer=model.optimizer,
                                 net=model)
manager = tf.train.CheckpointManager(checkpoint, config.weights_dir,
                                     max_to_keep=config_dict['keep_n_weights'],
                                     keep_checkpoint_every_n_hours=config_dict['keep_checkpoint_every_n_hours'])
manager_training = tf.train.CheckpointManager(checkpoint, str(config.weights_dir / 'latest'),
                                              max_to_keep=1, checkpoint_name='latest')

checkpoint.restore(manager_training.latest_checkpoint)
if manager_training.latest_checkpoint:
    print(f'\nresuming training from step {model.step} ({manager_training.latest_checkpoint})')
else:
    print(f'\nstarting training from scratch')

all_durations = np.array([])
iterator = tqdm(enumerate(dataset.all_batches()))
step = 0

char_list = [''] +list(model.text_pipeline.tokenizer.idx_to_token.values())
smt_config = CtcSegmentationParameters(char_list=char_list)
smt_config.index_duration = 0.0115545

for c, (spk_batch, mel_batch, phoneme_batch, mel_len_batch, phon_len_batch, fname_batch) in iterator:
    iterator.set_description(f'Processing dataset')

    model_out = model.predict(mel_batch)
    pred_phon = model_out['encoder_output']
    pred_phon = tf.nn.log_softmax(pred_phon)

    for i, name in enumerate(fname_batch):
        text = [phoneme_batch[i][:phon_len_batch[i]].numpy()]
        ground_truth_mat, utt_begin_indices = prepare_token_list(smt_config, text)
        timings, char_probs, state_list = ctc_segmentation(smt_config, pred_phon[i][:mel_len_batch[i]].numpy(), ground_truth_mat)
        utt_begin_indices = list(range(2, len(timings)))
        segments = determine_utterance_segments(
            smt_config, utt_begin_indices, char_probs, timings, text[0]
        )
        durations = []
        for j, segment in enumerate(segments):
            if(j==0):
                durations.append(round(segment[1]/smt_config.index_duration))
            elif(j==len(segments)-1):
                durations.append(round(mel_len_batch[i].numpy()-segment[0]/smt_config.index_duration))
            else:
                durations.append(round((segment[1] - segment[0])/smt_config.index_duration))

            np.save(str(config.duration_dir / f"{name.numpy().decode('utf-8')}.npy"), np.array(durations))

print('Done.')
