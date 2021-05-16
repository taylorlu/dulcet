import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os

from utils.config_manager import Config
from data.datasets import MelDurDataset
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

data_handler = MelDurDataset.from_config(config,
                                         tokenizer=model.text_pipeline.tokenizer,
                                         kind='phonemized')
dataset = data_handler.get_dataset(bucket_batch_sizes=config_dict['bucket_batch_sizes'],
                                   bucket_boundaries=config_dict['bucket_boundaries'],
                                   shuffle=False)

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

labelFile = open(r'/root/mydata/Corpus/transformer_tts_data.corpus/phonemized_metadata.NoStress2.txt', 'w')

for c, (spk_name_batch, mel_batch, phoneme_batch, mel_len_batch, phon_len_batch, fname_batch) in iterator:
    iterator.set_description(f'Processing dataset')

    model_out = model.predict(mel_batch)
    pred_phon = model_out['encoder_output']
    pred_phon = tf.nn.log_softmax(pred_phon)

    for i, name in enumerate(fname_batch):
        os.makedirs(os.path.join(config.duration_dir, spk_name_batch[i].numpy().decode()), exist_ok=True)

        text = list(phoneme_batch[i][:phon_len_batch[i]].numpy())
        while 358 in text:
            text.remove(358)
        text = [np.array(text)]

        ground_truth_mat, utt_begin_indices = prepare_token_list(smt_config, text)
        timings, char_probs, state_list = ctc_segmentation(smt_config, pred_phon[i][:mel_len_batch[i]].numpy(), ground_truth_mat)
        utt_begin_indices = list(range(2, len(timings)))
        segments = determine_utterance_segments(
            smt_config, utt_begin_indices, char_probs, timings, text[0]
        )

        durations = []
        if(segments[0][-1]<-0.001 or segments[0][1]<0.05):
            segments[0] = (0, segments[0][1], segments[0][2])
        else:
            durations.append([0, segments[0][0], 358])

        last_duration = None
        if(segments[-1][-1]<-0.001):
            segments[-1] = (segments[-1][0], segments[-1][1]+0.15, segments[-1][2])
            if(segments[-1][1]>mel_len_batch[i].numpy()*smt_config.index_duration or
                mel_len_batch[i].numpy()*smt_config.index_duration-segments[-1][1]<0.05):
                pass
            else:
                last_duration = [segments[-1][1], -1, 358]

        for j, seg in enumerate(segments):
            durations.append([seg[0], seg[1], phoneme_batch[i][j].numpy()])

        if(last_duration is not None):
            durations.append(last_duration)

        fr_lens = []
        phons = []
        for dur in durations[:-1]:
            fr_len = round((dur[1] - dur[0])/smt_config.index_duration)
            fr_lens.append(fr_len)
            phons.append(model.text_pipeline.tokenizer.idx_to_token[dur[-1]])

        if(sum(fr_lens)<mel_len_batch[i].numpy()):
            # the last phoneme should be justify.
            fr_len = mel_len_batch[i].numpy()-sum(fr_lens)
            fr_lens.append(fr_len)
            phons.append(model.text_pipeline.tokenizer.idx_to_token[dur[-1]])

            np.save(str(config.duration_dir / spk_name_batch[i].numpy().decode() / f"{name.numpy().decode('utf-8')}.npy"), np.array(fr_lens))
            phons = '{'+" ".join(phons)+'}'
            labelFile.writelines(f'{fname_batch[i].numpy().decode()}|{spk_name_batch[i].numpy().decode()}|{phons}\n')
        else:
            print(fname_batch[i].numpy().decode())
            print(durations)


print('Done.')
