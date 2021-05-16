import tensorflow as tf
import numpy as np

from utils.config_manager import Config
from data.datasets import ASRDataset
from utils.logging_utils import SummaryManager
from utils.scripts_utils import dynamic_memory_allocation, basic_train_parser
from ctc_segmentation import ctc_segmentation, determine_utterance_segments
from ctc_segmentation import CtcSegmentationParameters
from ctc_segmentation import prepare_token_list
import tgt

np.random.seed(42)
tf.random.set_seed(42)
dynamic_memory_allocation()


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
                                      kind='valid')
dataset = data_handler.get_dataset(bucket_batch_sizes=config_dict['bucket_batch_sizes'],
                                   bucket_boundaries=config_dict['bucket_boundaries'],
                                   shuffle=False)

# create logger and checkpointer and restore latest model
summary_manager = SummaryManager(model=model, log_dir=config.log_dir, config=config_dict)
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

if config_dict['debug'] is True:
    print('\nWARNING: DEBUG is set to True. Training in eager mode.')

for spk, mels, phonemes, mel_len, phon_len, fname in dataset.all_batches():
    phon_len = phon_len.numpy()
    mel_len = mel_len.numpy()
    phonemes = phonemes.numpy()
    for j, mel in enumerate(mels):
        # print(fname[j], phonemes[j], phon_len[j])
        temp = []
        for phoneme in phonemes[j]:
            if(phoneme!=358):
                temp.append(phoneme)
            else:
                phon_len[j] -= 1
        phonemes[j][:len(temp)] = temp

        model_out = model.predict(mel[np.newaxis, :mel_len[j], ...])
        pred_phon = model_out['encoder_output'][0]
        pred_phon = tf.nn.log_softmax(pred_phon)
        iphon_tar = model.text_pipeline.tokenizer.decode(phonemes[j][:phon_len[j]])
        iphon_tar = iphon_tar.split()
        
        char_list = [''] +list(model.text_pipeline.tokenizer.idx_to_token.values())
        config = CtcSegmentationParameters(char_list=char_list)
        config.index_duration = 0.0115545
        
        text = [phonemes[j][:phon_len[j]]]
        ground_truth_mat, utt_begin_indices = prepare_token_list(config, text)
        timings, char_probs, state_list = ctc_segmentation(config, pred_phon.numpy(), ground_truth_mat)
        utt_begin_indices = list(range(2, len(timings)))
        segments = determine_utterance_segments(
            config, utt_begin_indices, char_probs, timings, text[0]
        )

        tg = tgt.core.TextGrid('haa')
        tier = tgt.core.IntervalTier(name='phonemes')

        if(segments[0][-1]<-0.001):
            segments[0] = (0, segments[0][1], segments[0][2])
        else:
            itv = tgt.core.Interval(0, segments[0][0], text='sp')
            tier.add_interval(itv)

        if(segments[-1][-1]<-0.001):
            segments[-1] = (segments[-1][0], segments[-1][1]+0.15, segments[-1][2])
            if(segments[-1][1]>mel_len[j]*config.index_duration):
                pass
            else:
                itv = tgt.core.Interval(segments[-1][1], mel_len[j]*config.index_duration, text='sp') #mel_len[j]*config.index_duration
                tier.add_interval(itv)

        for i, seg in enumerate(segments):
            itv = tgt.core.Interval(seg[0], seg[1], text=iphon_tar[i])
            tier.add_interval(itv)

        tg.add_tier(tier)
        tgt.io.write_to_file(tg, f"textgrid/{fname[j].numpy().decode()}.textgrid", format='long')
        # break

print('Done.')
