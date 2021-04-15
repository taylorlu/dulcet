import argparse
import pickle

import tensorflow as tf
import numpy as np
from tqdm import tqdm
from p_tqdm import p_umap

from utils.config_manager import Config
from utils.logging_utils import SummaryManager
from data.datasets import AlignerPreprocessor
from utils.alignments import get_durations_from_alignment
from utils.scripts_utils import dynamic_memory_allocation
from data.datasets import AlignerDataset

np.random.seed(42)
tf.random.set_seed(42)
dynamic_memory_allocation()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', type=str, default='config/session_paths.yaml')
    parser.add_argument('--best', dest='best', action='store_true',
                        help='Use best head instead of weighted average of heads.')
    parser.add_argument('--autoregressive_weights', type=str, default=None,
                        help='Explicit path to autoregressive model weights.')
    parser.add_argument('--skip_durations', dest='skip_durations', action='store_true')
    args = parser.parse_args()
    weighted = not args.best
    tag_description = ''.join([
        f'{"_weighted" * weighted}{"_best" * (not weighted)}',
    ])
    writer_tag = f'DurationExtraction{tag_description}'
    print(writer_tag)
    config_manager = Config(config_path=args.config, aligner=True)
    config = config_manager.config
    config_manager.print_config()
    
    if not args.skip_durations:
        model = config_manager.load_model(args.autoregressive_weights)
        if model.r != 1:
            print(f"ERROR: model's reduction factor is greater than 1, check config. (r={model.r}")
        
        data_prep = AlignerPreprocessor.from_config(config=config_manager,
                                                    tokenizer=model.text_pipeline.tokenizer)
        data_handler = AlignerDataset.from_config(config_manager,
                                                  preprocessor=data_prep,
                                                  kind='phonemized')
        target_dir = config_manager.duration_dir
        config_manager.dump_config()
        dataset = data_handler.get_dataset(bucket_batch_sizes=config['bucket_batch_sizes'],
                                           bucket_boundaries=config['bucket_boundaries'],
                                           shuffle=False,
                                           drop_remainder=False)
        
        last_layer_key = 'Decoder_LastBlock_CrossAttention'
        print(f'Extracting attention from layer {last_layer_key}')
        
        summary_manager = SummaryManager(model=model, log_dir=config_manager.log_dir / 'Duration Extraction',
                                         config=config,
                                         default_writer='Duration Extraction')
        all_durations = np.array([])
        new_alignments = []
        iterator = tqdm(enumerate(dataset.all_batches()))
        step = 0
        for c, (mel_batch, text_batch, stop_batch, file_name_batch) in iterator:
            iterator.set_description(f'Processing dataset')
            outputs = model.val_step(inp=text_batch,
                                     tar=mel_batch,
                                     stop_prob=stop_batch)
            attention_values = outputs['decoder_attention'][last_layer_key].numpy()
            text = text_batch.numpy()
            
            mel = mel_batch.numpy()
            
            durations, final_align, jumpiness, peakiness, diag_measure = get_durations_from_alignment(
                batch_alignments=attention_values,
                mels=mel,
                phonemes=text,
                weighted=weighted)
            batch_avg_jumpiness = tf.reduce_mean(jumpiness, axis=0)
            batch_avg_peakiness = tf.reduce_mean(peakiness, axis=0)
            batch_avg_diag_measure = tf.reduce_mean(diag_measure, axis=0)
            for i in range(tf.shape(jumpiness)[1]):
                summary_manager.display_scalar(tag=f'DurationAttentionJumpiness/head{i}',
                                               scalar_value=tf.reduce_mean(batch_avg_jumpiness[i]), step=c)
                summary_manager.display_scalar(tag=f'DurationAttentionPeakiness/head{i}',
                                               scalar_value=tf.reduce_mean(batch_avg_peakiness[i]), step=c)
                summary_manager.display_scalar(tag=f'DurationAttentionDiagonality/head{i}',
                                               scalar_value=tf.reduce_mean(batch_avg_diag_measure[i]), step=c)
            
            for i, name in enumerate(file_name_batch):
                all_durations = np.append(all_durations, durations[i])  # for plotting only
                summary_manager.add_image(tag='ExtractedAlignments',
                                          image=tf.expand_dims(tf.expand_dims(final_align[i], 0), -1),
                                          step=step)
                
                step += 1
                np.save(str(target_dir / f"{name.numpy().decode('utf-8')}.npy"), durations[i])
        
        all_durations[all_durations >= 20] = 20  # for plotting only
        buckets = len(set(all_durations))  # for plotting only
        summary_manager.add_histogram(values=all_durations, tag='ExtractedDurations', buckets=buckets)
    
    print('Done.')
