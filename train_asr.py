import tensorflow as tf
import numpy as np
from tqdm import trange

from utils.config_manager import Config
from data.datasets import ASRDataset, ASRPreprocessor
from utils.decorators import ignore_exception, time_it
from utils.scheduling import piecewise_linear_schedule
from utils.logging_utils import SummaryManager
from model.transformer_utils import create_mel_padding_mask
from utils.scripts_utils import dynamic_memory_allocation, basic_train_parser
from data.metadata_readers import post_processed_reader

np.random.seed(42)
tf.random.set_seed(42)
dynamic_memory_allocation()


@ignore_exception
@time_it
def validate(model,
             val_dataset,
             summary_manager):
    val_loss = {'loss': 0.}
    norm = 0.
    for spk, mels, phonemes, mel_len, phon_len, fname in val_dataset.all_batches():
        model_out = model.val_step(spk=spk,
                                    mel_inp=mels,
                                    phon_tar=phonemes,
                                    mel_inp_len=mel_len,
                                    phon_tar_len=phon_len)
        norm += 1
        val_loss['loss'] += model_out['loss']

    val_loss['loss'] /= norm
    summary_manager.display_loss(model_out, tag='Validation', plot_all=True)
    summary_manager.display_attention_heads(model_out, tag='ValidationAttentionHeads')

    # predict phonemes
    for j, mel in enumerate(mels):
        model_out = model.predict(mel[np.newaxis, :mel_len[j], ...])
        pred_phon = model_out['encoder_output'][0]
        pred_phon, _ = tf.nn.ctc_beam_search_decoder(pred_phon[:,np.newaxis,:], mel_len[j][np.newaxis,...], beam_width=20, top_paths=1)
        iphon = model.text_pipeline.tokenizer.decode(pred_phon[0].values).replace('/', '')
        iphon_tar = model.text_pipeline.tokenizer.decode(phonemes[j][:phon_len[j]]).replace('/', '')
        summary_manager.display_audio(tag=f'Validation /{j} /{iphon}', step=model.step, 
                                    mel=mel[:mel_len[j], :], description=iphon_tar)
    return val_loss['loss']


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

train_data_handler = ASRDataset.from_config(config,
                                            tokenizer=model.text_pipeline.tokenizer,
                                            kind='train')
valid_data_handler = ASRDataset.from_config(config,
                                            tokenizer=model.text_pipeline.tokenizer,
                                            kind='valid')
train_dataset = train_data_handler.get_dataset(bucket_batch_sizes=config_dict['bucket_batch_sizes'],
                                               bucket_boundaries=config_dict['bucket_boundaries'],
                                               shuffle=True)
valid_dataset = valid_data_handler.get_dataset(bucket_batch_sizes=config_dict['val_bucket_batch_size'],
                                               bucket_boundaries=config_dict['bucket_boundaries'],
                                               shuffle=False,
                                               drop_remainder=True)

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

# main event
print('\nTRAINING')
test_spk, test_mel, test_phonemes, test_mel_len, test_phon_len, test_fname = valid_dataset.next_batch()

losses = []
t = trange(model.step, config_dict['max_steps'], leave=True)
for _ in t:
    t.set_description(f'step {model.step}')
    spk, mel, phonemes, mel_len, phon_len, fname = train_dataset.next_batch()
    learning_rate = piecewise_linear_schedule(model.step, config_dict['learning_rate_schedule'])
    model.set_constants(learning_rate=learning_rate)
    
    output = model.train_step(spk=spk,
                              mel_inp=mel,
                              phon_tar=phonemes,
                              mel_inp_len=mel_len,
                              phon_tar_len=phon_len)
    losses.append(float(output['loss']))

    t.display(f'step loss: {losses[-1]}', pos=1)
    for pos, n_steps in enumerate(config_dict['n_steps_avg_losses']):
        if len(losses) > n_steps:
            t.display(f'{n_steps}-steps average loss: {sum(losses[-n_steps:]) / n_steps}', pos=pos + 2)
    
    summary_manager.display_loss(output, tag='Train')
    summary_manager.display_scalar(scalar_value=t.avg_time, tag='Meta/iter_time')
    summary_manager.display_scalar(scalar_value=tf.shape(fname)[0], tag='Meta/batch_size')
    summary_manager.display_scalar(tag='Meta/learning_rate', scalar_value=model.optimizer.lr)
    if model.step % config_dict['train_images_plotting_frequency'] == 0:
        summary_manager.display_attention_heads(output, tag='TrainAttentionHeads')
        pred_phon = output['encoder_output'][0]
        pred_phon, _ = tf.nn.ctc_beam_search_decoder(pred_phon[:,np.newaxis,:], mel_len[0][np.newaxis,...], beam_width=20, top_paths=1)
        iphon = model.text_pipeline.tokenizer.decode(pred_phon[0].values).replace('/', '')
        iphon_tar = model.text_pipeline.tokenizer.decode(phonemes[0]).replace('/', '')
        summary_manager.display_audio(tag=f'Train /{0} /{iphon}', step=model.step, 
                                      mel=mel[0][:mel_len[0], :], description=iphon_tar)

    if model.step % 1000 == 0:
        save_path = manager_training.save()
    if model.step % config_dict['weights_save_frequency'] == 0:
        save_path = manager.save()
        t.display(f'checkpoint at step {model.step}: {save_path}', pos=len(config_dict['n_steps_avg_losses']) + 2)
    
    if model.step % config_dict['validation_frequency'] == 0:
        t.display(f'Validating', pos=len(config_dict['n_steps_avg_losses']) + 3)
        val_loss, time_taken = validate(model=model,
                                        val_dataset=valid_dataset,
                                        summary_manager=summary_manager)
        t.display(f'validation loss at step {model.step}: {val_loss} (took {time_taken}s)',
                  pos=len(config_dict['n_steps_avg_losses']) + 3)
    
    # if model.step % config_dict['prediction_frequency'] == 0 and (model.step >= config_dict['prediction_start_step']):
    #     for j in range(len(test_mel)):
    #         if j < config['n_predictions']:
    #             model_out = model.predict(test_mel[j])
    #             indices = tf.math.argmax(model_out['encoder_output'][j, ...], axis=-1)
    #             iphon = model.text_pipeline.tokenizer.decode(tf.gather_nd(indices, tf.where(indices > 0)))
    #             iphon_tar = " ".join(model.text_pipeline.tokenizer.decode(test_phonemes[j]))
    #             summary_manager.display_audio(tag=f'Test /{iphon}',
    #                                         mel=mel[j][:test_mel_len[j], :], description=iphon_tar)

print('Done.')
