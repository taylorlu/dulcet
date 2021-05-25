# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
from utils.scripts_utils import dynamic_memory_allocation, basic_train_parser
from utils.config_manager import Config
from ctc_segmentation import ctc_segmentation, determine_utterance_segments
from ctc_segmentation import CtcSegmentationParameters
from ctc_segmentation import prepare_token_list
import tgt, re
from pypinyin import pinyin, Style
from data.text import TextToTokens
from data.text.tokenizer import Phonemizer, Tokenizer
from data.audio import Audio


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


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

audio = Audio(config=config.config)

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

input_wav = r'D:\winbeta\Beta.VideoProcess\Src\test\test.wav'
input_text = '这种写作方式是媒体常用的写作方式。这种模式将新闻中最重要的消息写在第一段，或是以新闻提要的方式呈现新闻的最前端，有助于受众快速了解新闻重点。由于该模式迎合了受众的接受心理，所以成为媒体应用最为普遍的形式。这种模式写作的基本格式（除了标题）是：先在导语中写出新闻事件中最有新闻价值的部分（新闻价值通俗来讲就是新闻中那些最突出，最新奇，最能吸引受众的部分；其次，在报道主体中按照事件各要素的重要程度，依次递减写下来，最后面的是最不重要的；同时需要注意的是，一个段落只写一个事件要素，不能一段到底。因为这种格式不是符合事件发展的基本时间顺序，所以在写作时要尽量从受众的角度出发来构思，按受众对事件重要程度的认识来安排事件要素，因而需要长期的实践经验和宏观的对于受众的认识。'

english_lexicon = './data/text/lexicon/librispeech-lexicon.txt'
pinyin_lexicon_path = './data/text/lexicon/pinyin-lexicon-r.txt'
pinyin_lexicon = read_lexicon(pinyin_lexicon_path)

chartuples = []
lastend = 0
phones = []
pinyins = [
    p[0]
    for p in pinyin(
        input_text, style=Style.TONE3, strict=False, neutral_tone_with_five=True
    )
]
print(pinyins)
print(pinyin_lexicon)
for p in pinyins:
    if p in pinyin_lexicon:
        phones += pinyin_lexicon[p]
        chartuples.append((lastend, lastend+len(pinyin_lexicon[p])-1))
        lastend += len(pinyin_lexicon[p])
    else:
        phones.append("sp")
        chartuples.append((lastend, lastend))
        lastend += 1

print(phones)

tokenizer = Tokenizer(add_start_end=False)
phonemes = np.array([tokenizer(' '.join(phones))])

y, sr = audio.load_wav(input_wav)
mel = audio.mel_spectrogram(y)

model_out = model.predict(mel[np.newaxis, ...])
pred_phon = model_out['encoder_output'][0]
pred_phon = tf.nn.log_softmax(pred_phon)
iphon_tar = model.text_pipeline.tokenizer.decode(phonemes[0])
iphon_tar = iphon_tar.split()

char_list = [''] +list(model.text_pipeline.tokenizer.idx_to_token.values())
config = CtcSegmentationParameters(char_list=char_list)
config.index_duration = 0.0115545

text = phonemes
ground_truth_mat, utt_begin_indices = prepare_token_list(config, text)
timings, char_probs, state_list = ctc_segmentation(config, pred_phon.numpy(), ground_truth_mat)
utt_begin_indices = list(range(2, len(timings)))
segments = determine_utterance_segments(
    config, utt_begin_indices, char_probs, timings, text[0]
)
print(text.shape, len(segments))

tg = tgt.core.TextGrid('haa')
tier = tgt.core.IntervalTier(name='phonemes')

if(segments[0][-1]<-0.001):
    segments[0] = (0, segments[0][1], segments[0][2])
else:
    itv = tgt.core.Interval(0, segments[0][0], text='sp')
    tier.add_interval(itv)

if(segments[-1][-1]<-0.001):
    segments[-1] = (segments[-1][0], segments[-1][1]+0.15, segments[-1][2])
    if(segments[-1][1]>mel.shape[1]*config.index_duration):
        pass
    else:
        itv = tgt.core.Interval(segments[-1][1], mel.shape[1]*config.index_duration, text='sp')
        tier.add_interval(itv)

for i, chartuple in enumerate(chartuples):
    itv = tgt.core.Interval(segments[chartuple[0]][0], segments[chartuple[1]][1], text=input_text[i])
    tier.add_interval(itv)

tg.add_tier(tier)
tgt.io.write_to_file(tg, "test.textgrid", format='long')
