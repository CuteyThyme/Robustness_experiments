# coding=utf-8
from torch import optim
from fastNLP.core.sampler import BucketSampler
from fastNLP.core.trainer import Trainer
from fastNLP.core.callback import GradientClipCallback, WarmupCallback, EvaluateCallback
from fastNLP.core.metrics import SpanFPreRecMetric
from fastNLP.core.utils import cache_results
from nermodel import NERModel


from utils.data_embed import load_data
from config import Config

import argparse


parser = argparse.ArgumentParser()

parser.add_argument(
    '--dataset', type=str, 
    default='conll2003', choices=['ace', 'conll2003', 'en-ontonotes'],
    help='Dataset to train the model.'
)
parser.add_argument(
    '--char_embedding', type=str,
    default=None, choices=['cnn'],
    help='types of char embedding'
)
parser.add_argument(
    '--word_embedding', type=str, 
    default=None, choices=['rand', 'glove', 'bert', 'elmo'],
    help = 'pretrained word embedding'
)
parser.add_argument(
    '--transformation', type=str,
    default=None, choices=[ 'ori_concat', 'ori_crossCategory', 'ori_entityTyposSwap', 'ori_oov', 'ori_toLonger', 'ori_addAdverb', 
                            'ori_addLrrSent', 'ori_caseTitle', 'ori_caseUpper', 'ori_contraction', 'ori_keyboard', 'ori_mlm', 
                            'ori_number', 'ori_ocr', 'ori_punctuationAddBracket', 'ori_reverseNeg', 'ori_spelling', 'ori_tense', 
                            'ori_twitterRandom', 'ori_typosRandom', 'ori_wordEmbedding', 'ori_wordNetAntonym', 'ori_wordNetSynonym',
                            'concat', 'crossCategory', 'entityTyposSwap', 'oov', 'toLonger', 'addAdverb', 
                            'addLrrSent', 'caseTitle', 'caseUpper', 'contraction', 'keyboard', 'mlm', 'number',
                            'ocr', 'punctuationAddBracket', 'reverseNeg', 'spelling', 'tense', 'twitterRandom', 
                            'typosRandom', 'wordEmbedding', 'wordNetAntonym', 'wordNetSynonym'],
    help='transformation type for test dataset'
)
parser.add_argument(
    '--encoder', type=str, 
    default='lstm', choices=['cnn', 'transformer'],
    help='Encoder to extract contextual features.'
)
parser.add_argument(
    '--decoder', type=str, 
    default='crf', choices=['mlp', 'crf'],
    help='Decoder to inference.'
)

args = parser.parse_args()
config = Config()
print('当前设置为:\n', config)


data_bundle, embed = load_data(args.dataset, args.word_embedding, args.char_embedding)
print(data_bundle)

model = NERModel(embed=embed, num_classes=len(data_bundle.get_vocab('target')), num_layers=config.num_layers, 
                  hidden_size=config.hidden_size, dropout=0.2, encoder=args.encoder, decoder=args.decoder, 
                  target_vocab=data_bundle.get_vocab('target'))


if config.optim_type == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)
else:
    optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.999))


callbacks = []
clip_callback = GradientClipCallback(clip_type='value', clip_value=5)
evaluate_callback = EvaluateCallback(data_bundle.get_dataset('test'))

if config.warmup_steps>0:
    warmup_callback = WarmupCallback(config.warmup_steps, schedule='linear')
    callbacks.append(warmup_callback)

if args.transformation is not None:
    trans_type = args.transformation
    trans_evaluate_callback = EvaluateCallback(data_bundle.get_dataset(trans_type))
    callbacks.extend([clip_callback, trans_evaluate_callback])
else:
    callbacks.extend([clip_callback, evaluate_callback])

trainer = Trainer(data_bundle.get_dataset('train'), model, optimizer, batch_size=config.batch_size, sampler=BucketSampler(),
                  num_workers=0, n_epochs=config.base_epoch, dev_data=data_bundle.get_dataset('dev'), 
                  metrics=SpanFPreRecMetric(tag_vocab=data_bundle.get_vocab('target'), encoding_type=config.encoding_type),
                  dev_batch_size=config.batch_size, callbacks=callbacks, device=config.device, test_use_tqdm=False,
                  use_tqdm=True, print_every=300, save_path=None)

trainer.train(load_best_model=False)
