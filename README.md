# Chatbot en Español

## Instructions

- For training:
  1. Download dataset from here [here (2Gb)](http://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/mono/OpenSubtitles.raw.es.gz ) and put it on /data
  2. Generate data with `python pre_processing.py`. Arguments:
     - `--lines`: number of lines from the orignial dataset to be processed. Default 2_500_00
     - `--max_len`: max length of the sentence. Default: 20
     - `--min_count`: min count of a word to be left of the vocabulary. Default: 5
  3. Run the training notebook ofr training and evaluation of the model

For a detailed explanation of the processing see the notebook.

- For evaluation:
  1. Download [the parameters](https://drive.google.com/open?id=1YmAgP_K75znP599HsW4kGlA2e-oeguhX) and uncompress on /data
  2. Run the evaluation notebook.

## Model description

Seq2seq with Loung attention.

## Credits

- Pytorch tutorial, for the base of the model. [Link](https://pytorch.org/tutorials/beginner/chatbot_tutorial.html)
- [OpenSubtitle](http://www.opensubtitles.org/) and thier [collection of datasets](http://opus.nlpl.eu/OpenSubtitles.php) of movies subtitles in every language.

## TODO

- Data processing:
  - multi thread data generation
  - add trim by voc size
  - longer sequences ~30/50?
  - add pyotrch dataset
  - tokenization/lemmanization?
  - juntar pares con secuencias de longitud parecida para disminuir los PAD_TOKEN en los batches y por lo tanto disminuir calculos malgastados.
- Model
  - Add [Output embedding](http://www.aclweb.org/anthology/E17-2025)
  - Add [Negative sampling](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
  - Add dropout to output layer
  - Transformer
  - Leaky ReLU
  - [Layer Norm](https://arxiv.org/pdf/1607.06450.pdf)
- Train
  - add visualization of loss dict
  - generator from data pre-processing to feed data to the training
  - re train model with more data
  - use trained emmbedings? glove word2ve ulmfit bert | visualize embeding
  - implementar one cycle y ver cuanto dura
  - utilizar la propia estructura del dataset para sacar mas rendimiento en la evaluacion del modelo. "\n" y guion como comienzo de intercambio a la gpt-2 con los TL-DR. Comparar resultados(subjetivos) con y sin. add to README DESCRIPTION THAT WE ARE USING THIS
  - add enviroment.yml
  - add link to downlaod model
- Evaluation
  - acentos en los inputs?
  - evaluation script for console
  - port to app/slack
  - t-sne of the embedding
  - add script to run chatbot on terminal
  - reciclar el hidden en las conversaciones