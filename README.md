# Chatbot en Español

## Introuction

Conversational agent in spanish done with deep learning and a dataset of movies subtitles. If you want to walk directly to the transformer version you can do it [here](https://rubenchaves.com/chatbot).

## Model description

- Seq2seq. For a detailed explanation in spanish you can see this [blog post](https://medium.com/@ruben_onelove/como-hacer-un-chatbot-en-espa%C3%B1ol-y-que-te-trolee-en-el-intento-2a8105d66de8). Features:

  - [Loung attention](https://arxiv.org/abs/1508.04025)
  - [Output embedding with wegiht tying](http://www.aclweb.org/anthology/E17-2025)


- Transformer. Features:

  - Weight tying
  - Beam search
  - Quantization: [Pytorch Dynamic Quantization](https://pytorch.org/docs/stable/quantization.html#quantization-workflows). Model size reduced to 41% of the original and 2x inference speed up. Backends suported:
    - x86 CPUs with AVX2 support or higher (without AVX2 some operations have inefficient implementations)
    - ARM CPUs (typically found in mobile/embedded devices)

## Instructions

- For training:
  1. Download dataset from here [here (2Gb)](http://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/mono/OpenSubtitles.raw.es.gz ) and put it on /data
  2. Generate data with `python pre_processing.py`. Arguments:
     - `--lines`: number of lines from the orignial dataset to be processed. Default 500_00
     - `--max_len`: max length of the sentence. Default: 40
     - `--min_count`: min count of a word to be left of the vocabulary. Default: 10
  3. Run the training notebook for training and evaluating of the model

For a detailed explanation of the processing see the notebook.

- For evaluation:
  1. Download the parameters for the [seq2seq mode](https://drive.google.com/open?id=1YmAgP_K75znP599HsW4kGlA2e-oeguhX), the [full transformer model](https://drive.google.com/open?id=12u1dOvexsnhKdJAW53ri4PC6gOyvo3ZT) or the [quantized transformer model](https://drive.google.com/file/d/1ZE3_AUzJWNvwVXzkvWI_4DS8NgniFhFB/view?usp=sharing) and uncompress on /data
  2. Run the evaluation notebook.

## Credits

- Pytorch tutorial, for the base of the model. [Link](https://pytorch.org/tutorials/beginner/chatbot_tutorial.html)
- [OpenSubtitle](http://www.opensubtitles.org/) and thier [collection of datasets](http://opus.nlpl.eu/OpenSubtitles.php) of movies subtitles in every language.