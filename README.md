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

Seq2seq with Loung attention. Features:

- [Output embedding with wegiht tying](http://www.aclweb.org/anthology/E17-2025)

![Weight tying result](img/wt.png?raw=true)

## Credits

- Pytorch tutorial, for the base of the model. [Link](https://pytorch.org/tutorials/beginner/chatbot_tutorial.html)
- [OpenSubtitle](http://www.opensubtitles.org/) and thier [collection of datasets](http://opus.nlpl.eu/OpenSubtitles.php) of movies subtitles in every language.