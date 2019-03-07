# Chatbot en Español

## Instructions

1. Download dataset from here [here (2Gb)](http://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/mono/OpenSubtitles.raw.es.gz ) and put it on /data
2. Generate data with `python pre_processing.py`. Arguments:
   - `--lines`: number of lines from the orignial dataset to be processed. Default 2_500_00
   - `--max_len`: max length of the sentence. Default: 20
   - `--min_count`: min count of a word to be left of the vocabulary. Default: 5
3. Run the training notebook ofr training and evaluation of the model

For a detail explanation of the processing see the notebook.

## Credits

- Pytorch tutorial, for the base of the model. [Link](https://pytorch.org/tutorials/beginner/chatbot_tutorial.html)
- [OpenSubtitle](http://www.opensubtitles.org/) and thier [collection of datasets](http://opus.nlpl.eu/OpenSubtitles.php) of movies subtitles in every language.

## TODO

- generator from data pre-processing to feed data to the training
- re train model with more data
- tokenization/lemmanization?
- use trained emmbedings? glove word2ve ulmfit bert | visualize embeding
- EVAL: acentos en los inputs?
- implementar one cycle y ver cuanto dura
- utilizar la propia estructura del dataset para sacar mas rendimiento en la evaluacion del modelo. "\n" y guion como comienzo de intercambio a la gpt-2 con los TL-DR. Comparar resultados(subjetivos) con y sin.
- add enviroment.yml
- add link to downlaod model
- add script to run chatbot on terminal