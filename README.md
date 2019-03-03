# Chatbot en Español

## Credits

- https://pytorch.org/tutorials/beginner/chatbot_tutorial.html
- data: http://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/mono/OpenSubtitles.raw.es.gz (2Gb)

## TODO

- generator from data pre-processing to feed data to the training
- re train model with more data
- tokenization/lemmanization?
- use trained emmbedings? glove word2ve ulmfit bert | visualize embeding
- EVAL: acentos en los inputs?
- implementar one cycle y ver cuanto dura
- utilizar la propia estructura del dataset para sacr mas rendimiento en la evaluacion del modelo. \n y guion como comienzo de intercambio a la gpt-2 con los TL-DR. Comparar resultados(subjetivos) con y sin.
- add enviroment.yml