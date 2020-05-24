# Introduction

This is a proof of concept implementation of a simple replying bot to selected WhatsApp contacts. For learning it uses exported WhatsApp chats and Cornell Movie Dialogs Corpus and it can be adapted to most languages.

The model used is a sequence to sequence model that uses LSTM for generating text (a reply) for given a specific input.

# Technical

| Title | Detail |
| --- | --- |
| Environment | Windows 10 x64 |
| Language | Python 3.7.5 |
| Libraries | Tensorflow 2.1.0, Selenium 3.141.0, Pandas, [translate](https://pypi.org/project/translate/), [emoji](https://pypi.org/project/emoji/), [fuzzywuzzy](https://pypi.org/project/fuzzywuzzy/) |

# Instructions

1. Set up the learning dataset by creating a folder named _learn\_files_ in which the following files should be placed:
  1. In a folder named _WhatsAppChats_ all desired exported WhatsApp chats
  2. _movie\_lines.txt_ from from [Cornell Movie Dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)
  3. a language compatible word vectors file from [https://fasttext.cc/docs/en/crawl-vectors.html](https://fasttext.cc/docs/en/crawl-vectors.html)
2. Execute _translate\_movie\_lines.py_: select desired language to translate to (None for not translating, i.e. English); for language select first Translator provider.
3. Execute _make\_movie\_dialog\_pairs.py_: select desired language, must be same as WhatsApp chats
4. Execute _WAChats\_processing.py_: provide in settings.ini [WA\_process]REPLY\_AS\_ID with your WhatsApp username
5. Execute _learn.py_: tweak settings.ini as desired. A model and useful files will be saved in _/saves/_ folder as a unix timestamp. For continue learning even with new chats and/or translations, update _save\_name_ with the unix timestamp
6. Execute _test\_model.py_ for model testing in terminal. Update _load\_save_ with desired model from _/saves/_ folder
7. Execute _WA\_theBOT.py_ with model defined in _load\_save_, bot will reply to every contact or group defined in settings.ini from _/saves/{load\_save}/_ folder [WA\_srcaper]CHATTERS

Must manually configure WhatsApp Web and define the profile path settings.ini [WA\_srcaper]BROWSER\_PATH

# Reference

[Automatic-Encoder-Decoder\_Seq2Seq\_Chatbot](https://github.com/samurainote/Automatic-Encoder-Decoder_Seq2Seq_Chatbot)

[WhatsApp-Scraping](https://github.com/JMGama/WhatsApp-Scraping)
