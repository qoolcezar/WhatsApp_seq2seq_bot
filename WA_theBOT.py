import time
from WA_bot_libs import WA_srcaper
import configparser
import json
from emoji import UNICODE_EMOJI
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import re

load_save = '158xxxx' # from /saves/ folder

config_parser = configparser.RawConfigParser(comment_prefixes='/', allow_no_value=True, delimiters='=')

config_parser.read(f'saves\\{load_save}\\settings.ini', encoding='utf-8')
try:
    stop_bot_for = config_parser.get('PERSISTENT', 'stop_bot_for')
    stop_bot_for = [item.strip() for item in stop_bot_for.split(',') if item.strip() != '']
except:
    stop_bot_for = []
UNK_REPLACEMENT = config_parser.get('LEARN_BOT', 'UNK_REPLACEMENT')
VOCAB_SIZE = int(config_parser.get('BOT', 'VOCAB_SIZE'))
MAX_INPUT_LENGTH = int(config_parser.get('BOT', 'MAX_INPUT_LENGTH'))
WORD2VEC_DIMS = int(config_parser.get('LEARN_BOT', 'WORD2VEC_DIMS'))
LSTM_UNITS = int(config_parser.get('BOT', 'LSTM_UNITS'))

model_weights = f'saves\\{load_save}\\last_saved_model.h5'
VOCAB_json_tokenizer = f'saves\\{load_save}\\json_tokenizer.json'
with open(VOCAB_json_tokenizer) as f:
    json_tokenizer = json.load(f)

BOS = '<bos>'
EOS = '<eos>'
UNK = '<unk>'

# search your emoji
def is_emoji(s):
    return s in UNICODE_EMOJI

def format_text(text_list):
    '''
    Clean and format list of texts for preprocessing, returns list of texts
    '''

    # add space near your emoji
    def add_space_emojy(text):
        return ''.join(' ' + char if is_emoji(char) else char for char in text).strip()

    def clean_text(text):
        '''Clean text by removing unnecessary characters and altering the format of words.'''

        text = add_space_emojy(text)
        text = text.lower()
        #text = re.sub(r"\.", " . ", text)
        text = re.sub(r"[\.]"*3, r" ... ", text)
        text = re.sub(r"\,", " , ", text)
        text = re.sub(r"\!", " ! ", text)
        text = re.sub(r"\?", " ? ", text)
        text = re.sub(r"\:", " : ", text)
        text = re.sub(r"\;", " ; ", text)
        text = re.sub(r"\"", " ' ", text)
        text = re.sub(r"'", " ' ", text)
        text = re.sub(r"[()@;:{}`~|*#’]", "", text)
        text = re.sub(r"\ș", "s", text)
        text = re.sub(r"\â", "a", text)
        text = re.sub(r"\ă", "a", text)
        text = re.sub(r"\ă", "a", text)
        text = re.sub(r"\ț", "t", text)
        text = re.sub(r"\î", "i", text)
        text = re.sub(r"sh", "s", text)
        text = re.sub(r"tz", "t", text)
        text = re.sub(r"-", " ", text)
        text = re.sub(r"\b\.", " . ", text)

        return text

    formated_text = [clean_text(text) for text in text_list]
    formated_text = [BOS + ' ' + text + ' ' + EOS for text in formated_text]
    
    return formated_text

class text_to_seq():
    '''
    Preprocessing by tokenizer and pad_sequences

    json_tokenizer - json file with tokenizer data generated during learning by vocab_creator_text2seq()

    VOCAB_SIZE - desired amount of word in the vocabulary

    max_seq_lenght = None (default), if value sequence will be shortened to that value

    red_seq_from_common=True (default), for True shortening of sequence will be done by starting removing most common words. If False, most uncommon words.

    seq_text(text_input) returns sequenced array
    '''

    def __init__(self, json_tokenizer, VOCAB_SIZE, max_seq_lenght = None, red_seq_from_common=True):

        self.tokenizer = tokenizer_from_json(json_tokenizer)
        self.VOCAB_SIZE = VOCAB_SIZE
        self.max_seq_lenght = max_seq_lenght
        self.red_seq_from_common = red_seq_from_common

        dictionary = self.tokenizer.word_index
        self.word2idx = {}
        self.idx2word = {}
        for k, v in dictionary.items():
            if v < VOCAB_SIZE:
                self.word2idx[k] = v
                self.idx2word[v] = k
            if v >= VOCAB_SIZE-1:
                continue

        self.idx_2_keep = [self.word2idx[BOS.strip()], self.word2idx[EOS.strip()]]

    def seq_text(self, text_input):
        '''
        text_input - list of formated text

        Returns sequenced array
        '''
        text_input_sequence = self.tokenizer.texts_to_sequences(text_input)
        if self.max_seq_lenght is not None:
            text_input_sequence = self.reduce_sequences(text_input_sequence)
        text_input_sequence = pad_sequences(text_input_sequence, maxlen=self.max_seq_lenght, dtype='int32', padding='post', truncating='post')

        return text_input_sequence

    def reduce_sequences(self, sequences):
        for each_seq_id in range(len(sequences)):
            if self.red_seq_from_common:
                idx_2_remove = 1
            else:
                idx_2_remove = max(sequences[each_seq_id])

            while len(sequences[each_seq_id]) > self.max_seq_lenght and idx_2_remove <= max(sequences[each_seq_id]):
                if sequences[each_seq_id].count(idx_2_remove) > 0 and idx_2_remove not in self.idx_2_keep:
                    sequences[each_seq_id].remove(idx_2_remove)
                else:
                    if self.red_seq_from_common:
                        idx_2_remove += 1
                    else:
                        idx_2_remove -= 1

            if len(sequences[each_seq_id]) > self.max_seq_lenght:
                print(f'reduce_sequences-> still to long {sequences[each_seq_id]}')
        return sequences

seqeuncer = text_to_seq(json_tokenizer=json_tokenizer, VOCAB_SIZE=VOCAB_SIZE, max_seq_lenght=MAX_INPUT_LENGTH, red_seq_from_common=True)

def create_model(LSTM_UNITS, word_embedding_matrix=None):
    '''
    Weights for the Shared Embedding layer = None (default), weights for the Shared Embedding layer generated by embedding_matrix_creater function. Not needed when loading models
    '''
    from tensorflow.keras.layers import Input, Embedding, LSTM, concatenate, Dense
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.models import Model

    input_context = Input(shape=(MAX_INPUT_LENGTH,), dtype="int32", name="input_context")
    input_answer = Input(shape=(MAX_INPUT_LENGTH,), dtype="int32", name="input_answer")

    if word_embedding_matrix is not None:
        Shared_Embedding = Embedding(input_dim=VOCAB_SIZE, output_dim=WORD2VEC_DIMS, input_length=MAX_INPUT_LENGTH, weights=[word_embedding_matrix])
    else:
        Shared_Embedding = Embedding(input_dim=VOCAB_SIZE, output_dim=WORD2VEC_DIMS, input_length=MAX_INPUT_LENGTH)

    shared_embedding_context = Shared_Embedding(input_context)
    shared_embedding_answer = Shared_Embedding(input_answer)

    Encoder_LSTM = LSTM(units=LSTM_UNITS, kernel_initializer= "lecun_uniform")
    Decoder_LSTM = LSTM(units=LSTM_UNITS, kernel_initializer= "lecun_uniform")
    embedding_context = Encoder_LSTM(shared_embedding_context)
    embedding_answer = Decoder_LSTM(shared_embedding_answer)

    merge_layer = concatenate([embedding_context, embedding_answer], axis=1)
    dence_layer = Dense(VOCAB_SIZE/2, activation="relu")(merge_layer)
    outputs = Dense(VOCAB_SIZE, activation="softmax")(dence_layer)

    model = Model(inputs=[input_context, input_answer], outputs=[outputs])
    model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.00005))

    return model

model = create_model(LSTM_UNITS)
model.summary()
model.load_weights(model_weights)

def bot_reply(inp):
    inp = [inp]
    ans_partial = np.zeros((1,MAX_INPUT_LENGTH))
    ans_partial[0, -1] = seqeuncer.word2idx[BOS.strip()]  #  the index of the symbol BOS (begin of sentence)
    for k in range(MAX_INPUT_LENGTH - 1):
        ye = model.predict([inp, ans_partial])
        mp = np.argmax(ye)
        if mp == 0:
            print('print_result')
        ans_partial[0, 0:-1] = ans_partial[0, 1:]
        ans_partial[0, -1] = mp
    text = ''
    for k in ans_partial[0]:
        k = k.astype(int)

        try:    # for words not in vocabulary
            text = text + seqeuncer.idx2word[k] + ' '
        except:
            continue

    text = text.replace(BOS.strip(), '')
    text = text.replace(EOS.strip(), '')
    text = " ".join(text.split())
    text = text.replace(UNK, UNK_REPLACEMENT)

    return text

def main():
    """
    Loading all the configuration and opening the website
    (Browser profile where whatsapp web is already scanned)
    """

    scrapper = WA_srcaper(f'saves\\{load_save}\\settings.ini')
    chatters = scrapper.settings['chatters']
    click_delay = scrapper.settings['click_delay']

    try:
        while True:
            for chatter in chatters:
                if scrapper.select_chatter(chatter):
                    time.sleep(click_delay)

                    scrapper.read_web_messages()  # returns self.messages_df
                    scrapper.scan_messages()

                    print(f'chatter= {scrapper.chatter}')
                    print(scrapper.message)
                    print(f'START/STOP flag= {scrapper.start_stop_flg}')

                    if (scrapper.start_stop_flg == True) and (scrapper.chatter in stop_bot_for):
                        stop_bot_for.remove(scrapper.chatter)
                        config_parser.set('PERSISTENT', 'stop_bot_for', ','.join(stop_bot_for))
                        config_parser.write(open(f'saves\\{load_save}\\settings.ini', 'w', encoding='utf-8'))
                    elif (scrapper.start_stop_flg == False) and (scrapper.chatter not in stop_bot_for):
                        stop_bot_for.append(scrapper.chatter)
                        config_parser.set('PERSISTENT', 'stop_bot_for', ','.join(stop_bot_for))
                        config_parser.write(open(f'saves\\{load_save}\\settings.ini', 'w', encoding='utf-8'))

                    if (len(scrapper.message) > 0) and (scrapper.chatter not in stop_bot_for):

                        text_input_formated = format_text([scrapper.message])
                        seqenced_text = seqeuncer.seq_text(text_input_formated)
                        text_reply = bot_reply(seqenced_text[0])
                        print(f'Reply to send: {text_reply}')
                        scrapper.send_messages(text_reply)
                        print()
                time.sleep(click_delay)
    except KeyboardInterrupt:
        scrapper.stop_drivers()

if __name__ == '__main__':
    main()
