import pandas as pd
import os
import numpy as np
import re
import gc
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from emoji import UNICODE_EMOJI
import json
from datetime import datetime
from fuzzywuzzy import fuzz
import configparser

save_name = '158xxxxx'    # from /saves/ folder for continue learning

config_parser = configparser.RawConfigParser()

settings_file = f'saves\\{save_name}\\settings.ini'

if os.path.isfile(settings_file):
    print(f'Continue learning from saves\\{save_name}')
    config_parser.read(settings_file, encoding='utf-8')
    WACHATS_PAIRS_PATH = config_parser.get('LEARN_BOT', 'WACHATS_PAIRS_PATH')
    MOVIES_PAIRS_PATH = config_parser.get('LEARN_BOT', 'MOVIES_PAIRS_PATH')
    VOCAB_SIZE = int(config_parser.get('BOT', 'VOCAB_SIZE'))
    MAX_INPUT_LENGTH = int(config_parser.get('BOT', 'MAX_INPUT_LENGTH'))
    vector_file = config_parser.get('LEARN_BOT', 'vector_file')
    WORD2VEC_DIMS = int(config_parser.get('LEARN_BOT', 'WORD2VEC_DIMS'))
    UNK_REPLACEMENT = config_parser.get('LEARN_BOT', 'UNK_REPLACEMENT')
    LSTM_UNITS = int(config_parser.get('BOT', 'LSTM_UNITS'))
    TEST_frac = float(config_parser.get('LEARN_BOT', 'TEST_frac'))
    NUM_EPOCHS = int(config_parser.get('LEARN_BOT', 'NUM_EPOCHS'))
    BATCH_SIZE = int(config_parser.get('LEARN_BOT', 'BATCH_SIZE'))
    NUM_SUBSETS = int(config_parser.get('LEARN_BOT', 'NUM_SUBSETS'))
    SAVE_RATIO = int(config_parser.get('LEARN_BOT', 'SAVE_RATIO'))
    SAVE_EACH_EPOCH = int(config_parser.get('LEARN_BOT', 'SAVE_EACH_EPOCH'))
    
    VOCAB_json_tokenizer = f'saves\\{save_name}\\json_tokenizer.json'
    with open(VOCAB_json_tokenizer) as f:
        json_tokenizer = json.load(f)
    logs_df = pd.read_csv(f'saves\\{save_name}\\learn.log')
    model_weights = f'saves\\{save_name}\\last_saved_model.h5'
    epoch_start = logs_df.iloc[-1]["Epoch"]
else:
    config_parser.read('settings.ini', encoding='utf-8')
    WACHATS_PAIRS_PATH = config_parser.get('LEARN_BOT', 'WACHATS_PAIRS_PATH')
    MOVIES_PAIRS_PATH = config_parser.get('LEARN_BOT', 'MOVIES_PAIRS_PATH')
    VOCAB_SIZE = int(config_parser.get('BOT', 'VOCAB_SIZE'))
    MAX_INPUT_LENGTH = int(config_parser.get('BOT', 'MAX_INPUT_LENGTH'))
    vector_file = config_parser.get('LEARN_BOT', 'vector_file')
    WORD2VEC_DIMS = int(config_parser.get('LEARN_BOT', 'WORD2VEC_DIMS'))
    UNK_REPLACEMENT = config_parser.get('LEARN_BOT', 'UNK_REPLACEMENT')
    LSTM_UNITS = int(config_parser.get('BOT', 'LSTM_UNITS'))
    TEST_frac = float(config_parser.get('LEARN_BOT', 'TEST_frac'))
    NUM_EPOCHS = int(config_parser.get('LEARN_BOT', 'NUM_EPOCHS'))
    BATCH_SIZE = int(config_parser.get('LEARN_BOT', 'BATCH_SIZE'))
    NUM_SUBSETS = int(config_parser.get('LEARN_BOT', 'NUM_SUBSETS'))
    SAVE_RATIO = int(config_parser.get('LEARN_BOT', 'SAVE_RATIO'))
    SAVE_EACH_EPOCH = int(config_parser.get('LEARN_BOT', 'SAVE_EACH_EPOCH'))

    save_name = f'{int(datetime.now().timestamp())}'
    print(f'New learning process in saves\\{save_name}')
    json_tokenizer = None
    VOCAB_json_tokenizer = f'saves\\{save_name}\\json_tokenizer.json'
    os.makedirs(f'saves\\{save_name}', exist_ok=True)
    logs_df = pd.DataFrame(columns=['Epoch', 'Ratio'])
    model_weights = f'saves\\{save_name}\\last_saved_model.h5'
    os.system(f'copy settings.ini saves\\{save_name}\\settings.ini')

BOS = '<bos>'
EOS = '<eos>'
UNK = '<unk>'

if os.path.isfile(WACHATS_PAIRS_PATH):
    WAChats_convs_df = pd.read_csv(WACHATS_PAIRS_PATH, encoding='utf-8', engine = 'python')
else:
    print(f'Not found {WACHATS_PAIRS_PATH} ! Must process Whatsapp exported chats by executing WAChats_processing.py')

if os.path.isfile(MOVIES_PAIRS_PATH):
    convs_df = pd.read_csv(MOVIES_PAIRS_PATH)
    try:
        convs_df = pd.concat([WAChats_convs_df, convs_df])
    except:
        pass
else:
    print(f'Not found {MOVIES_PAIRS_PATH} ! Must process movie_lines.txt from Cornell Movie Dialogs Corpus by executing translate_movie_lines.py and make_movie_dialog_pairs.py')

convs_df = convs_df.sample(frac=1, random_state=1).reset_index(drop=True)   # set in sample() random_state=int(X) for load weights consistency

# code modified from https://github.com/samurainote/Automatic-Encoder-Decoder_Seq2Seq_Chatbot/blob/master/chatbot_keras.py

# search your emoji
def is_emoji(s):
    return s in UNICODE_EMOJI

def format_text(text_list):
    '''
    Clean and format list of texts for preprocessing, returns list of texts
    '''
    # search your emoji
    def is_emoji(s):
        return s in UNICODE_EMOJI

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

encoder_inputs = format_text(convs_df['Input'].tolist())
decoder_inputs = format_text(convs_df['Reply'].tolist())

def vocab_creator_text2seq(encoder_text, decoder_text, VOCAB_SIZE, max_seq_lenght = None, red_seq_from_common = True, load_json_tokenizer = None):
    '''
    Preprocessing by tokenizer and pad_sequences

    encoder_text and decoder_text - lists of formated text

    VOCAB_SIZE - desired amount of word in the vocabulary

    max_seq_lenght = None (default), if value sequence will be shortened to that value

    red_seq_from_common = True (default), for True shortening of sequence will be done by starting removing most common words. If False, most uncommon words.

    load_json_tokenizer = None (default), if None tokenizer will be generated from encoder_text + decoder_text. Load a tokenizer when train with new dialogues.

    Returns word2idx and idx2word dictionaries, encoder and decoder sequenced arrays, a Tokenizer json for further saving and loading
    '''

    def reduce_sequences(sequences, idx_2_keep=[], from_common=True, max_seq_lenght=max_seq_lenght):
        for each_seq_id in range(len(sequences)):
            if from_common:
                idx_2_remove = 1
            else:
                idx_2_remove = max(sequences[each_seq_id])

            while len(sequences[each_seq_id]) > max_seq_lenght and idx_2_remove <= max(sequences[each_seq_id]):
                if sequences[each_seq_id].count(idx_2_remove) > 0 and idx_2_remove not in idx_2_keep:
                    sequences[each_seq_id].remove(idx_2_remove)
                else:
                    if from_common:
                        idx_2_remove += 1
                    else:
                        idx_2_remove -= 1

            if len(sequences[each_seq_id]) > max_seq_lenght:
                print(f'reduce_sequences-> still to long {sequences[each_seq_id]}')
        return sequences

    if load_json_tokenizer is None:
        tokenizer = Tokenizer(num_words=VOCAB_SIZE, filters='', lower=True, oov_token=UNK)
        tokenizer.fit_on_texts(encoder_text + decoder_text)
    else:
        tokenizer = tokenizer_from_json(load_json_tokenizer)
    
    tokenizer_json = tokenizer.to_json()

    encoder_sequences = tokenizer.texts_to_sequences(encoder_text)
    decoder_sequences = tokenizer.texts_to_sequences(decoder_text)
    
    dictionary = tokenizer.word_index

    word2idx = {}
    idx2word = {}
    for k, v in dictionary.items():
        if v < VOCAB_SIZE:
            word2idx[k] = v
            idx2word[v] = k
        if v >= VOCAB_SIZE-1:
            continue
          
    idx_2_keep = [word2idx[BOS.strip()], word2idx[EOS.strip()]]
    if max_seq_lenght is not None:
        encoder_sequences = reduce_sequences(encoder_sequences, idx_2_keep=idx_2_keep, from_common=red_seq_from_common)
        decoder_sequences = reduce_sequences(decoder_sequences, idx_2_keep=idx_2_keep, from_common=red_seq_from_common)

    encoder_sequences = pad_sequences(encoder_sequences, maxlen=max_seq_lenght, dtype='int32', padding='post', truncating='post')
    decoder_sequences = pad_sequences(decoder_sequences, maxlen=max_seq_lenght, dtype='int32', padding='post', truncating='post')

    return word2idx, idx2word, encoder_sequences, decoder_sequences, tokenizer_json

word2idx, idx2word, encoder_input_data, decoder_input_data, json_tokenizer = vocab_creator_text2seq(encoder_inputs, decoder_inputs, VOCAB_SIZE=VOCAB_SIZE, max_seq_lenght=MAX_INPUT_LENGTH, red_seq_from_common=True, load_json_tokenizer=json_tokenizer )

# Save and load a tokenizer when train with new dialogues.
with open(VOCAB_json_tokenizer, 'w', encoding='utf-8') as f:
    f.write(json.dumps(json_tokenizer, ensure_ascii=False))

del encoder_inputs
del decoder_inputs

gc.collect()

def embedding_matrix_creater(vector_file, embedding_dimention, word2idx):
    '''
    Generates weights for the Shared Embedding layer

    vector_file - pretrained word vector file

    embedding_dimention - vector dimension of words in vector_file

    word2idx - word_to_id vocabulary generated by vocab_creater_text2seq

    TODO if embedding_vector not found for word, then grammatically process that word and search again for embedding_vector
    '''

    def load_vectors(fname):
        v_file = open(fname, encoding="utf-8", newline='\n', errors='ignore')
        n, d = map(int, v_file.readline().split())
        print(f'n={n} d={d}')
        data_vectors = {}
        for v_line in v_file:
            tokens = v_line.rstrip().split(' ')
            data_vectors[tokens[0]] = map(float, tokens[1:])
        return data_vectors

    unks = open("learn_files\\unks.txt", "a", encoding="utf-8")
    def add_unks(word):
        unks_file = open("learn_files\\unks.txt", "r", encoding="utf-8")
        word = word + ' '
        for line in unks_file:
            if word in line:
                unks_file.close()
                return False
        unks_file.close()
        return True

    word2vec_index = load_vectors(vector_file)

    vocab_size = len(word2idx) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dimention))

    for word, i in word2idx.items():
        embedding_vector = word2vec_index.get(word)

        if embedding_vector is not None:
            if isinstance(embedding_vector, map):
                embedding_vector_unpacked = list(embedding_vector)
                embedding_matrix[i] = np.asarray(embedding_vector_unpacked, dtype='float32')
                word2vec_index[word] = embedding_vector_unpacked
            else:
                embedding_matrix[i] = np.asarray(embedding_vector, dtype='float32')
            if is_emoji(word):
                print(f'word= {word} is emoji in word2vec_index, adding to unks')
                unks.write(word + ' # ' + word + '\n')
            continue
        
        if embedding_vector is None:
            print(f'embedding_matrix_creater -> no embeding for word= {word}', end=' ')

            # unknown emojis will be replaced with a known UNK_REPLACEMENT
            if is_emoji(word):
                print('emoji')

                embedding_vector = word2vec_index.get(UNK_REPLACEMENT)

                if embedding_vector is not None:
                    if isinstance(embedding_vector, map):
                        embedding_vector_unpacked = list(embedding_vector)
                        embedding_matrix[i] = np.asarray(embedding_vector_unpacked, dtype='float32')
                        word2vec_index[UNK_REPLACEMENT] = embedding_vector_unpacked
                    else:
                        embedding_matrix[i] = np.asarray(embedding_vector, dtype='float32')
                    print(f'embedding_matrix_creater -> word emoji= {word} replaced with UNK_REPLACEMENT= {UNK_REPLACEMENT}')
                    if add_unks(word):
                        unks.write(word + ' # ' + UNK_REPLACEMENT + '\n')
                    continue
                else:
                    if add_unks(word):
                        unks.write(word + ' ' + '\n')
                    print(f'embedding_matrix_creater -> UNK_REPLACEMENT= {UNK_REPLACEMENT} not in vector_file')

            else:
                if add_unks(word):
                    unks.write(word + ' ' + '\n')
                print()

    unks.close()
    del word2vec_index
    gc.collect()
    return embedding_matrix

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

if os.path.isfile(model_weights):
    model = create_model(LSTM_UNITS)
    model.summary()
    model.load_weights(model_weights)
else:
    if not os.path.isfile(vector_file):
        print(f'Not found {vector_file} ! Get a language compatible word vectors file from https://fasttext.cc/docs/en/crawl-vectors.html ')
        quit(0)
    epoch_start = 0
    word_embedding_matrix = embedding_matrix_creater(vector_file, WORD2VEC_DIMS, word2idx)
    model = create_model(LSTM_UNITS, word_embedding_matrix)
    model.summary()

N_SAMPLES = len(decoder_input_data)

N_TEST = int(N_SAMPLES*TEST_frac)

encoder_input_data_test = encoder_input_data[0:N_TEST,:]
decoder_input_data_test = decoder_input_data[0:N_TEST,:]
encoder_input_data = encoder_input_data[N_TEST + 1:,:]
decoder_input_data = decoder_input_data[N_TEST + 1:,:]

Step = int(np.around((N_SAMPLES - N_TEST) / NUM_SUBSETS))

SAMPLE_ROUNDS = Step * NUM_SUBSETS

def bot_reply(inp):
    inp = np.array(inp).reshape(-1, *inp.shape)
    ans_partial = np.zeros((1,MAX_INPUT_LENGTH))
    ans_partial[0, -1] = word2idx[BOS.strip()]  #  the index of the symbol BOS (begin of sentence)
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
            text = text + idx2word[k] + ' '
        except:
            continue

    text = text.replace(BOS.strip(), '')
    text = text.replace(EOS.strip(), '')
    text = " ".join(text.split())

    return text

def data_translate(idx_list):
    text = ''
    for each_id in idx_list:
        try:
            word = idx2word[each_id]
            text = text + word + ' '
        except:
            continue
    text = text.replace(BOS, '')
    text = text.replace(EOS, '')
    text = text.strip()
    return text

for n_epoch in range(epoch_start, NUM_EPOCHS):
    # Loop over training batches due to memory constraints
    for n_batch in range(0, int(SAMPLE_ROUNDS), Step):

        encoder_input_data2 = encoder_input_data[n_batch:n_batch+Step]

        counter = 0
        for id, sentence in enumerate(decoder_input_data[n_batch:n_batch+Step]):
            l = np.where(sentence == word2idx[EOS.strip()])  #  the position of the symbol EOS
            limit = l[0][0]
            counter += limit + 1

        question = np.zeros((counter, MAX_INPUT_LENGTH), dtype=np.uint16)
        answer = np.zeros((counter, MAX_INPUT_LENGTH), dtype=np.uint16)
        target = np.zeros((counter, VOCAB_SIZE), dtype=np.uint8)

        # Loop over the training examples:
        counter = 0
        for i, sentence in enumerate(decoder_input_data[n_batch:n_batch+Step]):
            ans_partial = np.zeros((1, MAX_INPUT_LENGTH), dtype=np.uint16)

            # Loop over the positions of the current target output (the current output sequence)
            l = np.where(sentence == word2idx[EOS.strip()])  #  the position of the symbol EOS
            limit = l[0][0]

            for k in range(1, limit + 1):
                # Mapping the target output (the next output word) for one-hot codding:
                target_2 = np.zeros((1, VOCAB_SIZE), dtype=np.uint8)
                target_2[0, sentence[k]] = 1

                # preparing the partial answer to input:
                ans_partial[0,-k:] = sentence[0:k]

                # training the model for one epoch using teacher forcing:
                question[counter, :] = encoder_input_data2[i:i+1]
                answer[counter, :] = ans_partial
                target[counter, :] = target_2
                counter += 1

        print(f'Training epoch: {n_epoch}, Training examples: {n_batch} - {n_batch + Step}')
        model.fit([question, answer], target, batch_size=BATCH_SIZE, epochs=1)

    if n_epoch % SAVE_EACH_EPOCH == 0:
        test_text_ratios = []
        epoch_ratio = 0
        while len(test_text_ratios) <= 4:

            i = np.random.randint(N_TEST)
            test_input = encoder_input_data_test[i]
            test_output = decoder_input_data_test[i]

            #print(f'input text= {data_translate(test_input)}')

            output_text = data_translate(test_output)
            result_text = bot_reply(test_input)

            text_ratio = fuzz.ratio(output_text, result_text)

            #print(f'outpt text= {output_text}')
            #print(f'reslt text= {result_text}')
            #print(f'text ratio= {text_ratio}')

            test_text_ratios.append(text_ratio)

        epoch_ratio = np.mean(test_text_ratios)
        print(f'epoch_ratio test = {epoch_ratio}')
        if epoch_ratio >= SAVE_RATIO:
            logs_df = logs_df.append({'Epoch': n_epoch, 'Ratio':int(epoch_ratio)}, ignore_index=True)
            logs_df.to_csv(f'saves\\{save_name}\\learn.log', index=False)
            model.save_weights(model_weights)

    del encoder_input_data2
    del question
    del answer
    del target
    gc.collect()
