# Translates movie_lines.txt from Cornell Movie Dialogs Corpus - https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html - and generates a csv file for further upload

from translate import Translator    # from https://pypi.org/project/translate/
import pandas as pd
import os

FILE_PATH = 'learn_files\\movie_lines_all_translate.csv'
TRANSLATE_LANG = 'ro'   # None for not translating

TRANSLATION_CHAR_LIMIT = 2000000
TRANSLATION_WORD_LIMIT = 500000

update_translated = True

try:
    mlt_df = pd.read_csv(FILE_PATH)
except:
    ORIGINAL_FILE_PATH = 'learn_files\\movie_lines.txt'
    if os.path.isfile(ORIGINAL_FILE_PATH):
        mlt_df = pd.read_csv(ORIGINAL_FILE_PATH, sep=r'\+\+\+\$\+\+\+' , names = ['Line_ID', 'User_ID', 'Movie_ID', 'User', 'Text'], engine = 'python', encoding='ISO-8859-1')
        mlt_df.to_csv(FILE_PATH, index=False)
    else:
        print(f'Not found {ORIGINAL_FILE_PATH} ! Get movie_lines.txt from Cornell Movie Dialogs Corpus from https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html ')

if TRANSLATE_LANG is not None:
    translated_column = f'Translated_{TRANSLATE_LANG}'
    if translated_column in mlt_df.columns.tolist():
        start_index = mlt_df[translated_column].last_valid_index() + 1
    else:
        start_index = 0
        mlt_df[translated_column] = pd.Series([], dtype="string")

    if start_index >= len(mlt_df):
        print(f'No translation needed for language {TRANSLATE_LANG}')
        mlt_df.to_csv(FILE_PATH, index=False)
    else:
        #secret_microsoft = 'secred_microsoft_key' # see uselfull steps here https://ordasoft.com/News/SEF-Translate-Documentation/how-to-get-bing-translate-api.html
        #translator = Translator(provider='microsoft', secret_access_key=secret_microsoft, to_lang=TRANSLATE_LANG)   # Microsoft translator
        translator = Translator(to_lang=TRANSLATE_LANG)  # MYMEMORY translator

        total_char_count = 0
        total_word_count = 0

        for i in range(start_index, mlt_df.shape[0]):

            try:
                text2translate = mlt_df.iloc[i]['Text'].strip()
                char_count = len(text2translate)
                word_count = text2translate.count(' ') + 1
                total_char_count += char_count
                total_word_count += word_count
            except:
                continue

            if total_char_count > TRANSLATION_CHAR_LIMIT or total_word_count > TRANSLATION_WORD_LIMIT:
                print(f'TRANSLATION_CHAR_LIMIT= {TRANSLATION_CHAR_LIMIT} or TRANSLATION_WORD_LIMIT= {TRANSLATION_WORD_LIMIT} reached --> {total_char_count} or {total_word_count}')
                break

            print(f'Tranlating text into {TRANSLATE_LANG} --> {text2translate} -->', end=' ')

            try:
                translated_text = translator.translate(text2translate)
                print(translated_text)
                print(f'Translated count -> chars= {total_char_count} words= {total_word_count}')
                
                if 'MYMEMORY WARNING' in translated_text or 'IS AN INVALID TARGET LANGUAGE' in translated_text:
                    print(translated_text)
                    update_translated = False
                    break
                mlt_df.loc[i, translated_column] = translated_text
            except Exception as e:
                print('Translation exception occured')
                print(e)
                update_translated = False
                break

            if update_translated:
                mlt_df.to_csv(FILE_PATH, index=False)
