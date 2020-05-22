import pandas as pd

FILE_PATH = 'learn_files\\movie_lines_all_translate.csv'
TRANSLATE_LANG = 'ro'   # None for original
#TRANSLATE_LANG = None
SILENCE_TEXT = '😈'

ml_df = pd.read_csv(FILE_PATH)

if f'Translated_{TRANSLATE_LANG}' in ml_df.columns.tolist():
    translated_column = f'Translated_{TRANSLATE_LANG}'
else:
    print(f'Language {TRANSLATE_LANG} not prezent in {FILE_PATH}. Will pair original text')
    translated_column = 'Text'
    TRANSLATE_LANG = 'orig'

ml_df['Line_ID'].replace({'L':''}, inplace=True, regex=True)
ml_df['Line_ID'] = pd.to_numeric(ml_df['Line_ID'], errors='coerce')

ml_df['Movie_ID'].replace({'m':''}, inplace=True, regex=True)
ml_df['Movie_ID'] = pd.to_numeric(ml_df['Movie_ID'], errors='coerce')

dialog_df = pd.DataFrame(columns=['Dialog_ID', 'Input', 'Reply'])

dialog_dict ={}
dialog_ID = 0
last_line_ID = -1

stop_index = ml_df[translated_column].last_valid_index()

for i in range(0, stop_index):
    if (ml_df.iloc[i]['Line_ID'] != last_line_ID - 1) or (ml_df.iloc[i]['Movie_ID'] != last_movie_ID):
        dialog_ID += 1
        dialog_dict = {}

    if len(dialog_dict) == 0:
        dialog_dict['Dialog_ID'] = dialog_ID
        dialog_dict['Reply'] = ml_df.iloc[i][translated_column]
        dialog_dict['Line_ID_Reply'] = str(ml_df.iloc[i]['Line_ID'])
        reply_user_ID = ml_df.iloc[i]['User_ID']
    else:
        dialog_dict['Input'] = ml_df.iloc[i][translated_column]
        dialog_dict['Line_ID_Input'] = str(ml_df.iloc[i]['Line_ID'])
        dialog_df = dialog_df.append(dialog_dict, ignore_index=True)
        dialog_dict = {}

    last_line_ID = ml_df.iloc[i]['Line_ID']
    last_movie_ID = ml_df.iloc[i]['Movie_ID']

dialog_df.fillna(SILENCE_TEXT, inplace=True)

dialog_df = dialog_df.rename_axis('MyIdx')
dialog_df.sort_values(by = ['Dialog_ID', 'MyIdx'], ascending = [True, False], inplace=True)

dialog_df.to_csv(f'learn_files\\movie_dialog_pairs_{TRANSLATE_LANG}.csv', index=False)
