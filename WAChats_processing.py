from WA_bot_libs import remove_links_fct, WAChats_processing
import configparser


config_parser = configparser.RawConfigParser()
config_parser.read('settings.ini', encoding='utf-8')
reply_id = config_parser.get('WA_process', 'REPLY_AS_ID')

WAChats_convs_df = WAChats_processing(reply_id = reply_id)
WAChats_convs_df.to_csv('learn_files\\conversation_pairs_WAchats.csv', index=False)
