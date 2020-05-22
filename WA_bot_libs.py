import pandas as pd
import os
import re

def remove_links_fct(string, replace_with=''):
    regex = r"\b((?:https?://)?(?:(?:www\.)?(?:[\da-z\.-]+)\.(?:[a-z]{2,6})|(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)|(?:(?:[0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,7}:|(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,5}(?::[0-9a-fA-F]{1,4}){1,2}|(?:[0-9a-fA-F]{1,4}:){1,4}(?::[0-9a-fA-F]{1,4}){1,3}|(?:[0-9a-fA-F]{1,4}:){1,3}(?::[0-9a-fA-F]{1,4}){1,4}|(?:[0-9a-fA-F]{1,4}:){1,2}(?::[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:(?:(?::[0-9a-fA-F]{1,4}){1,6})|:(?:(?::[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(?::[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(?:ffff(?::0{1,4}){0,1}:){0,1}(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])|(?:[0-9a-fA-F]{1,4}:){1,4}:(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])))(?::[0-9]{1,4}|[1-5][0-9]{4}|6[0-4][0-9]{3}|65[0-4][0-9]{2}|655[0-2][0-9]|6553[0-5])?(?:/[\w\.-]*)*/?)\b"
    matches = re.findall(regex, string)
    if len(matches)>0:
        for each in matches:
            link_start = string.find(each)
            link_end = string.find(' ', link_start)
            if link_end == -1:
                link_end = len(string)
            full_link = string[link_start:link_end]
            string = string.replace(full_link, replace_with)
    return string

def WAChats_processing(reply_id, max_conv_pause=300, chats_path='learn_files\\WhatsAppChats', remove_links = True, new_line_char = None):
    '''
    reply_id - name of chat user that replys

    max_conv_pause = 300 (default) in seconds, pause in chat to be considered a new conersation

    chats_path = 'learn_files\\WhatsAppChats' (default), only exported chats\
    
    remove_links = True (default) attempt to remove links

    Returns a pandas with columns Input and Reply
    '''

    if new_line_char is None:
        new_line_char = ' '
    
    def load_chat_file(file):
        OUT_df = pd.read_csv(file, sep=" [A,P]M - " , names = ["Time", "Name+Msg"], encoding='utf-8', engine = 'python')
        new = OUT_df["Name+Msg"].str.split(": ", n = 1, expand = True)
        OUT_df["Name"]= new[0]
        OUT_df["Message"]= new[1]
        OUT_df.drop(columns =["Name+Msg"], inplace = True)
        # OUT_df cleaning

        # delete sistem messages
        OUT_df.drop(OUT_df[OUT_df['Name'].str.len() > 15 ].index, inplace=True)
        OUT_df.reset_index(drop=True, inplace=True)
        data_index = 0

        while data_index < len(OUT_df):
            # drop <Media omitted> flag
            if OUT_df['Message'][data_index]  == '<Media omitted>':
                OUT_df.drop(data_index, inplace=True)
                OUT_df.reset_index(drop=True, inplace=True)
                data_index -= 1
            # concat newlines in previous Message
            try:
                OUT_df['Time'][data_index] = pd.to_datetime(OUT_df['Time'][data_index])
            except:
                OUT_df['Message'][data_index - 1] = OUT_df['Message'][data_index - 1] + new_line_char + OUT_df['Time'][data_index]
                OUT_df.drop(data_index, inplace=True)
                OUT_df.reset_index(drop=True, inplace=True)
                data_index -= 1

            data_index += 1

        return OUT_df

    for filename in os.listdir(chats_path):
        try:
            chats_data = chats_data.append(load_chat_file(f'{chats_path}\\{filename}'))
        except:
            chats_data = load_chat_file(f'{chats_path}\\{filename}')
    chats_data.reset_index(drop=True, inplace=True)

    prev_msg_time = chats_data['Time'][0]
    convs_df = pd.DataFrame(columns=['Input', 'Reply'])
    last_msg_from = ''
    convs_df_row = 0

    for _, row in chats_data.iterrows():

        if remove_links:
            row['Message'] = remove_links_fct(row['Message'])

        if (row['Time'] - prev_msg_time).total_seconds() > max_conv_pause:
            convs_df_row += 1
            last_msg_from = ''
        elif reply_id in last_msg_from and row['Name'] != last_msg_from:
            convs_df_row += 1

        if reply_id in row['Name']:
            
            if last_msg_from == row['Name'] and not convs_df.loc[convs_df_row, 'Reply'] == '':
                convs_df.loc[convs_df_row, 'Reply'] = convs_df.loc[convs_df_row, 'Reply'] + new_line_char + row['Message']
            else:
                convs_df.loc[convs_df_row, 'Reply'] = row['Message']
        else:
            if last_msg_from == row['Name'] and not convs_df.loc[convs_df_row, 'Input'] == '':
                convs_df.loc[convs_df_row, 'Input'] = convs_df.loc[convs_df_row, 'Input'] + new_line_char + row['Message']
            else:
                convs_df.loc[convs_df_row, 'Input'] = row['Message']

        last_msg_from = row['Name']
        prev_msg_time = row['Time']

    convs_df = convs_df.drop(convs_df[convs_df['Reply'] == ''].index)
    convs_df = convs_df.drop(convs_df[convs_df['Input'] == ''].index)

    convs_df.dropna(inplace=True)
    convs_df.reset_index(inplace=True, drop=True)

    return convs_df

# modified from https://github.com/JMGama/WhatsApp-Scraping
import configparser
import time
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.keys import Keys

class WA_srcaper:
    def __init__(self, settings_file):
        self.settings_file = settings_file

        self.settings = self.load_settings()
        self.driver, self.hellper_driver = self.load_driver()
        self.driver.get(self.settings['page'])  # load https://web.whatsapp.com/
        self.regex = re.compile(r'>(.*?)<img')
        self.regex_last = re.compile(r'">(.*?)</span>')
        self.chatter = None
        self.message = None
        self.start_stop_flg = None

    def load_settings(self):
        """
        Loading and assigning global variables from our settings.txt file
        """
        config_parser = configparser.RawConfigParser()
        config_parser.read(self.settings_file, encoding='utf-8')

        browser = config_parser.get('WA_srcaper', 'BROWSER')
        browser_path = config_parser.get('WA_srcaper', 'BROWSER_PATH')
        page = config_parser.get('WA_srcaper', 'PAGE')
        chatters = config_parser.get('WA_srcaper', 'CHATTERS')
        chatters = [item.strip() for item in chatters.split(',')]
        click_delay = int(config_parser.get('WA_srcaper', 'CLICK_DELAY'))

        self.START_BOT = config_parser.get('WA_process', 'START_BOT')
        self.STOP_BOT = config_parser.get('WA_process', 'STOP_BOT')
        self.REPLY_AS_ID = config_parser.get('WA_process', 'REPLY_AS_ID')
        self.max_conv_pause = int(config_parser.get('WA_process', 'max_conv_pause'))
        self.MAX_INPUT_LENGTH = int(config_parser.get('BOT', 'MAX_INPUT_LENGTH'))
        self.BOT_ICON = config_parser.get('WA_process', 'BOT_ICON')
        self.BOT_ICON = self.BOT_ICON.strip() + ' '

        settings = {
            'browser': browser,
            'browser_path': browser_path,
            'page': page, 
            'chatters': chatters,
            'click_delay': click_delay
        }
        return settings

    def load_driver(self):
        """
        Load the Selenium driver depending on the browser
        (Edge and Safari are skipped)
        """
        driver = None
        hellper_driver = None
        
        if self.settings['browser'] == 'firefox':
            firefox_profile = webdriver.FirefoxProfile(self.settings['browser_path'])
            driver = webdriver.Firefox(firefox_profile)
            options = webdriver.firefox.options.Options()
            options.headless = True
            hellper_driver = webdriver.Firefox(options=options)
        elif self.settings['browser'] == 'chrome':
            chrome_options = webdriver.ChromeOptions()
            chrome_options.add_argument('user-data-dir=' + self.settings['browser_path'])
            driver = webdriver.Chrome(options=chrome_options)
            hellper_driver = webdriver.Chrome()
        elif self.settings['browser'] == 'safari':
            pass
        elif self.settings['browser'] == 'edge':
            pass

        return driver, hellper_driver

    def select_chatter(self, chatter):
        """
        Function that search the specified user and activates his chat
        """
        if isinstance(chatter, list):
            chatter = chatter[0]
        while True:
            time.sleep(1)
            chatters_obj = self.driver.find_elements_by_xpath(".//div[contains(@class, '_25Ooe')]")
            for chatter_obj in chatters_obj:
                if chatter_obj.text == chatter:
                    chatter_obj.click()
                    self.chatter = chatter
                    self.message = None
                    self.start_stop_flg = None
                    return True
            if len(chatters_obj) > 0:
                print(f'select_chatter -> chatter= {chatter} not found')
                return False

    def read_web_messages(self):
        messages_df = pd.DataFrame(columns=['Time+Name', 'Message'])
        time.sleep(2)
        messages_container = self.driver.find_elements_by_xpath("//div[contains(@class,' message')]")

        for messages in messages_container:
            try:
                message_container = None
                data = None
                message = ''
                message_container = messages.find_element_by_xpath(".//div[contains(@class,'copyable-text')]")
                data = message_container.get_attribute("data-pre-plain-text")

                try:
                    message_container = message_container.find_element_by_xpath(".//span[contains(@class,'selectable-text invisible-space copyable-text')]")
                    message = message_container.text

                    emojis = message_container.find_elements_by_xpath(".//img[contains(@class,'selectable-text invisible-space copyable-text')]")
                    if len(emojis) > 0:
                        # recreate message because it contains emojis
                        message = ''
                        innerHTML = message_container.get_attribute('innerHTML')
                        scan_string = True
                        while scan_string:
                            substrings = self.regex.finditer(innerHTML)
                            scan_string = False
                            for each in substrings:
                                g0 = each.group(0)
                                g1 = each.group(1)
                                if (len(g1) > 0) and ('span>' not in g1):
                                    replacement_text = '><span>' + g1 + '</span><img'
                                    g0 = re.escape(g0)
                                    innerHTML = re.sub(g0, replacement_text, innerHTML, 1)
                                    scan_string = True
                                    break

                        last_substrings = self.regex_last.finditer(innerHTML)
                        for each in last_substrings:
                            g0 = each.group(0)
                            g1 = each.group(1)
                            if (len(g1) > 0) and ('<span>' not in g1):
                                replacement_text = '"><span>'+g1+'</span></span>'
                                g0 = re.escape(g0)
                                innerHTML = re.sub(g0, replacement_text, innerHTML, 1)

                        # recreate message
                        self.hellper_driver.get("data:text/html;charset=utf-8," + innerHTML)
                        for each in self.hellper_driver.find_elements_by_xpath("//span/*"):

                            if each.tag_name == 'span':
                                message = message + each.text
                            elif each.tag_name == 'img':
                                message = message + each.get_attribute("data-plain-text")

                except: # if message has only emojis
                    message_container = message_container.find_element_by_xpath(".//div[contains(@class,'selectable-text invisible-space copyable-text')]")
                    for emoji in message_container.find_elements_by_xpath(".//img[contains(@class,'selectable-text invisible-space copyable-text')]"):
                        message = message + emoji.get_attribute("data-plain-text")

                messages_df = messages_df.append({'Time+Name': data, 'Message':message}, ignore_index=True)

            except Exception as e:  # pictures/docs and deleted messages; maybe other stuff
                print(e)
                print('WA_srcaper > read_web_messages --> Exception cause: possible pictures, docs or deleted messages')
                pass

        new = messages_df["Time+Name"].str.split("] ", n = 1, expand = True)
        messages_df["Time"]= new[0]
        messages_df["Name"]= new[1]
        messages_df.drop(columns =["Time+Name"], inplace = True)
        
        messages_df["Time"] = messages_df["Time"].str.replace('[','')
        messages_df["Name"] = messages_df["Name"].str.replace(':','')

        messages_df["Time"] = pd.to_datetime(messages_df["Time"])

        self.messages_df = messages_df
        return self.messages_df

    def scan_messages(self, messages_df = None, START_BOT = None, STOP_BOT = None, max_conv_pause = None, MAX_INPUT_LENGTH = None, new_line_char = None):
        '''
        messages_df result of read_web_messages()

        START_BOT, STOP_BOT (default None) as strings, flags for self.start_stop_flg - True for START_BOT detected, False for STOP_BOT detected, None for not detected.

        max_conv_pause as integer in seconds, pause in chat to be considered a new conersation

        MAX_INPUT_LENGTH as integer, maximum number of most recent words and emojis in self.message

        Returns self.message, self.start_stop_flg - self.message is pre-preocessed with 
        '''
        if messages_df is None:
            messages_df = self.messages_df
        reply_as = self.REPLY_AS_ID
        if START_BOT is None:
            START_BOT = self.START_BOT
        if STOP_BOT is None:
            STOP_BOT = self.STOP_BOT
        if max_conv_pause is None:
            max_conv_pause = self.max_conv_pause
        if MAX_INPUT_LENGTH is None:
            MAX_INPUT_LENGTH = self.MAX_INPUT_LENGTH
        if new_line_char is None:
            new_line_char = ' '
        
        message = ''
        start_stop_flg = None

        # count START/STOP flags
        last_stop_idx = messages_df[messages_df['Message'] == STOP_BOT].index.values
        if len(last_stop_idx) == 0:
            last_stop_idx = -1
        else:
            last_stop_idx = int(last_stop_idx[-1])
        last_start_idx = messages_df[messages_df['Message'] == START_BOT].index.values
        if len(last_start_idx) == 0:
            last_start_idx = -1
        else:
            last_start_idx = int(last_start_idx[-1])
        if  last_start_idx < last_stop_idx:
            start_stop_flg = False
        elif last_start_idx > last_stop_idx:
            start_stop_flg = True

        if reply_as in messages_df['Name'].tail(1).values[0]:
            self.message = message
            self.start_stop_flg = start_stop_flg
            return self.message, self.start_stop_flg

        messages_df = messages_df.sort_index(axis=0 ,ascending=False)
        messages_df.dropna(inplace=True)
        prev_msg_time = None
        for _, row in messages_df.iterrows():
            if reply_as not in row['Name']:
                if prev_msg_time is None or (prev_msg_time - row['Time']).total_seconds() <= max_conv_pause :
                    row_msg = row['Message']
                    row_msg = remove_links_fct(row_msg)
                    row_msg = ' '.join(row_msg.split())
                    if message.strip().count(' ') + row_msg.count(' ') + 2 <= MAX_INPUT_LENGTH:
                        message = row_msg + new_line_char + message
                        message = message.strip()
                        prev_msg_time = row['Time']
                    else:
                        break
                else:
                    break
            else:
                break
        
        self.message = message
        self.start_stop_flg = start_stop_flg
        return self.message, self.start_stop_flg

    def send_messages(self, message, new_line_char = None, BOT_ICON = None):

        if BOT_ICON is None:
            BOT_ICON = self.BOT_ICON
        send_msg_obj = self.driver.find_element_by_xpath(".//div[contains(@class, '_3F6QL _2WovP')]")
        send_msg_obj = send_msg_obj.find_element_by_xpath(".//div[contains(@class, '_2S1VP copyable-text selectable-text')]")
        send_msg_obj.click()

        if new_line_char is None or new_line_char == ' ':
            messages = [message]
        else:
            messages = [item.strip() for item in message.split(new_line_char)]
        for each_message in messages:
            send_msg_obj.send_keys(BOT_ICON + each_message + Keys.ENTER)
        return

    def stop_drivers(self):
        self.driver.quit()
        self.hellper_driver.quit()
        return