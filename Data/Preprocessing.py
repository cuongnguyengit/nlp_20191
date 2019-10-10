import pandas as pd
import sys
import re

import unicodedata as ud

PATH_VNE_DICT_ADD_TONE = 'add_tone_model/data/vne_dict.json'

PATH_BINARY_MODEL_TONE = 'add_tone_model/models'


class Preprocessing:

    def __init__(self, path_dir='pattern/'):
        self.special_signs = ["!", '"', "$", "&", "'",
                         "(", ")", ",", "-", ".",
                         ":", ";", "<", "=", ">", "?",
                         "@", "[", "\\", "]", "^", "`",
                         "{", "|", "}", "~"]

        path_excel = path_dir + 'DS_Tuviettat_mbccs.xlsx'
        self.tu_viet_tat = [str(i).lower() for i in pd.read_excel(path_excel, 'Sheet1')['token']]
        self.tu_day_du = [str(i).lower() for i in pd.read_excel(path_excel, 'Sheet1')['fullname']]

    def sylabelize(self, text):
        text = ud.normalize('NFC', text)

        specials = ["==>", "->", "\.\.\.", ">>"]
        digit = "\d+([\.,_]\d+)+"
        email = "([a-zA-Z0-9_.+-]+@([a-zA-Z0-9-]+\.)+[a-zA-Z0-9-]+)"
        #web = "^(http[s]?://)?(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+$"
        web = "\w+://[^\s]+"
        #datetime = [
        #    "\d{1,2}\/\d{1,2}(\/\d{1,4})(^\dw. )+",
        #    "\d{1,2}-\d{1,2}(-\d+)?",
        #]
        word = "\w+"
        non_word = "[^\w\s]"
        abbreviations = [
            "[A-ZĐ]+\.",
            "Tp\.",
            "Mr\.", "Mrs\.", "Ms\.",
            "Dr\.", "ThS\."
        ]

        patterns = []
        patterns.extend(abbreviations)
        patterns.extend(specials)
        patterns.extend([web, email])
        #patterns.extend(datetime)
        patterns.extend([digit, non_word, word])

        patterns = "(" + "|".join(patterns) + ")"
        if sys.version_info < (3, 0):
            patterns = patterns.decode('utf-8')
        tokens = re.findall(patterns, text, re.UNICODE)

        return ' '.join([token[0] for token in tokens])

    def get_nomal_sentence(self, target_name):
        """
        :param target_name:
        :return:
        """

        # normalize the format long number
        #
        # remove unnecessary sign
        target_name = self.sylabelize(target_name).lower()

        for sign in self.special_signs:
            if sign in target_name:
                target_name = target_name.replace(sign, " ")
        #
        # remove the unnecessary white spaces
        list_words = target_name.split()
        target_name = ' '.join(word.strip() for word in list_words)

        # # Fix typos using dictionary
        target_name = " " + target_name + " "

        for i, tvt in enumerate(self.tu_viet_tat):
            tvt = ' ' + tvt.strip() + ' '
            tdd = ' '+ self.tu_day_du[i].strip() + ' '
            if tvt in target_name:
                target_name = target_name.replace(tvt, tdd)

        return target_name.strip()


if __name__ == '__main__':
    pre = Preprocessing()
    print(pre.get_nomal_sentence('hom nay la thu 2 1/2?'))