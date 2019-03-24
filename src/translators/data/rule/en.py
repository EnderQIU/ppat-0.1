from string import digits

from big_phoney import PredictionModel, PhoneticDictionary

pred_model = PredictionModel()
phonetic_dict = PhoneticDictionary()
remove_digits = str.maketrans('', '', digits)  # Remove stress levels from big_phoney's results


def lookup_or_predict(word):
    """
    Lookup an english word's phonetics using big_phoney.
    If not found, predict the phonetics as the ARPAbet classified
     using big_phoney.PredictionModel
    :param word:
    :return:
    """
    result = phonetic_dict.lookup(word)
    if result is not None:
        result =  result.translate(remove_digits).split(' ')
    else:
        result = pred_model.predict(word).translate(remove_digits).split(' ')
    # Transform "OY" (written as "oi" in "boy") into a sequence of "OW" "IH"
    pos = result.index('OY')
    if pos:
        result[pos: pos+1] = ('OW', 'IH')
    return result

def post_process_people(word):
    """
    6-1 (弗)用于词首
    """
    s_word = ""
    if word.startswith('夫'):
        s_word += '弗'
        s_word += word[1:]
    else:
        s_word = word
    return s_word

def post_process_places(word):
    """
    1-1 汉字译名若产生望文生义现象时，应用该音节的同音异字译写。如“东”、“南”、“西”
    出现在地名开头时，用“栋”、“楠”、“锡”译写;“海”出现在地名结尾时，用“亥”译写。
    """
    s_word = ""
    if word.startswith('东'):
        s_word += '栋'
        s_word += word[1:]
    elif word.startswith('南'):
        s_word += '楠'
        s_word += word[1:]
    elif word.startswith('西'):
        s_word += '锡'
        s_word += word[1:]
    else:
        s_word = word
    s1_word = ""
    if s_word.endswith('海'):
        s1_word = s_word[:-1]
        s1_word += '亥'
    else:
        s1_word = s_word
    return s1_word
    