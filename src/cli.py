"""
Transliterate in CLI mode
"""
import sys

from translators.translator import IndexTranslator, RuleTranslator



if __name__ == '__main__':
    print("""!!!Welcome to Places & People Automate Translator!!!
     /$$$$$$$  /$$$$$$$   /$$$$$$  /$$$$$$$$
    | $$__  $$| $$__  $$ /$$__  $$|__  $$__/
    | $$  \ $$| $$  \ $$| $$  \ $$   | $$   
    | $$$$$$$/| $$$$$$$/| $$$$$$$$   | $$   
    | $$____/ | $$____/ | $$__  $$   | $$   
    | $$      | $$      | $$  | $$   | $$   
    | $$      | $$      | $$  | $$   | $$   
    |__/      |__/      |__/  |__/   |__/   
    """)

    index_translator = IndexTranslator()
    rule_translator = RuleTranslator()
    available_codes = ''
    for k in rule_translator.rules.keys():
        available_codes += '\t' + k + '\t' + rule_translator.rules[k]['meta']['language_name'] + '\n'
    print("""Usage:
    Type names and hit ENTER to get transliterations.
    Use option "-l" to specify language codes. Available codes are:
    
    {}
    Default is ALL language codes.
    Use Ctrl+C to quit.
    """.format(available_codes))

    while True:
        line = input('> ').split('-l')
        words = line[0].rstrip()
        lang_codes = []
        if len(line) == 2:
            lang_codes = line[1].lstrip().split(' ')
        print('Result:\n=================')
        print('From dictionary:')
        d_result = index_translator.search(words)
        print('People:')
        for k in d_result['people'].keys():
            print(k)
            print('{}\t{}\t{}'.format(d_result['people'][k]['name'],
                                      d_result['people'][k]['culture'],
                                      d_result['people'][k]['chinese']))
        print('-----------------')
        print('Places:')
        for k in d_result['places'].keys():
            print(k)
            print('{}\t{}\t{}'.format(d_result['places'][k]['name'],
                                      d_result['places'][k]['culture'],
                                      d_result['places'][k]['chinese']))
        print('-----------------')
        print('From rule:')
        r_result = rule_translator.translate(words, lang_codes)
        for r_l in r_result:
            print('Language:', r_l['lang_code'])
            for t in r_l['transliterations']:
                print('People:', t['people'], 'Places:', t['places'])
