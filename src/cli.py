"""
Transliterate in CLI mode
"""
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
    s_available_codes = ''
    l_available_codes = []
    for k in rule_translator.rules.keys():
        s_available_codes += '\t' + k + '\t' + rule_translator.rules[k]['meta']['language_name'] + '\n'
        l_available_codes.append(k)
    print("""Usage:
    Type names and hit ENTER to get transliterations.
    Use option "-l" to specify language codes. Available codes are:
    
    {}
    Default is ALL language codes.
    Use Ctrl+C to quit.
    """.format(s_available_codes))

    while True:
        line = input('> ').split('-l')
        words = line[0].rstrip()
        lang_codes = []
        if len(line) == 2:
            lang_codes = line[1].strip().split(' ')
            for c in lang_codes:
                if c not in l_available_codes:
                    print('ERROR: "{}" is not an available language code.'.format(c))
                    exit(-1)
        print('Result:\n===================================================')
        print('From dictionary:')
        d_result = index_translator.search(words)
        print('Keyword\tLanguage\tCategory\tChinese')
        for d_r in d_result['transliterations']:
            print('{}\t{}\t{}\t{}'.format(d_r['keyword'], d_r['language'], d_r['category'], d_r['chinese']))
        print('---------------------------------------------------')
        print('From rule:')
        print('Keyword\tLanguage\tCategory\tChinese')
        r_result = rule_translator.translate(words, lang_codes)
        for d_r in r_result['transliterations']:
            print('{}\t{}\t{}\t{}'.format(d_r['keyword'], d_r['language'], d_r['category'], d_r['chinese']))
        print('===================================================')
