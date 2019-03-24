import os
import json
import re
import importlib


class IndexTranslator:
    """
    Transliterating by dictionary
    """

    def __init__(self):
        print('==========================================')
        print('Initializing index translator...')
        with open(os.path.join('.', 'translators', 'data', 'index', 'people.json'), 'r', encoding='utf8') as people_json:
            with open(os.path.join('.', 'translators', 'data', 'index', 'places.json'), 'r', encoding='utf8') as place_json:
                try:
                    print('Loading file "data/index/places.json" ... ', end='')
                    people_data = json.loads(people_json.read())
                    print('OK.')
                    print('Loading file "data/index/people.json" ... ', end='')
                    place_data = json.loads(place_json.read())
                    print('OK.')
                except json.JSONDecodeError as e:
                    print(str(e))
                    exit(1)
                else:
                    self.people_index = {}
                    self.places_index = {}

                    # build indexes
                    print('Indexing people names data ... ', end='')
                    for m in people_data:
                        if self.people_index.__contains__(m['name']):
                            self.people_index[m['name']].append(m)
                        else:
                            self.people_index[m['name']] = [m]
                    print('OK.')
                    print('Indexing places names data ... ', end='')
                    for n in place_data:
                        if self.places_index.__contains__(n['name']):
                            self.places_index[n['name']].append(n)
                        else:
                            self.places_index[n['name']] = [n]
                    print('OK.')
                    print('Index Translator initialized successfully!')
                    print('==========================================')

    def search(self, keyword):
        """
        param: keyword 
        Must be a name of a place or a person.
        Spaces are not permitted.
        
        return: [<dict>...]
        """
        assert isinstance(keyword, str) and ' ' not in keyword

        keyword = keyword.capitalize()
        results = {'transliterations': []}
        people_index_items = self.people_index.get(keyword, -1)
        places_index_items = self.places_index.get(keyword, -1)
        if people_index_items != -1:
            for people_index_item in people_index_items:
                result = {
                    'keyword': keyword,
                    'category': 'People',
                    'language': people_index_item['culture'],
                    'chinese': people_index_item['chinese'],
                }
                results['transliterations'].append(result)
        if places_index_items != -1:
            for places_index_item in places_index_items:
                result = {
                    'keyword': keyword,
                    'category': 'Places',
                    'language': places_index_item['culture'],
                    'chinese': places_index_item['chinese'],
                }
                results['transliterations'].append(result)

        return results


class NoRuleMatched(Exception):
    def __init__(self, section, l_phonetics=None):
        self.section = section
        self.l_phonetics = l_phonetics

    def __str__(self):
        return 'PHONECTIS: "{}"\n \
            NO RULE MATCHED IN "{}" section. CHECK YOUR RULE FILES'.format(str(self.l_phonetics), self.section)


class RuleTranslator:
    """
    Transliterating by rules
    """
    def __init__(self):
        print('Initializing rule translator...')
        self.rules = {}
        self.current_rule = {}
        for file_path in os.listdir(os.path.join('.', 'translators', 'data', 'rule')):
            if os.path.splitext(file_path)[1] == '.rule':
                print('Found rule file: {} ... '.format(file_path), end='')
                file_path = os.path.join('.', 'translators', 'data', 'rule', file_path)
                with open(file_path, 'r', encoding='utf8') as rule_file:
                    print('loading...', end='')
                    self._load_rule(file_path, rule_file)
                    print('OK.')
        print('==========================================')
        print('All "*.rule" files in "data/rule/" are loaded!')
        print('==========================================')

    def _get_kv(self, line, line_number, lang_code):
        """
        Get a k, v pair from a line likes 'key   =  value'.
         Key and value could contains spaces but useless spaces between '=' will be removed
        :param line:
        :return: (k, v)
        """
        items = line.split('=')

        assert len(items) == 2, "Invalid key/value pair at line: {} \n in file: {}.rule".format(line_number, lang_code)

        return items[0].rstrip(), items[1].lstrip().rstrip('\n')

    def _load_kv(self, line, current_section, line_number, lang_code):
        """
        Load Key/Value of a line
        :param line:
        :param current_section:
        :param line_number:
        :param lang_code:
        :return:
        """
        def parse_k_cv(_k):  # Parse Key in Consonants/Vowels section: Get possible <pre> and <post>
            s_pre, s_pattern, s_post = '', '', ''
            for i in range(len(_k)):
                if _k[i] == ')':
                    s_pre = _k[0: i].strip()
                elif _k[i] == '(':
                    s_post = _k[i + 1:].strip()
            s_pattern = _k.lstrip(s_pre + ')').rstrip(s_post + '(').strip()  # remove pre and post to get middle pattern
            assert s_pattern is not '', \
                'Invalid key (<match> not found): {} in "{}" section at line {} in file {}.py'\
                    .format(current_section, _k, line_number, lang_code)
            # parse s_pre, s_match, s_post into list, tuple(list) and list
            # parse s_match
            _l_pattern = re.split(r'\s*[|]\s*', s_pattern)
            l_pattern = []
            for p in _l_pattern:
                l_pattern.append(tuple(p.split(' ')))
            tl_pattern = tuple(l_pattern)
            # parse s_pre and s_post
            t_pre = () if s_pre == '' else tuple(s_pre.split(' '))
            t_post = () if s_post == '' else tuple(s_post.split(' '))
            # tuple is hashable while list is not.
            return t_pre, tl_pattern, t_post  # (<pre>, ([<match1>], [<match2>], ...), <post>)

        def parse_k_t(_k):  # Parse Key in Transliteration section: 'm, n' --> (m, n)
            m, n = 0, 0
            for i in range(len(_k)):
                if _k[i] == ',':
                    m = int(_k[0: i])
                    n = int(_k[i + 1:].lstrip(' '))
                    break
            assert m > 0 and n > 0 and m * n > 1, \
                'x, y in "<{}>" pair should be greater than 0 and not be 1 in the same time \
                    in ".transliteration" section at line {} in file {}.py'\
                        .format(_k, line_number, lang_code)
            return m, n

        try:
            if current_section not in self.rules[lang_code].keys():
                self.rules[lang_code][current_section] = {}  # init 'current_section' as a dict
            k, v = self._get_kv(line, line_number, lang_code)

            # validate k/v
            if current_section == 'phonetics':
                assert k in ('consonants', 'vowels'), \
                    'Invalid key (should be "consonants" or "vowels"): "{}" ' \
                    'in ".phonetics" section at line {} in file {}.py' \
                        .format(k, line_number, lang_code)
                self.rules[lang_code][current_section][k] = v.split('|')
            elif current_section.startswith('consonants') or current_section.startswith('vowels'):
                # v should be a digit
                assert v.isdigit(), 'Value should be a digit in section "{}": line {} in file: {}.rule'\
                    .format(current_section, line_number, lang_code)
                self.rules[lang_code][current_section][parse_k_cv(k)] = int(v)
            elif current_section.startswith('transliteration'):
                # Key in this section should be a <k, v> pair
                assert v is not '', 'Value should not be empty in section "{}": line {} in file: {}.rule'\
                    .format(current_section, line_number, lang_code)
                self.rules[lang_code][current_section][parse_k_t(k)] = v
        except Exception as e:
            print('Error: {} while loading k/v in "{}" section at line {} in file: {}.rule' \
                  .format(str(e), current_section, line_number, lang_code))
            exit(1)

    def _check_meta_k(self, k, line_number, lang_code):
        """
        Check whether the key in meta section is available.
        :param k:
        :param line_number:
        :param lang_code:
        :return:
        """
        self.available_meta_keys = ('language_name',)
        assert k in self.available_meta_keys, \
            'Invalid key in ".meta" section at line {} in file: {}.rule'.format(line_number, lang_code)

    def _load_func(self, section, line, lang_code, line_number, rule_file):
        """
        Load sections contain functions
        :param section:
        :param line:
        :return:
        """
        if section not in self.rules[lang_code].keys():  # init 'section' as a list
            self.rules[lang_code][section] = []
        k, v = self._get_kv(line, line_number, lang_code)  # lines here will be 'out = in' or 'out = fun(in)'
        if k == 'out' and v == 'in':  # just copy words
            self.rules[lang_code][section] = lambda x: x
        elif re.match(r'\w+\(in\)', v) is not None:
            function_name = v.split('(')[0]  # Get the function name and import it dynamically

            assert os.path.exists(os.path.join('.', 'translators', 'data', 'rule', lang_code + '.py')), \
                'No such file: {}.py in "data/rule/" for {}() in ".{}" section' \
                ' at line {} in file: {}.rule'.format(lang_code, function_name, section, line_number, lang_code)

            self.rules[lang_code][section] = \
                importlib.import_module('translators.data.rule.{}'.format(lang_code)).__getattribute__(function_name)
        else:
            print('Syntax Error in ".{}" section at line {} in rule file: {}'.format(section, line_number, rule_file))
            print('"out = in" or syntax like "out = <func>(in)" expected.')
            exit(1)


    def _load_rule(self, file_path, rule_file):
        """
        Load rules in "*.rule" file. Store it into self.rules
        :param file_path:
        :param rule_file:
        """
        current_section = ''
        line_number = 0
        lang_code = os.path.split(os.path.splitext(file_path)[0])[1]  # Get 'a' from '/c/d/a.rule'
        self.rules[lang_code] = {'meta': {}}
        available_sections = ('phonetics',  # consonants + vowels
                              'consonants people',
                              'vowels people',
                              'transliteration people',
                              'consonants places',
                              'vowels places',
                              'transliteration places',
                              )
        for line in rule_file.readlines():  # this will **preserve** the \n at end of lines
            line_number += 1
            if line.startswith('//') or line.startswith('\n'):  # line of comment
                continue
            if line.startswith('.'):  # line of section header
                current_section = line.rstrip('\n').lstrip('.')  # current_section = 'XX XX'
                continue

            # If this line is not continued by starting with dot,
            # treat the line as a key/value pair section content
            if current_section == 'meta':  # meta information
                k, v = self._get_kv(line, line_number, lang_code)
                self._check_meta_k(k, line_number, lang_code)
                self.rules[lang_code]['meta'][k] = v
            elif current_section == 'to_phonetics':  # rules for translating words to phonetics
                self._load_func(current_section, line, lang_code, line_number, rule_file)
            elif current_section.startswith('post'):
                self._load_func(current_section, line, lang_code, line_number, rule_file)
            elif current_section in available_sections:
                self._load_kv(line, current_section, line_number, lang_code)
            else:
                print('Invalid section name "{}" at line {} in rule file: {}'.format(current_section,
                                                                                     line_number, rule_file))
                exit(1)

    def _check_pre(self, pre_phonetic, pre_pattern):
        """
        Check whether pre_phonetic matches pre
        :param pre_phonetic: list e.g. ['AE', 'OW', 'O']
        :param pre_pattern: tuple  e.g. ('$', 'AE', '@',)
                                    head <--------- tail
        :return: True if matched, else False
        """
        assert isinstance(pre_phonetic, list)
        assert isinstance(pre_pattern, tuple) and len(pre_pattern) != 0
        l_all_consonants = self.current_rule['phonetics']['consonants']
        l_all_vowels = self.current_rule['phonetics']['vowels']
        list(pre_pattern).reverse()
        if not pre_phonetic:  # No phonetic in pre
            if pre_pattern == ('$',):
                return True
            else:
                return False
        index = len(pre_phonetic)
        for i in pre_pattern:
            if i == '@' and pre_phonetic[index] not in l_all_vowels:  # Any Vowels
                return False
            if i == '&' and pre_phonetic[index] not in l_all_consonants:  # Any Consonants
                return False
            if i == '$' and index != 1:
                return False
            if i not in ['@', '&', '$'] and i != pre_phonetic[index]:
                return False
            index -= 1
        return True

    def _check_post(self, post_phonetic, post_pattern):
        """
        Check whether pre_phonetic matches pre
        :param post_phonetic: list e.g. ['AE', 'OW', 'O']
        :param post_pattern: tuple  e.g. ('AE', '@', '$',)
                                    head <--------- tail
        :return: True if matched, else False
        """
        assert isinstance(post_phonetic, list)
        assert isinstance(post_pattern, tuple) and len(post_pattern) != 0
        l_all_consonants = self.current_rule['phonetics']['consonants']
        l_all_vowels = self.current_rule['phonetics']['vowels']
        if not post_phonetic:  # No phonetic in post
            if post_pattern == ('^',):
                return True
            else:
                return False
        index = 0
        for i in post_pattern:
            if i == '@' and post_phonetic[index] not in l_all_vowels:  # Any Vowels
                return False
            if i == '&' and post_phonetic[index] not in l_all_consonants:  # Any Consonants
                return False
            if i == '^' and index != (len(post_phonetic) - 1):
                return False
            if i not in ['@', '&', '^'] and i != post_phonetic[index]:
                return False
            index += 1
        return True

    def _match(self, phonetic, i_start, rule):
        """
        Match a longest pattern in rule at the i_start index of phonetic
        If matched, return >=1.
        Else, return 0.
        :param phonetic: a complete phonetic list of a word
        :param i_start: int: start index of the phonetic that need to match
        :param rule: self.rule[lang_code + category]
        :return: (<value of the matched rule>, <length of the matched pattern>)
        """
        assert isinstance(rule, dict)
        assert isinstance(i_start, int)
        assert isinstance(phonetic, list)

        matched_rule_value = 0
        pattern_len = 0  # value of the matched rule

        def _match_len(l_string, l_patterns):
            """
            len of matching pattern

            :param l_string:
            :param l_patterns:
            :return:
            """
            r = 0
            # if len(l_patterns) > len(l_string), no need for match
            if len(l_patterns) > len(l_string):
                return 0
            for i in range(len(l_patterns)):
                if l_string[i] == l_patterns[i]:
                    r += 1
                else:
                    return 0
            return r

        for k in rule.keys():  # k: ('<(pre)>', [<[match1]>, <[match2]>, ...], '<(post)>'), v: int
            pre = k[0]
            post = k[2]

            for patterns in k[1]:  # <match>: ['', '', ...]
                _pattern_len = _match_len(phonetic[i_start:], patterns)
                if _pattern_len != 0:  # matched, check <pre> and <post>
                    if len(k[0]) != 0 and not self._check_pre(phonetic[0:i_start], pre):  # Have <pre>
                        break  # invalid match, check next
                    if len(k[2]) != 0 and not self._check_post(phonetic[i_start + _pattern_len:], post):  # Have <post>
                        break  # invalid match, check next
                    # <pre> and <post> are both satisfied, compare the match length
                    if _pattern_len > pattern_len:
                        pattern_len = _pattern_len
                        matched_rule_value = rule[k]

        return matched_rule_value, pattern_len

    def _find(self, coord_c, coord_v, l_rule_t):
        """
        Find chinese transliteration at (coord_c, coord_v) of l_rule_t
        :param coord_c: coordinate of consonants row
        :param coord_v: coordinate of vowels column
        :param l_rule_t:
        :return: str
        """
        assert isinstance(coord_c, int)
        assert isinstance(coord_v, int)

        try:
            s_return = l_rule_t[(coord_c, coord_v)]
        except KeyError:
            raise NoRuleMatched('.transliteration')
        return s_return

    def _phonetics2chinese(self, l_phonetics, category):
        """
        Phonetic to chinese in the current_rule's category

        Match longest pattern in vowels column.
         If matched,
         find chinese at (1, coord_v).
         start index ++
         Else,
         Match longest pattern in consonants row.
         If matched,
         Excepted the matched consonant
         Match longest pattern in vowels column.
         find chinese at (coord_c, coord_v)

        self.current_rule Example:

        {'meta':
            {'language_name':''},
         'to_phonetic': func,
         'consonants people': [rule1, rule2, ...],
         'vowels people': [rule1, rule2, ...],
         'transliteration people': [rule1, rule2, ...],
         'consonants places': [rule1, rule2, ...],
         'vowels places': [rule1, rule2, ...],
         'transliteration places': [rule1, rule2, ...],
        }
        :param l_phonetics: list of phonetics -- e.g. ['AA', 'L', 'AE', 'X']
        :param category: 'places' | 'people'
        :return: str
        """
        assert category in ('people', 'places', )
        
        l_rule_c = self.current_rule['consonants ' + category]       # .consonants      section's rules
        l_rule_v = self.current_rule['vowels ' + category]           # .vowels          section's rules
        l_rule_t = self.current_rule['transliteration ' + category]  # .transliteration section's rules
        l_func_p = self.current_rule['post ' + category]             # .post            section's function

        s_return = ''
        i_start = 0
        while i_start != len(l_phonetics):
            coord_v, p_len = self._match(l_phonetics, i_start, l_rule_v)
            if coord_v:
                s_return += self._find(1, coord_v, l_rule_t)
                i_start += p_len
            else:
                coord_c, p_len = self._match(l_phonetics, i_start, l_rule_c)
                if coord_c:
                    i_start += p_len
                    # the consonant is the last phonetic of the word, no need to check vowels
                    if i_start == len(l_phonetics):
                        s_return += self._find(coord_c, 1, l_rule_t)
                        break
                    coord_v, p_len = self._match(l_phonetics, i_start, l_rule_v)
                    if coord_v:
                        s_return += self._find(coord_c, coord_v, l_rule_t)
                        i_start += p_len
                    else:
                        s_return += self._find(coord_c, 1, l_rule_t)
                else:
                    raise NoRuleMatched('.consonants', l_phonetics)
        return l_func_p(s_return)

    def _words2phonetics(self, func, keyword):
        """

        :param func: a translation function
        :param keyword: str e.g. 'Mike'
        :return: list

        r ----> [
                    ['AH', 'L', 'AE', 'X'], <------ gen by funcs(words[0])
                    ['G', 'R', 'E', 'N'],   <------ gen by funcs(words[1])
                    ...
                ]
        """
        return func(keyword)

    def translate(self, keyword, lang_codes):
        """
        Outer interface, translate words into chinese characters in selected cultures.
        :param keyword: a string that not contains spaces
        :param lang_codes: list: if empty, select all lang_codes.
        :return:
        """
        assert isinstance(keyword, str) and isinstance(lang_codes, list) and ' ' not in keyword

        keyword = keyword.capitalize()

        # Combine lang_codes
        _lang_codes = []
        if len(lang_codes) == 0:
            _lang_codes = self.rules.keys()
        else:
            for m in self.rules.keys():
                for n in lang_codes:
                    if m == n:
                        _lang_codes.append(m)
                        break

        results = {'transliterations': []}  # store results for every lang_code [<lang_code1>, <lang_code2>, ...]

        for _lang_code in _lang_codes:
            # Select rule for lang_code
            self.current_rule = self.rules[_lang_code]  # lang_code specified rule

            # to phonetics
            l_phonetics = self._words2phonetics(self.current_rule['to_phonetics'], keyword)

            results['transliterations'].append({
                'keyword': keyword,
                'language': self.current_rule['meta']['language_name'],
                'category': 'People',
                'chinese': self._phonetics2chinese(l_phonetics, 'people'),
            })
            results['transliterations'].append({
                'keyword': keyword,
                'language': self.current_rule['meta']['language_name'],
                'category': 'Places',
                'chinese': self._phonetics2chinese(l_phonetics, 'places'),
            })

        return results
