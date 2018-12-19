
import os, re, nltk

translit_characters = '布普德特格克夫弗兹茨斯丝什奇赫姆恩尔伊古库胡阿巴帕达塔加卡瓦娃法扎察萨莎贾查哈马玛纳娜拉亚娅瓜夸埃贝佩泰盖凯韦费泽策塞热' \
                      '谢杰切黑梅内莱雷蕾耶圭奎惠厄伯珀沃瑟舍哲彻赫默娜纳勒果阔霍伊比皮迪蒂吉基维威菲齐西希奇米尼妮利莉里丽伊圭奎惠欧奥博波多托戈科沃' \
                      '福佐措索若肖乔莫诺洛罗萝约乌杜图古库武伍富祖楚苏茹舒朱穆努卢鲁尤久丘休缪纽柳留艾拜派代戴怀宰蔡赛夏柴海迈奈赖鲍保道陶高考藻曹绍' \
                      '焦豪毛瑙劳尧安班潘丹坦甘坎万凡赞灿桑詹钱汉曼兰关宽环昂邦庞当唐冈康旺方仓让尚章昌杭芒南朗扬光匡黄本彭登滕根肯文芬曾岑森任申真' \
                      '琴亨门嫩伦延昆因英宾平丁廷金京温津欣青辛兴钦明宁林琳翁宏蓬顿敦东通贡孔丰尊宗聪孙松容顺雄准春琼洪蒙农云隆龙律大施卜'
high_incidence_foreign_chars = '罗埃菲莱巴克'
deceptive_chars = '藏日华'

def readposfile(file):
    '''reads in a Chinese pos_tagged file and places each sentence into a list'''
    f = open(file)
    sents_with_proper_noun = []
    for line in f.readlines():
        if '_NR' in line:
            sents_with_proper_noun.append(line)
    return sents_with_proper_noun

def split_tagged_sent(sentence):
    '''splits a tagged sentence on space'''
    split = sentence.split()
    return split

def make_labeled_data(sentences):
    '''labels the tagged data based on the regex
    l_foreign = likely foreign = NR identified by the regex (and doesn't end in 京)
    l_Chinese = likely Chinese = NR not found by the regex
    Chinese = all other parts of speech
    '''
    labeled_data = []
    char_regex = make_regex()
    for sentence in sentences:
        tagged_sent = []
        split = split_tagged_sent(sentence)
        for item in split:
            pair = tuple(item.split("_"))
            # labeled_pair = tuple()
            if pair[1] == 'NR' and not pair[0].endswith('京') and re.match(char_regex, pair[0]):
                labeled_pair = (pair, 'l_foreign')
            elif pair[1] == 'NR':
                labeled_pair = (pair, 'l_Chinese')
            else:
                labeled_pair = (pair, 'Chinese')
            tagged_sent.append(labeled_pair)
        labeled_data.extend(tagged_sent)
    return labeled_data

def make_regex():
    '''method that compiles the regex to be used to identify likely foreign proper names'''
    char_regex = r'(([%s]+|[A-Z])([‧·]([%s]+|[A-Z]))*\b)|\b.*[%s].*\b|[A-Za-z]+' % \
                 (translit_characters, translit_characters, high_incidence_foreign_chars)
    return re.compile(char_regex)

def make_proper_noun_labels(labeled_data):
    '''method that throws out non-proper names (which would hurt the classifier if left in)'''
    return [l for l in labeled_data if l[1] is not 'Chinese']

def create_feature_sets(labeled_data):
    '''method that builds the training and test sets for the data'''
    feature_sets = [(char_features(word), label) for (word, label) in labeled_data]
    training_data = feature_sets[100:]
    test_data = feature_sets[:100]
    return training_data, test_data


def char_features(wordpair):
    '''makes a set of features based on whether the NR (proper noun)
    has one of the characters found in translit_characters or not
    '''
    features = {}
    # print(wordpair)
    for t in translit_characters:
        features[t] = (t in wordpair[0])
    features['has_NR'] = wordpair[1].startswith('NR')
    return features

def train_classifier(training_data):
    '''trains the classifier with a set of training data'''
    classifier = nltk.NaiveBayesClassifier.train(training_data)
    classifier.show_most_informative_features(50)
    return classifier

def evaluate_classifier(classifier, test_data):
    '''tests the accuracy of the classifier on the test data
    it also prints a confusion matrix
    '''
    gold_st = [t[1] for t in test_data]
    test_st = [classifier.classify(t[0]) for t in test_data]
    cm = nltk.ConfusionMatrix(gold_st, test_st)
    print(cm.pretty_format(sort_by_count=True, show_percents=True, truncate=9))
    print(nltk.classify.accuracy(classifier, test_data))

def create_evaluate_data(file):
    ''' method that assigns values of l_foreign or l_Chinese to each NR based on my
    identification of each NR'''
    sentences = []
    evaluate_data = []
    for line in file.readlines():
        if '_NR' in line:
            sentences.append(line)
    for sent in sentences:
        tagged_sent = []
        split = split_tagged_sent(sent)
        # print(split)
        for word in split:
            pair = tuple(word.split("_"))
            # print(pair)
            # labeled_pair = tuple()
            if pair[1] == 'NRfor':
                labeled_pair = (pair, 'l_foreign')
            elif pair[1] == 'NRnat':
                labeled_pair = (pair, 'l_Chinese')
            else:
                labeled_pair = (pair, 'Chinese')
            tagged_sent.append(labeled_pair)
        evaluate_data.extend(tagged_sent)
    return evaluate_data




def run_classifier(classifier, file):
    '''runs the classifier on the tagged data found in the file,
    where I tagged every NR (proper noun) as foreign or native (Sinitic)
    I tagged as follows for chtb_4114.bc.pos:
    transliterated = foreign
    uses native Chinese name (even Japanese/Korean names) = Sinitic
    '''
    folder = 'ctb9.0/data/postagged'
    f = open(file, 'r')
    tagged_sentences = create_evaluate_data(f)
    proper_noun_data = make_proper_noun_labels(tagged_sentences)
    feature_sets = [(char_features(word), label) for (word, label) in proper_noun_data]
    # classification = [(instance, classifier.classify())]
    gold_st = [t[1] for t in feature_sets]
    test_st = [classifier.classify(t[0]) for t in feature_sets]
    cm = nltk.ConfusionMatrix(gold_st, test_st)
    print(cm.pretty_format(sort_by_count=True, show_percents=True, truncate=9))
    print(nltk.classify.accuracy(classifier, feature_sets))


def main():
    ''' main method '''
    labeled_sents = []
    folder = 'ctb9.0/data/postagged'
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        if path.endswith('bn.pos'):
            print(file)
            sents = readposfile(path)
            labeled_sents.extend(make_labeled_data(sents))
    for l in labeled_sents:
        if '罗' in l[0]:
            print(l)
    proper_noun_data = make_proper_noun_labels(labeled_sents)
    training_data, test_data = create_feature_sets(proper_noun_data)
    classifier = train_classifier(training_data)
    evaluate_classifier(classifier, test_data)
    # f = open('ctb9.0/data/postagged/chtb_4114.bc.pos', 'r')
    #tagged_sentences = create_evaluate_data(f)

    file1 = 'ctb9.0/data/postagged/chtb_4112.bc.pos'
    file2 = 'ctb9.0/data/postagged/chtb_4113.bc.pos'
    file3 = 'ctb9.0/data/postagged/chtb_4114.bc.pos'
    run_classifier(classifier, file1)
    run_classifier(classifier, file2)
    run_classifier(classifier, file3)










