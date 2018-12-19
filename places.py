import random
import re
import nltk


def evaluate_classifier(classifier, test_set):
	test_accuracy = nltk.classify.accuracy(classifier, test_set)
	print('Test accuracy: {}'.format(test_accuracy))
	#classifier.show_most_informative_features(10)
	return

def data_split(instances,test_split=0.2, random_split=True):

	if random_split:
		random.shuffle(instances)

	total_count = len(instances)
	train_count = total_count - int(total_count * test_split)

	train_set = instances[:train_count]
	test_set = instances[train_count:]
	return train_set, test_set

def feature_extract(s, idx):
	features = {}

	if idx != 0:
		#features['prev_word'] = s[idx-1][0]
		features['prev_tag'] = s[idx-1][1]
	else:
		#features['prev_word'] = '<START>'
		features['prev_tag'] = '<START>'

	if idx != len(s) - 1:
		#features['post_word'] = s[idx+1][0]
		features['post_tag'] = s[idx+1][1]
	else:
		#features['post_word'] = '<END>'
		features['post_tag'] = '<END>'

	ss = str(s)
	adm_units = ['省', '国', '区', '市', '县', '乡', '镇','街','路','村','屯','湾', '海']

	features['adm_units'] = 0
	for u in adm_units:
		if u in ss:
			features['adm_units'] = 1
			break

	directions = ['东','南','西','北','以东','以南', '以西','以北']
	
	features['directions'] = 0
	for direct in directions:
		if direct in ss:
			features['directions'] = 1
			break

	#singal_words_general = ['来自', '前往', '到访', '出访','去','坐落于','位于','地处']
	#singal_words_geo = ['山','江','河','海','川']
	#popular_char_in_name = ['州','城', '阳','江','安','平','宁','新','昌','丰']
	return features

def find_idx_ns(s):
	res = []
	#print(s[0])
	for i in range(len(s)):
		#print('element {}'.format(s[i]))
		if s[i][1] == 'ns':
			res.append(i)
	return res


def main():
	infile = open("taggedwords.txt", "r")
	sentences = (infile.read()).split("。/w")
	pos_instances = []
	neg_instances = []

	for s in sentences[:2000]:
		#print('original {}'.format(s))
		s = s.split()
		s = list(map(lambda x: x.split('/'), s))
		idxs = find_idx_ns(s)

		for i in range(len(s)):
			if i in idxs:
				#features = feature_extract(s, i)
				pos_instances.append([s, i])
			else:
				neg_instances.append([s, i])

	neg_instances = random.sample(neg_instances, len(pos_instances))
	pos_instances = list(map(lambda x: [feature_extract(x[0], x[1]), 1], pos_instances))
	neg_instances = list(map(lambda x: [feature_extract(x[0], x[1]), 0], neg_instances))
	instances = pos_instances + neg_instances

	instance = map
	#print(instances)

	training_set, test_set = data_split(instances)
	classifier = nltk.DecisionTreeClassifier.train(training_set)
	for element in test_set:
		pred = classifier.classify(element[0])
		#print('pred is {} while actual is {} on the feature {}'.format(
		#	pred, element[1], element[0]))
		#if element[1]== 1:
		#	print(pred)
	evaluate_classifier(classifier, test_set)
	print(classifier.pretty_format())
