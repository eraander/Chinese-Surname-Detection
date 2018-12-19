import random
import re
import nltk

import tools

RANDOM_SEED = 501
random.seed(RANDOM_SEED)

# cluster tags into different groups
TAG_GROUP ={
	'NS': ['ns', 'n]ns'],
	'N': ['Ng', 'n', 'nr', 'nt', 'nx', 'nz', 'n]nt', 'j]nt', 'j]nz'],
	'V': ['v', 'Vg', 'vd', 'vn'],
	'ADJ': ['Ag', 'a', 'ad', 'an'],
	'ADV': ['Dg', 'd'],
	'IDIOMS': ['i', 'l'],
	'NUMBER': ['m', 'Qg', 'q', 'Mg'],
	'FUNCTIONS': ['b', 'Bg', 'c', 'e', 'f', 'g', 'h', 'j', 'k', 'o', 'p', 'Rg',
		'r', 's', 'Tg', 't', 'Ug', 'u', 'w', 'x', 'Yg', 'y', 'z'],
}

# keywords to be extracted from the sentences
ADM = ['省', '国', '区', '市', '县', '乡', '镇', '街','路','村','屯']
DIRECTION = ['东','南','西','北']
GEO = ['山','岭','江','河','海','川', '湾']
NAME_CHAR = ['华','州','城', '阳','江','安','平','宁','新','昌','丰']
VISIT = ['来', '前往', '访','去','于','地处']


def data_split(instances, test_split=0.20, random_split=True):
	'''
	split the data into training and test set
	Args:
		instances: list of Samples
	'''
	if random_split:
		random.shuffle(instances)

	total_count = len(instances)
	train_count = total_count - int(total_count * test_split)

	train_set = instances[:train_count]
	test_set = instances[train_count:]
	return train_set, test_set


def get_tag(raw_tag):
	'''
	group the tag w.r.t. TAG_GROUP
	Args:
		raw_tag: string of tag to be grouped
	'''
	for key in TAG_GROUP:
		if raw_tag in TAG_GROUP[key]:
			return key
	return 'ELSE'

def feature_extract(s, idx):
	'''
	extract feature for a given instance in the format of (sentence, index)
	Args:
		s: sentence
		idx: the index of the target word in the sentence
	'''
	features = {}

	features['prev_direction'] = 0
	features['post_direction'] = 0
	features['prev_tag'] = '<START>'
	features['post_tag'] = '<END>'
	features['post_adm'] = 0
	features['prev_visit'] = 0

	features['adm'] = 1 if any(element in str(s[idx]) for element in ADM) else 0
	features['direction'] = 1 if any(element in str(s[idx]) for element in DIRECTION) else 0
	features['geo'] = 1 if any(element in str(s[idx]) for element in GEO) else 0
	features['char'] = 1 if any(element in str(s[idx]) for element in NAME_CHAR) else 0

	if idx != 0:
		features['prev_tag'] = get_tag(s[idx-1][1])
		if any(element in str(s[idx-1][0]) for element in DIRECTION):
			features['prev_direction'] = 1
		if any(element in str(s[idx-1][0]) for element in VISIT):
			features['prev_visit'] = 1

	if idx != len(s) - 1:
		features['post_tag'] = get_tag(s[idx+1][1])
		if any(element in str(s[idx+1][0]) for element in DIRECTION):
			features['prev_direction'] = 1
		if any(element in str(s[idx+1][0]) for element in ADM):
			features['post_adm'] = 1

	return features


def find_idx_ns(s, target='ns'):
	'''
	find the word with the correct label (e.g., ns)
	Args:
		s: sentence
		target: target tag to be matched
	'''
	res = []
	for i in range(len(s)):
		if s[i][1] == target:
			res.append(i)
	return res


def get_DT(training_set, test_set):
	'''
	Build and evaluate a nltk decision tree based on a given training and test set
	Args:
		training_set: training set in format of [[features, label], ...]
		test_set: training set in format of [[features, label], ...]
	'''
	print('Training decision tree classifier... \n')

	classifier = nltk.DecisionTreeClassifier.train(training_set)
	for element in test_set:
		pred = classifier.classify(element[0])

	print('Rules extracted from the decision tree:')
	print(classifier.pseudocode())

	X_test = [element[0] for element in test_set]
	Y_test = [element[1] for element in test_set]
	Y_pred = [classifier.classify(element) for element in X_test]
	accuracy, tn, fp, fn, tp = tools.eval_res(Y_pred, Y_test)

	print('{} model has test accuracy: {}, tn: {}, fp: {}, fn: {}, tp: {}'
        .format('DT', accuracy, tn, fp, fn, tp))


def main():
	infile = open("taggedwords.txt", "r")
	sentences = (infile.read()).split("。/w")
	pos_instances = []
	neg_instances = []

	sentences = random.sample(sentences, 10000)
	for s in sentences:
		s = s.split()
		s = list(map(lambda x: x.split('/'), s))
		idxs = find_idx_ns(s)

		for i in range(len(s)):
			if i in idxs:
				pos_instances.append([s, i])
			else:
				neg_instances.append([s, i])

	neg_instances = random.sample(neg_instances, int(2.5 * len(pos_instances)))
	pos_instances = list(map(lambda x: [feature_extract(x[0], x[1]), 1], pos_instances))
	neg_instances = list(map(lambda x: [feature_extract(x[0], x[1]), 0], neg_instances))

	instances = pos_instances + neg_instances

	training_set, test_set = data_split(instances)

	get_DT(training_set, test_set)
	all_res = tools.eval_classifier(training_set, test_set)

	tools.plot_all_res(all_res)


if __name__ == "__main__":
	main()
