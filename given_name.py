import namesex
import nltk
import numpy as np
import ast
import surname_detection
def create_labeled_data():
	"""
	This method open training data file and convert str to list of tuple, then get feature of every name,
	and create the labeled_data.
	'write_file.txt' is a file that contains names and sex of that name.
	This method returns the labeled_date. 
	"""
	f = open('write_file.txt', 'r')
	data = f.read()
	data.split()
	data = list(ast.literal_eval(data))
	labeled_data = [(wsd_features(instance[0]),instance[1]) for instance in data]
	# print(len(labeled_data))
	# print(type(labeled_data))
	# print(labeled_data[:8])
	return labeled_data

def create_feature_sets(labeled_data):
	"""
	This method creates feature sets.

	"""
	threshold = 0.9
	test_set = labeled_data[round(len(labeled_data) * threshold):]
	train_set = labeled_data[:round(len(labeled_data) * threshold)]
	return train_set, test_set

def wsd_features(instance):
	"""
	extract features.

	"""
	if len(instance) == 1:
		return {'pre': instance[0]}
	else:
	    return {'pre':instance[0],'pos':instance[1]}
	
def train_classifier(training_set):
	"""
	# create the classifier.
	"""
	classifier = nltk.NaiveBayesClassifier.train(training_set)
	return classifier

def evaluate_classifier(classifier, test_set):
	print("Accuracy: ",nltk.classify.accuracy(classifier,test_set))

def name_quality1(name):
	"""
	Take name as parameter (After running the classifier and determinate the name is a male name. 
	There are strings that contains chinese character of different attribute.
	If the name contains two character of a certain attribute, then it duplicates, which will return not a good name. otherwise, will return " a good name."
	"""
	print("This is a male name.")
	handsome = "帅才双玉树临风温文尔雅淑人君子清新俊逸品貌非凡才貌双绝惊才风逸风流才子雅人深致城北徐公堂堂正正七尺男儿英俊潇洒顶天立地男足智多谋风流正义玉树临风表高大威猛英俊潇洒风倜傥度翩气宇酷无情温文淑清逸非凡逸流雅薄云天铁骨铮"
	wealth = "珠光宝气珠围翠绕金玉满堂 荣华富贵 富贵荣华 朱门绣户 锦衣玉食 侯服玉食 纸醉金迷 朱轮华毂 一掷千金 堆金积玉"
	health = "红光满面焕发饱满振奋十足炯一身正气身强龙精虎猛长乐永康身强壮龙精寿比福如东海虎体熊腰虎背熊腰膘肥体生龙心气壮如龙神马壮益壮人高马大虎背熊腰精神抖擞红光龙钟寿年丰足常奋进取乐自得其乐"
	smart = "聪明伶俐点头会意见经识经精明能千伶百俐手急眼快别具慧眼百伶百俐辨日炎凉冰雪绝世聪明正直大巧若拙慧心妙舌巧思绝顶聪明精明能干精明强干绝圣弃智敬谢不敏谨谢不敏锦心绣肠口齿伶俐兰质蕙心目达耳通敏学偶变投隙巧妇难为七行俱下千虑一失七窍玲珑识时务者为俊杰时势造英雄上智下愚投机取巧剔透万物之灵小时了了秀外慧中小黠大痴左手画方颖悟绝伦绝人予智予雄抓乖卖俏抓乖弄俏自作聪明足智谋"
	personality = "清白公正凛然正直无私刚正克己奉公负重奉公忠心忠贞谦虚谨慎廉洁毅然决然豁达大度乐观坦白舍己勤奋刻苦认真专注钻研踏实勤恳虚心好学高尚德厚厚德蕙心德劭志士仁人杰出超伦自爱自尊自强自谦德高宽容宽宏律己助人好施仗义助人雄心壮志光明磊落"
	if len(name) == 1:
		print(name, "is a good name.")
	elif name[0] in handsome and name[1] in handsome:
		print(name, "is not a good name, because both character has same meaning.") 
	elif name[0] in wealth and name[1] in wealth:
		print(name, "is not a good name, because both character has same meaning.")
	elif name[0] in health and name[1] in health:
		print(name, "is not a good name, because both character has same meaning.")
	elif name[0] in smart and name[1] in smart:
		print(name, "is not a good name, because both character has same meaning.")
	elif name[0] in personality and name[1] in personality:
		print(name, "is not a good name, because both character has same meaning.")
	else:
		print(name, "It is a good name!")


def name_quality0(name):
	"""
	Take name as parameter (After running the classifier and determinate the name is a femal name. 
	There are strings that contains chinese character of different attribute.
	If the name contains two character of a certain attribute, then it duplicates, which will return not a good name. otherwise, will return " a good name."
	"""
	print(name, "This is a female name.")
	beauty = "贤惠颖德慧智雅静梦洁惠茜桑榆畅淑姝娈玲嫣婧"
	wealth = "珠光宝气 珠围翠绕 金玉满堂 荣华富贵 富贵荣华 朱门绣户 锦衣玉食 侯服玉食 纸醉金迷 朱轮华毂 一掷千金 堆金积玉"
	health = "红光满面焕发饱满振奋十足炯一身正气身强龙精虎猛长乐永康身强壮龙精寿比福如东海虎体熊腰虎背熊腰膘肥体生龙心气壮如龙神马壮益壮人高马大虎背熊腰精神抖擞红光龙钟寿年丰足常奋进取乐自得其乐"
	smart = "聪明伶俐点头会意见经识经精明能千伶百俐手急眼快别具慧眼百伶百俐辨日炎凉冰雪绝世聪明正直大巧若拙慧心妙舌巧思绝顶聪明精明能干精明强干绝圣弃智敬谢不敏谨谢不敏锦心绣肠口齿伶俐兰质蕙心目达耳通敏学偶变投隙巧妇难为七行俱下千虑一失七窍玲珑识时务者为俊杰时势造英雄上智下愚投机取巧剔透万物之灵小时了了秀外慧中小黠大痴左手画方颖悟绝伦绝人予智予雄抓乖卖俏抓乖弄俏自作聪明足智谋"
	personality = "清白公正凛然正直无私刚正克己奉公负重奉公忠心忠贞谦虚谨慎廉洁毅然决然豁达大度乐观坦白舍己勤奋刻苦认真专注钻研踏实勤恳虚心好学高尚德厚厚德蕙心德劭志士仁人杰出超伦自爱自尊自强自谦德高宽容宽宏律己助人好施仗义助人雄心壮志光明磊落"
	if len(name) == 1:
		print(name, "is a good name")
	elif name[0] in beauty and name[1] in beauty:
		print(name, "is not a good name, because both character has same meaning.") 
	elif name[0] in wealth and name[1] in wealth:
		print(name, "is not a good name, because both character has same meaning.")
	elif name[0] in health and name[1] in health:
		print(name, "is not a good name, because both character has same meaning.")
	elif name[0] in smart and name[1] in smart:
		print(name, "is not a good name, because both character has same meaning.")
	else:
		print(name, "It is a good name!")

def main(surname):
	labeled_data = create_labeled_data()
	# print(labeled_data)
	training_set, test_set = create_feature_sets(labeled_data)
	classifier = train_classifier(training_set)
	# name can be changed to test.
	for name in surname:
	#running classsifier to see the sex of the name.
		print(classifier.classify(wsd_features(name)))
	#base on the result of the name, test whether the name is good or not
		if classifier.classify(wsd_features(name))==0:
			name_quality0(name)
		else:
			name_quality1(name)
	#evaluate the accuracy of the classifirer.
	evaluate_classifier(classifier, test_set)


