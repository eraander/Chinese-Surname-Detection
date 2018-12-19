
class SurDetection(object):
    def __init__(self, surname_path, title_path):
        """

        :param surname_path:
        :param title_path:
        """
        self.START = '<s>'
        self.END = '</s>'
        self.surname = set()
        self.surrounding = []
        with open(surname_path, 'r') as f1, open(title_path, 'r') as f2:
            lines = f1.readlines()
            titles = f2.readlines()
            self.surname_set = set()
            self.title_set =set()
            self.surname_set.update(set([l.strip() for l in lines]))
            self.title_set.update(set([l.strip() for l in titles]))

    def _match(self, word, word_set):
        """
        take every character in in word_set to check if it is in the word
        return true if the character in word, return false if the character is not in the word
        :param word:
        :param word_set:
        :return:
        """
        for s in word_set:
            if s in word:
                return True
        return False

    def _snt_split(self, snt, padding=True):
        """

        :param snt:
        :param padding:
        :return:
        """
        snt_lst = snt.split()
        if padding:
            snt_lst.append(self.END)
            snt_lst.insert(0, self.START)
            return snt_lst
        else:
            return snt_lst

    def surrounding_extraction(self, data_path):
        with open(data_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                word_lst = self._snt_split(line)
                for i, word in enumerate(word_lst):
                    if self._match(word, self.surname_set):
                        self.surrounding.append((word, [word_lst[i-1], word_lst[i+1]]))

    def surname_extraction(self):

        for s in self.surrounding:
            surrs = s[1]
            if self._match(surrs, self.title_set):
                self.surname.add(s[0])



def main():
    sd = SurDetection('data/dc_surname', 'data/vocab.title')
    sd.surrounding_extraction('data/train.snt')
    sd.surname_extraction()
    print(sd.surname)
