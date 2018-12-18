
class SurDetection(object):
    def __init__(self, surname_path):
        self.START = '<s>'
        self.END = '</s>'
        self.surrounding = []
        with open(surname_path, 'r') as f:
            lines = f.readlines()
            self.surname_set = set()
            self.surname_set.update([l.split() for l in lines])

    def _match(self, word):
        for s in self.surname_set:
            if s in word:
                return True
        return False

    def _snt_split(self, snt, padding=True):
        snt_lst = snt.split()
        if padding:
            snt_lst.append(self.END)
            snt_lst.insert(0, self.START)
            return snt_lst
        else:
            return snt_lst

    def surname_extraction(self, data_path):
        with open(data_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                word_lst = self._snt_split(line)
                for i, word in enumerate(word_lst):
                    if self._match(word):
                        self.surrounding.append({word: word_lst[i-1: i+2]})











