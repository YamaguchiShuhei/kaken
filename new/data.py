import collections

class Dataset:
    def __init__(self,params):
        self.data = None
        self.label = None
        self.b_lattice = None
        self.m_lattice = None
        self.e_lattice = None
        self.s_lattice = None
        self.sentence_len = None
        self.char_id = None
        self.params = params

    def _read_path(self, filepath):
        data_list = [ ['<sta>'] ]
        for line in open(filepath, 'r'):
            line = line.strip().split()
            if line[0] == "EOS":
                data_list.append(line)
                data_list.append(["<sta>"])
            else:
                data_list.append(line)
        data_list.pop()
        return data_list
    
    def _character_id(self, data_list):
        char_list = [ data[0] for data in data_list ]
        counter = collections.Counter(char_list)
        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        count_pairs_highfreq = [ i for i in count_pairs if i[1] > self.params["FREQ_times"] ] 
        words, _ = list(zip(*count_pairs_highfreq))
        char_id = dict(zip(words, range(len(words))))
        return char_id
    
    # B=0 M=1 E=2 S=3 <sta>=4 EOS=5
    def _label_change(self, label):
        return {'b':0, 'm':1, 'e':2, 's':3}[label]
        # if label == 'b':
        #     return 0
        # if label == 'm':
        #     return 1
        # if label == 'e':
        #     return 2
        # if label == 's':
        #     return 3

    def _multi_append(self, active_list, passive_list):
        for i in range(len(active_list)):
            passive_list[i].append(active_list[i])
        
    def _sentence_list(self, data_list, char_id):
        sentence_list = []
        sentence_label_list = []
        sentence_b_lattice_list = []
        sentence_m_lattice_list = []
        sentence_e_lattice_list = []
        sentence_s_lattice_list = []
        sent_len_list = []
        sentence = []
        label = []
        b_lattice = []
        m_lattice = []
        e_lattice = []
        s_lattice = []
        for data in data_list:
            if data == ["EOS"]:
                # because resorce error
                # wiki_corpus include error
                if len(sentence) > 1000:
                    sentence = []
                    label = []
                    b_lattice = []
                    m_lattice = []
                    e_lattice = []
                    s_lattice = []
                else:
                    sent_len_list.append(len(sentence)+1)
                    sentence.append(char_id[data[0]])
                    label.append(5)
                    self._multi_append([0,0,0,0], [b_lattice,m_lattice,e_lattice,s_lattice])
                    sentence_list.append(sentence)
                    sentence_label_list.append(label)
                    self._multi_append([b_lattice, m_lattice, e_lattice, s_lattice], [sentence_b_lattice_list, sentence_m_lattice_list, sentence_e_lattice_list, sentence_s_lattice_list])
                    sentence = []
                    label = []
                    b_lattice = []
                    m_lattice = []
                    e_lattice = []
                    s_lattice = []
            elif data == ["<sta>"]:
                sentence.append(char_id[data[0]])
                label.append(4)
                self._multi_append([0,0,0,0], [b_lattice, m_lattice, e_lattice, s_lattice])
            else:
                if data[0] in char_id:
                    sentence.append(char_id[data[0]])
                else:
                    sentence.append(len(char_id))
                label.append(self._label_change(data[1]))
                self._multi_append(list(map(int,data[2:6])), [b_lattice, m_lattice, e_lattice, s_lattice])
        return sentence_list, sentence_label_list, sentence_b_lattice_list, sentence_m_lattice_list, sentence_e_lattice_list, sentence_s_lattice_list, sent_len_list
    
    def read(self, data_path, char_id=None):
        self.data_list = self._read_path(data_path)
        if char_id != None:
            self.char_id = char_id
        else:
            self.char_id = self._character_id(self.data_list)
        self.data, self.label, self.b_lattice, self.m_lattice, self.e_lattice, self.s_lattice, self.sentence_len = self._sentence_list(self.data_list, self.char_id)
