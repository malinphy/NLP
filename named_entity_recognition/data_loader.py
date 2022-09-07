import re
def load_sentences(filepath):

    final = []
    sentences = []

    with open(filepath, 'r') as f:
        
        for line in f.readlines():
            
            if (line == ('-DOCSTART- -X- -X- O\n') or line == '\n'):
                if len(sentences) > 0:
                    final.append(sentences)
                    sentences = []
            else:
                l = line.split(' ')
                sentences.append((l[0], l[3].strip('\n')))
    
    return final


def list_maker(data_set,col):
    v2 = []

    for i in data_set:
        v1 = []

        for j in i:
            v1.append(j[col])
        x = ' '.join(v1)
        x = re.sub('  ',' ',x)
        x = re.sub(r"\s+$", "", x, flags=re.UNICODE)
        x = re.sub("^\s+|\s+$", "", x, flags=re.UNICODE)
        v2.append(x)

        # v2.append(' '.join(v1))

    return v2
