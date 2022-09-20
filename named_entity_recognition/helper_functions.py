def tag_encoder(tags,tags_2enc):
    '''
    encoding tags using tag corpus(enc_2tags)
    '''

    encoded_tags = []

    for i in tags:
        t1 = []

        for j in i.split():

            t1.append(tags_2enc[str(j)])
        encoded_tags.append(t1)

    return encoded_tags
	
class str_maker :

  def __init__(self, data):
    self.data = data 

  def splitter(self):
    str_data = []
    for i in self.data:
      str_data.append(' '.join(i.split()))

    return str_data

def tag_encoder_2(tags,tags_2enc):
    '''
    encoding tags using tag corpus(enc_2tags)
    '''

    encoded_tags = []

    for i in tags:
        t1 = []

        for j in i:

            t1.append(tags_2enc[str(j)])
        encoded_tags.append(t1)

    return encoded_tags
