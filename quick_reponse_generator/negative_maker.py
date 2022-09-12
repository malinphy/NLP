import pandas as pd 
import numpy as np 



class negative_maker():
    def __init__(self,df):
        self.df = df 
    def neg_maker(self):
        neg_pos = []
        neg_title = []
        neg_answer = []
        for i in range(len(self.df)):
            x = np.random.randint(0, len(self.df))
            neg_pos.append(x)
            neg_title.append(self.df['title'][x])
            neg_answer.append(self.df['first_answer'][x])

        self.df['neg_title'] = neg_title
        self.df['neg_answer'] = neg_answer

        return self.df