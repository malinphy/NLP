#!pip install --quiet datasets
import pandas as pd 
import datasets




class data_loader():
    def __init__(self,partition):
        self.partition = partition
        
    def frame_maker(self):
        # 'train_eli5'
        eli5 = datasets.load_dataset('eli5', split = self.partition)
        df = pd.DataFrame({'title':eli5['title'], 'selftext':eli5['selftext'], 'answer':eli5['answers']})

        answer_len = []
        first_answer = []
        for i in range(len(df)):
            answer_len.append(len(df['answer'][i]['text']))
            first_answer.append(df['answer'][i]['text'][0])

        df['first_answer'] = first_answer

        unique_answer = df['first_answer'].unique()
        num_unique_answer = len(unique_answer)

        unique_questions = df['title'].unique()
        num_unique_questions =  len(unique_questions)

        return df