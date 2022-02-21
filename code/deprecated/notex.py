    def getDFFromTask (self, task, df):
        
        # Merge tweets
        if 'merge' in self.options['tasks'][task] and self.options['tasks'][task]['merge']:
            
            # @var columns_to_keep List
            columns_to_keep = ['user', 'tweet', 'label', '__split']
            
            
            # As we need to merge all the tweets of the same author, we need to merge 
            # again training and development datasets
            df.loc[df['__split'] == 1, '__split'] = 0
            
            
            # Merge
            df['tweet'] = df.groupby (['user', '__split'])['tweet'].transform (lambda x: '. '.join (x))
            
            
            # Remove duplicates
            df = df[columns_to_keep].drop_duplicates ().reset_index (drop = True)
            
            
            # Back to split training into training and evaluation
            df.loc[df.loc[df['__split'] == 0].sample (frac = 0.20, replace = False).index, '__split'] = 1
    
        # Adjust the label
        return df
