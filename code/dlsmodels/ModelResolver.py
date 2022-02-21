"""
ModelResolver

@author José Antonio García-Díaz <joseantonio.garcia8@um.es>
@author Rafael Valencia-Garcia <valencia@um.es>
"""

import json
import sys
import os.path

class ModelResolver ():
    """
    ModelResolver
    """
    
    def get_choices (self):
        """
        get_choices
        
        @return List
        """
        return [
            'transformers',
            'deep-learning',
            'ensemble'
        ]
        
        
    def get_default_choice (self):
        """
        get_default_choice
        
        @return String
        """
        return "deep-learning"
    
    def get_from_resume (self, resume):
        """
        @param self 
        @param resume 
        """
        
        model = self.get (resume.model if 'model' in resume else self.get_default_choice ())
        
        return model
        
        
    def get (self, model = 'machine-learning'):
        """
        @param model string
        """
        
        if 'transformers' == model:
            from .BertModel import BertModel
            return BertModel ()
            
        if 'deep-learning' == model:
            from .DeepLearningTechniques import DeepLearningTechniques
            return DeepLearningTechniques ()

        if 'ensemble' == model:
            from .EnsembleModel import EnsembleModel
            return EnsembleModel ()
            