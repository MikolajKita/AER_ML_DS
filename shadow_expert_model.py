import numpy as np
from river import base, metrics, ensemble, dummy, tree, naive_bayes, forest
from river.drift.binary import HDDM_A, EDDM

class ModelFactory:
    """Factory to generate fresh River models based on a string identifier."""
    GLOBAL_SEED = 42

    @classmethod
    def createModel(cls, modelType):
        if modelType == 'HT':
            return tree.HoeffdingTreeClassifier()
        if modelType == 'ARF':
            return forest.ARFClassifier(seed=cls.GLOBAL_SEED)
        if modelType == 'SRP':
            return ensemble.SRPClassifier(seed=cls.GLOBAL_SEED)
        if modelType == 'NoChange':
            return dummy.NoChangeClassifier()
        if modelType == 'MajorityClass':
            return dummy.PriorClassifier()
        if modelType == 'NaiveBayes':
            return naive_bayes.GaussianNB()
        raise ValueError(f"Unknown model type: {modelType}")

class ShadowExpertModel(base.Classifier):
    """
    A wrapper that can use any model from ModelFactory.
    It replaces the internal model with a fresh instance whenever drift is detected.
    """
    def __init__(self, model_type='NaiveBayes', drift_detector=None):
        self.model_type = model_type
        # Initialize the first instance of the model
        self.model = ModelFactory.createModel(self.model_type)
        
        # Initialize drift detector (default to HDDM_A if none provided)
        self.drift_detector = (
            drift_detector.clone()
            if drift_detector is not None
            else EDDM(warm_start=50, alpha=0.8, beta=0.5)
        )
        
        self.drifts_detected = []
        self.instance_count = 0
        
        self.expert_counter = 0
        self.active_expert_label = f"Expert_{self.expert_counter}_{self.model.__class__.__name__}"
        
    def get_active_expert_label(self):
        return self.active_expert_label
        
    def predict_one(self, x, **kwargs):
        return self.model.predict_one(x, **kwargs)
        
    def predict_proba_one(self, x, **kwargs):
        return self.model.predict_proba_one(x, **kwargs)
        
    def learn_one(self, x, y):
        self.instance_count += 1
        
        # 1. Update the drift detector based on prediction error
        y_pred = self.model.predict_one(x)
        error = 0.0 if y_pred == y else 1.0
        self.drift_detector.update(error)
        
        # 2. Check for drift
        if self.drift_detector.drift_detected:
            self.drifts_detected.append(self.instance_count)
            
            # Use the Factory to replace the current model with a brand new one
            self.model = ModelFactory.createModel(self.model_type)
            
            self.expert_counter += 1
            self.active_expert_label = f"Expert_{self.expert_counter}_{self.model.__class__.__name__}"
            
            # Reset detector
            self.drift_detector = self.drift_detector.clone()
            
        # 3. Train the (potentially new) model
        self.model.learn_one(x, y)
        return self