
from abc import ABCMeta, abstractmethod

class CognoaCalculator(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def is_high_risk(self, risk_level):
        pass

    @abstractmethod
    def get_risk_level(self, row):
        pass

    @abstractmethod
    def compute_raw_scores_on_df(self, input_df):
        pass
