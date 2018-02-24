### Algorithms taken from https://github.com/cognoa/cognoa/tree/develop/app/calculators,
### Converted from ruby

from base_calculator import CognoaCalculator
import numpy as np


class Cog2Calculator(CognoaCalculator):
    def __init__(self):
        pass

    def is_high_risk(self, risk_level):
        return 1 if risk_level == 'high_risk' else 0

    def get_risk_level(self, score):
        if score >= 0.6:
            return 'high_risk'
        elif score >= 0.3:
            return 'medium_risk'
        else:
            return 'low_risk'

    def clean_column(self, value):
        if value in [7, 8]:
            return 0
        elif value in [0, 1, 2, 3, 4]:
            return value
        raise ValueError('unexpected score ' + str(value))

    def compute_raw_scores_on_df(self, input_df):
        questions = ['ados2_a5', 'ados2_a8', 'ados2_b1', 'ados2_b3', 'ados2_b6', 'ados2_b8', 'ados2_b10', 'ados2_d2',
                     'ados2_d4']
        df_for_calculation = input_df[questions]
        ### Clean it
        for question in questions:
            df_for_calculation[question] = df_for_calculation[question].apply(self.clean_column)
        log_odds = (-15.8657 + 2.2539 * input_df['ados2_a5'] + 3.0323 * input_df['ados2_a8'] + \
                    3.8820 * input_df['ados2_b1'] + 4.3625 * input_df['ados2_c4'] + \
                    5.0750 * input_df['ados2_b8'] + 4.0215 * input_df['ados2_b8'] + \
                    3.8299 * input_df['ados2_b9'] + 3.4053 * input_df['ados2_d2'] + 2.6616 * input_df['ados2_c3'])
        probability_of_diagnosis = 1. / (1. + np.exp(-1 * log_odds))
        input_df['cog2_response'] = probability_of_diagnosis
        input_df['risk_level'] = input_df['cog2_response'].apply(self.get_risk_level)
        input_df['is_high_risk'] = input_df['risk_level'].apply(self.is_high_risk)
        return input_df
