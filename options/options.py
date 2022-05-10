class Options:
    def __init__(self):
        # runtime related
        self.device = "cpu" # idk niet eens gebruikt lol
        self.random_state = 1

        # data related

        self.col_names = ['age', 'sex', 'marital_status', 'occupation', 'education', 'med_prep_by', 'medication',
                          'know_reason', 'know_dosage', 'familiar_timing',
                          'take_regurarly', 'know_med', 'forget_med', 'untroubled_after_dose', 'stop_med_feel_better',
                          'stop_med_feel_worse', 'other_med_if_side_effects', 'reduce_med_no_consult', 'break_from_med',
                          'to_many_med_stop_no_consult', 'no_med_morning', 'no_med_noon', 'no_med_evening',
                          'take_only_considered_important',
                          'weekly_med_forget', 'all_columns_together']

        self.notIntegerColumns = ['sex', 'marital_status', 'occupation', 'education', 'med_prep_by']

        self.feature_col = ['age', 'sex', 'marital_status', 'occupation', 'education', 'med_prep_by', 'medication',
                            'know_reason', 'know_dosage', 'familiar_timing',
                            'know_med', 'forget_med', 'untroubled_after_dose', 'stop_med_feel_better',
                            'stop_med_feel_worse', 'other_med_if_side_effects', 'reduce_med_no_consult',
                            'break_from_med',
                            'to_many_med_stop_no_consult', 'no_med_morning', 'no_med_noon', 'no_med_evening',
                            'take_only_considered_important',
                            'weekly_med_forget', 'all_columns_together']

        self.target_col = 'take_regurarly'
