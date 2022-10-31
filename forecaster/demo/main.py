from forecaster.Test_Models.EnsembleModel import EnsembleModel
# from Test_Models.ShiftingMethod import ShiftingMethod

from predictionService import run_one_month_exp

run_one_month_exp("C", "dec", 0.75, '13min', EnsembleModel(14, 10, '13min'), "cpu")

'''
Types of models to run
 - shifting_method ( naive method )
 - moving_average ( statistical )
 - linear_regression ( machine learning )
 - svm_polynomial ( machine learning )
 - svm_rbf_polynomial ( machine learning )
 - weekly_average ( statistical )
 - lstm ( Deep learning )
 - facebook_prophet ( machine learning )
'''

'''
According to accuracy
 - lstm
 - naive_method
 - moving_average
 - weekly_average
 - facebook_prophet
 - linear_regression
 - svm_polynomial
 - svm_rbf_polynomial
'''

