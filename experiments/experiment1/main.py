__author__ = 'acba'

import numpy
from regressorstock import RegressorStock
from regressors.regressortools import Error
from regressors.regressor import Regressor
from stock import Stock

from matplotlib import pyplot as plt


securities = open('ibovespa.txt', 'r').read().split('\n')
securities = securities[0:1]
#print(securities)
# regressors = ["elmk", "elmr", "svr"]
regressors = ["mlp"]
# scale = [None, "minmax", "standardization"]
scale = [None]
# transformation = [None, "pca", "kpca"]
transformation = [None]
# series_types = ["stock", "return"]
series_types = ["return"]
hor = 50

plt.figure(figsize=(24.0, 12.0))

for sec in securities:
    stock = Stock(name=sec, file_name="data/a_" + sec + ".csv")

    for series_type in series_types:
        print(series_type, "\n")

        pr_results = {}
        te_results = {}

        for reg in regressors:
            pr_results[reg] = {}
            te_results[reg] = {}
            for scaling in scale:
                pr_results[reg][scaling] = {}
                te_results[reg][scaling] = {}
                for transf in transformation:
                    print("Regressor: ", reg, " Scale: ", scaling,
                          " Transformation: ", transf)

                    regressor = Regressor(regressor_type=reg)
                    regressor = RegressorStock(regressor=regressor, stock=stock)

                    tr_result, te_result, pr_result = \
                        regressor.auto(series_type, hor, scale=scaling,
                                       transf=transf)

                    te_results[reg][scaling][transf] = te_result
                    pr_results[reg][scaling][transf] = pr_result

        ##### Plot #####

        if series_type is "stock":
            plt.plot(stock.time_series_data, 'ko', label='expected targets')
            predict_range = numpy.arange(stock.time_series_data.size - hor,
                                          stock.time_series_data.size)
        else:
            plt.plot(stock.returns, 'ko', label='expected targets')
            predict_range = numpy.arange(stock.returns.size - hor,
                                          stock.returns.size)
        
        for reg in regressors:
             for scaling in scale:
                 for transf in transformation:
        
             # #### Plot Testing ####
             #
             # if reg is "elmk":
             #     co = "bx"
             # elif reg is "elmr":
             #     co = "rx"
             # elif reg is "svr":
             #     co = "gx"
             # else:
             #     co = "cx"
             #
             # plt.plot(predict_range, te_results[reg].predicted_targets,
             #          co, label=reg + " test")
        
             #### Plot Predicting ####
        
                     if reg is "elmk":
                         co = "b"
                     elif reg is "elmr":
                         co = "r"
                     elif reg is "svr":
                         co = "g"
                     else:
                         co = "c"
        
                     if scaling is "minmax":
                         if transf is "pca":
                             co += "."
                         elif transf is "kpca":
                             co += ","
                         else:
                             co += "v"
        
                     elif scaling is "standardization":
                         if transf is "pca":
                             co += "<"
                         elif transf is "kpca":
                             co += ">"
                         else:
                             co += "^"
                     else:
                         if transf is "pca":
                             co += "s"
                         elif transf is "kpca":
                             co += "p"
                         else:
                             co += "o"
        
                     plt.plot(predict_range,
                              pr_results[reg][scaling][transf].predicted_targets,
                              co, label=reg + str(scaling) + str(transf))
        
                     # plt.plot(predict_range,
                     #          te_results[reg][scaling][transf].predicted_targets,
                     #          co, label=reg + str(scaling) + str(transf))
        
        plt.xlabel('Time (t)')
        plt.ylabel(series_type + " values")
        plt.title(sec + " " + series_type)
        plt.legend(loc='upper left', shadow=True)
        plt.show()

plt.close()