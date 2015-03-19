import numpy as np
import pickle
import os
import datetime
from tools import *


import pai
from pai.portfolio import get_data


assets = open('ibovespa.txt', 'r').read().split('\n')
results_path = "results/"
search_functions = ["random search", "grid search", "particle swarm",
                    "nelder-mead", "cma-es"]
# search_functions = ["random search", "grid search", "nelder-mead"]

if not os.path.exists(results_path):
    os.makedirs(results_path)

r = pai.Regressor("elmk")

result = {}

for i, asset in enumerate(assets):

    filename = results_path + asset + ".p"

    if check(filename):
        try:
            stock = get_data(asset)
            stock.create_database(4, series_type="return")
            data = stock.get_database()

            result = {"finished": False}

            for f in search_functions:
                result[f] = {}

                print(asset, f)

                metrics = []
                time = []
                for it in range(50):
                    start = datetime.datetime.now()
                    r.search_param(database=data, cv="ts", cv_nfolds=10,
                                   of="rmse", opt_f=f, eval=50,
                                   print_log=False,
                                   kf=["rbf"])
                    end = datetime.datetime.now()
                    delta = end - start

                    time.append(delta.total_seconds())
                    metrics.append(r.regressor.cv_best_error)

                result[f]["cv"] = metrics
                result[f]["cv_mean"] = np.mean(metrics)
                result[f]["cv_std"] = np.std(metrics)

                result[f]["time"] = time
                result[f]["time_mean"] = np.mean(time)
                result[f]["time_std"] = np.std(time)

            cvs = [result[f]["cv"] for f in search_functions]
            ts = [result[f]["time"] for f in search_functions]

            # Create paired test matrices only if exists differences among
            # sample distributions
            if not non_parametric_check_samples(cvs):
                cv_nparam = matrix_non_parametric_paired_test(cvs)
                result["cv_nparam"] = cv_nparam

            if not non_parametric_check_samples(ts):
                t_nparam = matrix_non_parametric_paired_test(ts)
                result["t_nparam"] = t_nparam

            result["finished"] = True
            pickle.dump(result, open(filename, "wb"))

            import pprint
            pprint.pprint(result)

        except:
            print("Error: ", asset)


