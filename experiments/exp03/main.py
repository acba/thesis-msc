import numpy as np
import pickle
import os
import datetime
from tools import *

try:
    import mkl
    mkl.set_num_threads(8)
except ImportError:
    print("MKL not found")

import pai

assets = open('ibovespa.txt', 'r').read().split('\n')
results_path = "results/"
search_function = "particle swarm"
regressors = ["elmk", "elmr", "svr", "mean", "mlp"]
eval = 80


def run():

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    for asset in assets:

        filename = results_path + asset + ".p"

        if check(filename):
            try:
                stock = get_data(asset)
                stock.create_database(4, series_type="return")
                data = stock.get_database()

                if os.path.exists(filename):
                    result = pickle.load(open(filename, 'rb'))
                else:
                    result = {"finished": False}

                for reg in regressors:
                    r = pai.Regressor(reg)

                    # print(type(result), result)
                    if reg not in result:
                        result[reg] = {}

                        result[reg]["finished"] = False
                        print(asset, reg)

                        metrics = []
                        time = []
                        for it in range(50):
                            start = datetime.datetime.now()
                            r.search_param(database=data, cv="ts",
                                           cv_nfolds=10, of="rmse",
                                           opt_f=search_function,
                                           eval=eval, print_log=False,
                                           kf=["rbf"], f=["sigmoid"])
                            end = datetime.datetime.now()
                            delta = end - start

                            time.append(delta.total_seconds())
                            metrics.append(r.regressor.cv_best_error)

                            if reg == "mlp":
                                metrics = (np.ones(50) * r.regressor.cv_best_error).tolist()
                                break

                        result[reg]["cv"] = metrics
                        result[reg]["cv_mean"] = np.mean(metrics)
                        result[reg]["cv_std"] = np.std(metrics)

                        result[reg]["time"] = time
                        result[reg]["time_mean"] = np.mean(time)
                        result[reg]["time_std"] = np.std(time)

                        result[reg]["finished"] = True

                        print("***********************")

                        pickle.dump(result, open(filename, "wb"))
                    else:
                        print(reg)
                        print(result[reg]["cv_mean"], result[reg]["cv_std"])
                        print(result[reg]["time_mean"], result[reg]["time_std"])
                        print()

                cvs = [result[reg]["cv"] for reg in regressors]
                ts = [result[reg]["time"] for reg in regressors]

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


if __name__ == "__main__":

    print()
    print("Experiment 03")
    print()

    run()
