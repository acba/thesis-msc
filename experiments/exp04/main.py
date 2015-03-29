import numpy as np
import pickle
import os
import datetime
from tools import *
from elm import mltools

# try:
#     import mkl
#     mkl.set_num_threads(8)
# except ImportError:
#     print("MKL not found")

import pai

assets = open('ibovespa.txt', 'r').read().split('\n')
results_path = "results/"
search_function = "particle swarm"
eval = 100
nfolds = [3, 5, 10, 15, 30]

def run(name):

    r = pai.Regressor(name)

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    for i, asset in enumerate(assets):

        filename = results_path + name + "_" + asset + ".p"

        if check(filename):
            try:
                stock = get_data(asset)
                stock.create_database(4, series_type="return")
                data = stock.get_database()

                result = {"finished": False}

                for nf in nfolds:
                    result[nf] = {}

                    print(asset, nf)

                    metrics = []
                    time = []
                    for it in range(50):
                        start = datetime.datetime.now()

                        # Search for best hyper parameters
                        params = r.search_param(database=data, cv="ts",
                                                cv_nfolds=nf, of="rmse",
                                                opt_f=search_function,
                                                eval=eval, print_log=False,
                                                kf=["rbf"], f=["sigmoid"])

                        tr, te = mltools.split_sets(data, training_percent=.8)

                        # Train with 80% of dataset using parameters found
                        tr_result = r.train(tr, params)
                        te_result = r.test(te)
                        print(te_result.get("rmse"))

                        end = datetime.datetime.now()
                        delta = end - start

                        time.append(delta.total_seconds())
                        metrics.append(te_result.get("rmse"))

                    result[nf]["test"] = metrics
                    result[nf]["test_mean"] = np.mean(metrics)
                    result[nf]["test_std"] = np.std(metrics)

                    result[nf]["time"] = time
                    result[nf]["time_mean"] = np.mean(time)
                    result[nf]["time_std"] = np.std(time)

                tests = [result[nf]["test"] for nf in nfolds]
                ts = [result[nf]["time"] for nf in nfolds]

                # Create paired test matrices only if exists differences among
                # sample distributions
                if not non_parametric_check_samples(tests):
                    test_nparam = matrix_non_parametric_paired_test(tests)
                    result["test_nparam"] = test_nparam

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
    print("Experiment 04")
    print()

    print()
    print("Run ELMK")
    print()
    run("elmk")

    print()
    print("Run ELMR")
    print()
    run("elmr")

    # print()
    # print("Run SVR")
    # print()
    # run("svr")
