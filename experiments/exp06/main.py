import numpy as np
import pickle
import os
import datetime
from tools import *
from elm import mltools

import pai

assets = open('ibovespa.txt', 'r').read().split('\n')
results_path = "results/"
search_function = "particle swarm"
eval = 100
cv = "ets"
nfolds = [2, 5, 8, 10, 15, 30]


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

                if os.path.exists(filename):
                   result = pickle.load(open(filename, 'rb'))
                else:
                   result = {"finished": False}

                for nf in nfolds:
                    if nf not in result:
                        result[nf] = {}
                        result[nf]["finished"] = False

                        print(asset, nf)

                        tests_results = []
                        cv_results = []
                        time_results = []
                        for it in range(50):
                            start = datetime.datetime.now()

                            # Search for best hyper parameters
                            params = r.search_param(database=data, cv=cv,
                                                    cv_nfolds=nf, of="rmse",
                                                    opt_f=search_function,
                                                    eval=eval, print_log=False,
                                                    kf=["rbf"], f=["sigmoid"])

                            cv_results.append(r.regressor.cv_best_error)
                            tr, te = mltools.split_sets(data, training_percent=.8)

                            # Train with 80% of dataset using parameters found
                            tr_result = r.train(tr, params)
                            te_result = r.test(te)
                            print(te_result.get("rmse"))

                            end = datetime.datetime.now()
                            delta = end - start

                            time_results.append(delta.total_seconds())
                            tests_results.append(te_result.get("rmse"))

                        result[nf]["test"] = tests_results
                        result[nf]["test_mean"] = np.mean(tests_results)
                        result[nf]["test_std"] = np.std(tests_results)

                        result[nf]["cv"] = cv_results
                        result[nf]["cv_mean"] = np.mean(cv_results)
                        result[nf]["cv_std"] = np.std(cv_results)

                        result[nf]["time"] = time_results
                        result[nf]["time_mean"] = np.mean(time_results)
                        result[nf]["time_std"] = np.std(time_results)

                        result[nf]["finished"] = True

                        pickle.dump(result, open(filename, "wb"))

                tests = [result[nf]["test"] for nf in nfolds]
                cvs = [result[nf]["cv"] for nf in nfolds]
                ts = [result[nf]["time"] for nf in nfolds]

                # Create paired test matrices only if exists differences among
                # sample distributions
                if not non_parametric_check_samples(tests):
                    test_nparam = matrix_non_parametric_paired_test(tests)
                    result["test_nparam"] = test_nparam

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
    print("Experiment 06")
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
