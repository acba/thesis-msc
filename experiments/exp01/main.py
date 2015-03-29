from tools import *

import pai
from pai.portfolio import get_data


assets = open('ibovespa.txt', 'r').read().split('\n')
results_path = "results/"
search_functions = ["random search", "grid search", "particle swarm",
                    "nelder-mead", "cma-es"]


def run(name):

    # Init regressor
    r = pai.Regressor(name)

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Iterate over assets list
    for i, asset in enumerate(assets):

        filename = results_path + name + "_" + asset + ".p"

        # Check if need to run this asset
        if check(filename):
            try:
                # Get data and create dataset matrix
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
                                       kf=["rbf"], f=["sigmoid"])
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

                # Save data
                pickle.dump(result, open(filename, "wb"))

            except:
                print("Error: ", asset)


if __name__ == "__main__":

    print()
    print("Experiment 01")
    print()

    print()
    print("Run ELMK")
    print()
    run("elmk")

    print()
    print("Run ELMR")
    print()
    run("elmr")

    print()
    print("Run SVR")
    print()
    run("svr")
