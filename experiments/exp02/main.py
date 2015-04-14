from tools import *

import pai

assets = open('ibovespa.txt', 'r').read().split('\n')
results_path = "results/"
search_function = "particle swarm"
evals = [10, 30, 50, 80, 100, 150, 200]


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

                for ev in evals:
                    result[ev] = {}

                    print(asset, ev)

                    metrics = []
                    time = []
                    for it in range(50):
                        start = datetime.datetime.now()
                        r.search_param(database=data, cv="ts", cv_nfolds=10,
                                       of="rmse", opt_f=search_function, eval=ev,
                                       print_log=False, kf=["rbf"], f=["sigmoid"])
                        end = datetime.datetime.now()
                        delta = end - start

                        time.append(delta.total_seconds())
                        metrics.append(r.regressor.cv_best_error)

                    result[ev]["cv"] = metrics
                    result[ev]["cv_mean"] = np.mean(metrics)
                    result[ev]["cv_std"] = np.std(metrics)

                    result[ev]["time"] = time
                    result[ev]["time_mean"] = np.mean(time)
                    result[ev]["time_std"] = np.std(time)

                cvs = [result[ev]["cv"] for ev in evals]
                ts = [result[ev]["time"] for ev in evals]

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
