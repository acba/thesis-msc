from main import *


def analysis(name):
    time_means_mx = []
    cv_means_mx = []
    ranks_mx = []

    for asset in assets:
        filename = results_path + name + "_" + asset + ".p"

        try:
            result = pickle.load(open(filename, 'rb'))

            if result["finished"]:
                print(name, "Asset: ", asset)
                # print(result["cv_nparam"])
                # print()

                if "cv_nparam" not in result:
                    print()
                    for i in evals:
                        print(i)
                        print(result[i]["cv"])

                    tests = [result[sf]["cv"] for sf in evals]
                    print(non_parametric_check_samples(tests))
                    test_nparam = matrix_non_parametric_paired_test(tests)
                    print("cv_nparam")
                    print(test_nparam)


                ranks = []
                time_means = []
                cv_means = []
                for i in range(result["cv_nparam"].shape[0]):
                    cv_means.append(result[evals[i]]["cv_mean"])
                    time_means.append(result[evals[i]]["time_mean"])
                    ranks.append(np.sum(result["cv_nparam"][i, :]))
                    # sum = np.sum(result["cv_nparam"][i, :])
                    # print(evals[i], " is worst than ", sum, " functions")
                # print()

                cv_means_mx.append(cv_means)
                time_means_mx.append(time_means)
                ranks_mx.append(ranks)

        except:
            pass
            # print("Error: ", asset)
            # print()

    cv = np.array(cv_means_mx)
    ts = np.array(time_means_mx)
    rk = np.array(ranks_mx)

    cv_mean, cv_std = mean_std(cv, "cv")
    ts_mean, ts_std = mean_std(ts, "ts")
    rk_mean, rk_std = mean_std(rk, "rk")
    print()
    print("Evaluations | Ranking Sums")
    for j, ev in enumerate(evals):print(ev, ": ", np.sum(rk[:, j]))
    print()

    rk_samples = [rk[:, j] for j in range(rk.shape[1])]
    print("Rankings have same distribution: ",
          non_parametric_check_samples(rk_samples))
    rk_final = matrix_non_parametric_paired_test(rk_samples)
    # print("Rk final:")
    # print(rk_final)
    # print()

    print(rk.shape)
    print("(Evaluations | Rank | Rank Mean | Time Mean)")
    rank = [(evals[i], np.sum(rk_final[i, :]), rk_mean[i], ts_mean[i]) for i in range(rk_final.shape[0])]
    rank = sorted(rank, key=lambda f: f[1] + f[2])
    print()
    for r in rank:print(r)


if __name__ == "__main__":

    print()
    print("Experiment 02")
    print()

    print()
    print("Analysis ELMK")
    print()
    analysis("elmk")

    print()
    print("Analysis ELMR")
    print()
    analysis("elmr")

    print()
    print("Analysis SVR")
    print()
    analysis("svr")
