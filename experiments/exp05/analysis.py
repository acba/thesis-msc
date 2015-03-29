from main import *


def analysis(name):
    time_means_mx = []
    test_means_mx = []
    ranks_mx = []

    for asset in assets:
        filename = results_path + name + "_" + asset + ".p"

        try:
            result = pickle.load(open(filename, 'rb'))

            if result["finished"]:
                print(name, "Asset: ", asset)

                if "test_nparam" not in result:
                    print()
                    for i in cvs:
                        print(i)
                        print(result[i]["test"])

                    tests = [result[nf]["test"] for nf in cvs]
                    print(non_parametric_check_samples(tests))
                    test_nparam = matrix_non_parametric_paired_test(tests)
                    print("test_nparam")
                    print(test_nparam)

                ranks = []
                time_means = []
                test_means = []
                for i in range(result["test_nparam"].shape[0]):
                    test_means.append(result[cvs[i]]["test_mean"])
                    time_means.append(result[cvs[i]]["time_mean"])
                    ranks.append(np.sum(result["test_nparam"][i, :]))

                #     sum = np.sum(result["test_nparam"][i, :])
                #     print(cvs[i], " is worst than ", sum, " functions")
                # print()
                #
                # print("ranks: ", ranks)
                # print("adj: ", adj(ranks))

                test_means_mx.append(test_means)
                time_means_mx.append(time_means)
                ranks_mx.append(ranks)

        except:
            pass
            # print("Error: ", asset)
            # print()

    te = np.array(test_means_mx)
    ts = np.array(time_means_mx)
    rk = np.array(ranks_mx)

    te_mean, te_std = mean_std(te, "test")
    ts_mean, ts_std = mean_std(ts, "ts")
    rk_mean, rk_std = mean_std(rk, "rk")

    print()
    print("Cross-Validation Method | Ranking Sums")
    for j, cv in enumerate(cvs):print(cv, ": ", np.sum(rk[:, j]))
    print()

    rk_samples = [rk[:, j] for j in range(rk.shape[1])]
    print("Rankings have same distribution: ",
          non_parametric_check_samples(rk_samples))

    rk_final = matrix_non_parametric_paired_test(rk_samples)
    # print("Rk final:")
    # print(rk_final)
    # print()

    print(rk.shape)
    print("(Cross-Validation Method | Rank | Rank Mean | Time Mean)")
    rank = [(cvs[i], np.sum(rk_final[i, :]), rk_mean[i], ts_mean[i]) for i in range(rk_final.shape[0])]
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

    # print()
    # print("Analysis SVR")
    # print()
    # analysis("svr")
