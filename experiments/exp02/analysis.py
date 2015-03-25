from main import *


def mean_std(mx, name="", bprint=False):

    mx_mean = np.mean(mx, axis=0)
    mx_std = np.std(mx, axis=0)

    if bprint:
        print(mx.shape)
        print(name, ":")
        print(mx)
        print(name, "mean: ", mx_mean)
        print(name, "std: ", mx_std)

    return mx_mean, mx_std


def analysis(name):
    time_means_mx = []
    cv_means_mx = []
    ranks_mx = []

    for asset in assets:
        filename = results_path + name + "_" + asset + ".p"

        try:
            result = pickle.load(open(filename, 'rb'))

            if result["finished"]:
                # print(name, "Asset: ", asset)
                # print(result["cv_nparam"])
                # print()

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

    rk_samples = [rk[:, j] for j in range(rk.shape[1])]
    rk_final = matrix_non_parametric_paired_test(rk_samples)
    # print("Rk final:")
    # print(rk_final)
    # print()

    print("(Eval | Rank | Rank Mean | Time Mean)")
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

    # print()
    # print("Analysis ELMR")
    # print()
    # analysis("elmr")

    # print()
    # print("Analysis SVR")
    # print()
    # analysis("svr")
