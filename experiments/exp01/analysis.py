from main import *
from tabulate import tabulate

def histo(mx):

    import matplotlib.pyplot as plt

    for j in range(mx.shape[1]):

        plt.hist(mx[:, j], normed=False, facecolor=colors[j], label=search_functions[j])

    plt.legend()
    plt.show()
    plt.close()


# def bargraph(mx):
#
#     import matplotlib.pyplot as plt
#
#     N = len(search_functions)
#
#     ind = np.arange(N)  # the x locations for the groups
#     width = 0.35       # the width of the bars
#
#     fig, ax = plt.subplots()
#
#
#     for j in range(mx.shape[1]):

def analysis(name):
    mx_time_means = []
    mx_cv_means = []
    mx_ranks = []
    mx_pct = []

    for asset in assets:
        filename = results_path + name + "_" + asset + ".p"

        try:
            result = pickle.load(open(filename, 'rb'))

            if result["finished"]:
                # print(name, "Asset: ", asset)

                vec_ranks = []
                vec_time_means = []
                vec_cv_means = []
                for i in range(result["cv_nparam"].shape[0]):
                    vec_cv_means.append(result[search_functions[i]]["cv_mean"])
                    vec_time_means.append(result[search_functions[i]]["time_mean"])
                    vec_ranks.append(np.sum(result["cv_nparam"][i, :]))

                #     sum = np.sum(result["cv_nparam"][i, :])
                #     print(search_functions[i], " is worst than ", sum, " functions")
                # print()


                print(tabulate(result["cv_nparam"], headers="firstrow", tablefmt="latex"))

                vec_pct = 100 * (vec_cv_means - np.min(vec_cv_means))/np.min(vec_cv_means)

                mx_pct.append(vec_pct)
                mx_cv_means.append(vec_cv_means)
                mx_time_means.append(vec_time_means)
                mx_ranks.append(vec_ranks)

        except:
            pass
            # print("Error: ", asset)
            # print()

    cv = np.array(mx_cv_means)
    ts = np.array(mx_time_means)
    rk = np.array(mx_ranks)
    pc = np.array(mx_pct)

    cv_mean, cv_std = mean_std(cv, "cv")
    ts_mean, ts_std = mean_std(ts, "ts")
    rk_mean, rk_std = mean_std(rk, "rk")
    pc_mean, pc_std = mean_std(pc, "pc")


    table = [["Search Function", "Ranking Sums"]]
    for j, sf in enumerate(search_functions):table.append([sf, np.sum(rk[:, j])])
    print(tabulate(table, headers="firstrow", tablefmt="latex"))

    rk_samples = [rk[:, j] for j in range(rk.shape[1])]
    print()
    print("Rankings have same distribution: ",
          non_parametric_check_samples(rk_samples))
    print()

    rk_final = matrix_non_parametric_paired_test(rk_samples)
    # print("Rk final:")
    # print(rk_final)
    # print()

    print(rk.shape)
    table = [["Search Function", "Rank", "Rank Mean", "% to Best", "Time Mean"]]
    rank = [(search_functions[i], np.sum(rk_final[i, :]), rk_mean[i], pc_mean[i], ts_mean[i]) for i in range(rk_final.shape[0])]
    rank = sorted(rank, key=lambda f: f[1] + f[2])
    for r in rank:table.append(list(r))
    print(tabulate(table, headers="firstrow", tablefmt="latex"))


if __name__ == "__main__":

    print()
    print()
    print("Experiment 01")
    print()
    print()

    print()
    print()
    print("Analysis ELMK")
    print()
    print()
    analysis("elmk")

    print()
    print()
    print("Analysis ELMR")
    print()
    print()
    analysis("elmr")

    print()
    print()
    print("Analysis SVR")
    print()
    print()
    analysis("svr")
