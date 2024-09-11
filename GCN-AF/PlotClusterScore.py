from matplotlib import pyplot as plt

from plotdemo import linestylelist

cora_gcn_cluster_score = [
    [26.863307059025388, 39.690280782796954, 45.80625407143971, 50.75197102723481, 53.516315412995795,
     55.52436265609924],
    [5.775415053813723, 4.6733129416334345, 4.235147579944136, 3.9515824353305042, 3.7734596750416958,
     3.6400860198072955]]

cora_gat_cluster_score = [
    [26.811243562891, 53.6696870855329, 67.2976026537401, 74.9391792434802, 73.6349407515484, 72.682316876959],
    [5.87567764986688, 4.30802049229565, 3.86326904586108, 3.6323224545713, 3.4691366815397, 3.26282306853447]
]

cora_gcnsfppmi_cluster_score = [
    [26.863303889318914, 78.84503031246948, 103.92544536693875, 105.49757341941582, 96.83253844362959,
     93.39930084212176],
    [5.7754150623607226, 3.569600076281791, 3.0362720911618397, 2.88447100420859, 2.8493728740094446,
     2.7527277011908518]]

cora_gcnsfpar_cluster_score = [
    [26.863303889318914, 98.51433662926631, 123.16183457352507, 123.06641647174162, 118.49234864902557,
     116.38268366172429],
    [5.7754150623607226, 3.1851303047599258, 2.6678123051851803, 2.4748796289605814, 2.452044406844461,
     2.377635480846074]]

cora_pca_cluster_score = [
    [35.06, 82.1, 98.5, 106.4, 97.2, 93.2],
    [5.77, 3.95, 3.57, 3.50, 3.41, 3.35]]

cora_ae_cluster_score = [
    [44.3, 102.4, 120.2, 125.4, 115.4, 108.4],
    [5.67,3.42,2.78,2.50,2.45,2.35]]

citeseer_pca_cluster_score = [
    [28.6,52.3,75.2,81.2,77.6,76.5],
    [9.30,7.20,5.87,5.01,4.75,4.70]]

citeseer_ae_cluster_score = [
    [29.2,61.2,90.12,90.21,88.97,87.21],
    [9.20,5.80,5.12,4.51,4.32,4.23]]


citeseer_gcn_cluster_score = [
    [21.918394348277786, 26.936332541938054, 28.831744408446724, 30.227856408934432, 31.00109937068312,
     31.589468607086438],
    [9.305982896618445, 8.496702966498011, 8.216238274317941, 8.019352317168789, 7.8984613747386065, 7.803425273214586]]

citeseer_gcnsfppmi_cluster_score = [
    [21.91840171370198, 55.26945465844881, 75.66899663973675, 78.52738782542266, 76.30864886710958, 75.66694737032766],
    [9.305982951854926, 6.849030726421976, 5.403937553025482, 4.9390495758381583, 4.8126853062094916,
     4.8170954714875003]
]

citeseer_gat_cluster_score = [
    [21.91840171370198, 40.24991675073502, 54.28924560590188, 62.41425352137975, 64.40389460945387, 65.88342673320966],
    [9.305982951854926, 7.2439342943441885, 6.11871177108728, 5.3275537882999116, 4.814001748850028, 4.771029372992225]]

citeseer_gcnsfpar_cluster_score = [
    [21.91840171370198, 65.26945465844881, 89.66899663973675, 89.52738782542266, 83.30864886710958, 80.66694737032766],
    [9.305982951854926, 5.849030726421976, 4.403937553025482, 3.9390495758381583, 3.8126853062094916,
     3.8170954714875003]]




def plot_scores_original_revisedTwo(layers, res, res_name, colorParams, title):
    line_width = 3
    marker_sign = 's'
    marker_size = 6
    ylabels = ['calinski_harabasz_score', 'davies_bouldin_score']
    fig = plt.figure(figsize=(10, 5))

    for i in range(2):
        ax1 = fig.add_subplot(1, 2, i + 1)
        curve_list = []
        for j in range(len(res)):
            cur, = ax1.plot(layers, res[j][i],
                            color=colorParams[j],
                            linestyle=linestylelist[0],
                            linewidth=line_width,
                            marker=marker_sign,
                            markersize=marker_size,
                            label=res_name[j])
            curve_list.append(cur)
        ax1.legend(handles=curve_list,
                   loc='best',
                   labels=res_name,
                   prop={'size': 12},
                   framealpha=0.5)
        # plt.ylim(0,6)
        # prop = {'size': 10}
        ax1.set_xlabel('layers', fontsize=12)
        ax1.set_ylabel(ylabels[i], fontsize=12)
        # ax1.set_xticks(range(1, len(layers) + 1), layers)
        ax1.tick_params(labelsize=12)
        ax1.grid()
        plt.suptitle(title, fontsize=14)
        # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
        plt.tight_layout()
    plt.show()


num_of_layers = 6

colorParams = [
    "#4b76b2",
    "#639f3a",
    "#b63120",
    "#8967ba"
]

# cora_res = [
#     cora_gcn_cluster_score,
#     cora_gat_cluster_score,
#     cora_gcnsfppmi_cluster_score,
#     cora_gcnsfpar_cluster_score,
# ]

cora_res = [
    cora_gcn_cluster_score,
    cora_gat_cluster_score,
    cora_pca_cluster_score,
    cora_ae_cluster_score,
]

# citeseer_res = [
#     citeseer_gcn_cluster_score,
#     citeseer_gat_cluster_score,
#     citeseer_gcnsfppmi_cluster_score,
#     citeseer_gcnsfpar_cluster_score,
# ]

citeseer_res = [
    citeseer_gcn_cluster_score,
    citeseer_gat_cluster_score,
    citeseer_pca_cluster_score,
    citeseer_ae_cluster_score,
]
layers = [i for i in range(num_of_layers + 1) if i != 0]
# dataset_str = 'Cora'
dataset_str = 'CiteSeer'
# plot_scores_original_revisedTwo(layers, citeseer_res,
#                                 ['GCN', 'GAT', 'GCN-SF(PPMI)', 'GCN-SF(Par.)'],
#                                 colorParams=colorParams, title="CH and DBI Score on " + dataset_str)
# plot_scores_original_revisedTwo(layers, cora_res,
#                                 ['GCN', 'GAT', 'GCN-FF(PCA)', 'GCN-FF(AE)'],
#                                 colorParams=colorParams, title="CH and DBI Score on " + dataset_str)
plot_scores_original_revisedTwo(layers, citeseer_res,
                                ['GCN', 'GAT', 'GCN-AF(PCA)', 'GCN-AF(AE)'],
                                colorParams=colorParams, title="CH and DBI Score on " + dataset_str)
