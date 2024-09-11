from matplotlib import pyplot as plt

from FeatUtils import plot_bar
from plotdemo import linestylelist, colorlist

ppmi_pca = [87.88,77.70,62.21,53.87,37.25,84.96,87.50,86.08]
ppmi_ae = [88.57,78.87,68.68,55.45,37.11,85.13,88.12,86.65]
par_pca = [87.22,78.31,67.91,55.94,37.98,86.21,88.52,86.97]
par_ae = [89.54,79.66,69.50,56.61,38.52,86.98,89.73,87.12]

dataset_strs_plot = ['Cora', 'Cite.', 'Cham.', 'Squi.', 'Actor', 'Corn.', 'Texa.', 'Wisc.']

line_width = 3
marker_sign = 's'
marker_size = 6
# ylabels = ['calinski_harabasz_score', 'davies_bouldin_score']
# fig = plt.figure()
res = ['GCN-SF-AF(PPMI&PCA)', 'GCN-SF-AF(PPMI&AE)', 'GCN-SF-AF(Par.&PCA)', 'GCN-SF-AF(Par.&AE)', ]

y_list = [ppmi_pca, ppmi_ae, par_pca, par_ae]

cora_curve_list = []
fig, ax = plt.subplots(1, 1, figsize=(8,6))

plot_bar(ax,x1=dataset_strs_plot,
         y_list=y_list,
         xlabel='Datasets',ylabel='Accuracy(%)',
         legends_list=res,
         color_list=[colorlist[i] for i in range(4)],
         bar_width=0.2,
         title='Accuracy of GCN-SF-AFs on Datasets'
         )
# for i, r in enumerate(res):
#     cur, = ax.plot(x, y_list[i],
#                                color=colorlist[i],
#                                linestyle=linestylelist[i],
#                                linewidth=line_width,
#                                marker=marker_sign,
#                                markersize=marker_size,
#                                label=res[i]
#                                )
#     cora_curve_list.append(cur)

ax.legend(loc='best',
          labels=res,
          prop={'size': 12},
          framealpha=0.5)
ax.set_xlabel('Datasets', fontsize=12)
ax.set_ylabel('Accuracy(%)', fontsize=12)
# ax1.set_xticks(range(1, len(layers) + 1), layers)
ax.tick_params(labelsize=12)
ax.grid()
ax.set_title(ax.get_title(),fontsize=12)
plt.show()
