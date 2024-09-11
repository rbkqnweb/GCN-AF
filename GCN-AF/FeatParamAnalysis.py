import numpy as np
from matplotlib import pyplot as plt

# Prepare Data
# from PlotClusterScore import colorParams
from plotdemo import linestylelist, colorlist

cora_list = [1024, 512, 256, 128, 64, 32, 16, 8, 4]
cora_list.reverse()
cite_list = [2048, 1024, 512, 256, 128, 64, 32, 16, 8]
cite_list.reverse()

pca_cora = [74.56, 75.83, 79.86, 82.51, 87.31, 88.67, 87.72, 86.72, 85.93]
pca_cite = [69.11, 70.43, 75.73, 76.86, 77.29, 78.17, 76.94, 76.25, 75.16]
ae_cora = [77.22, 80.58, 83.74, 86.12, 89.34, 88.92, 88.53, 88.31, 87.43]
ae_cite = [70.43, 72.89, 75.02, 76.19, 78.75, 79.57, 78.53, 77.16, 76.21]

line_width = 3
marker_sign = 's'
marker_size = 6
# ylabels = ['calinski_harabasz_score', 'davies_bouldin_score']
# fig = plt.figure()
res = ['GCN-AF(PCA)', 'GCN-AF(AE)']

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
cur_pca_cora, = ax[0].plot(cora_list, pca_cora,
                           color=colorlist[0],
                           linestyle=linestylelist[0],
                           linewidth=line_width,
                           marker=marker_sign,
                           markersize=marker_size,
                           label='GCN-AF(PCA)'
                           )
cur_ae_cora, = ax[0].plot(cora_list, ae_cora,
                          color=colorlist[1],
                          linestyle=linestylelist[0],
                          linewidth=line_width,
                          marker=marker_sign,
                          markersize=marker_size,
                          label='GCN-AF(AE)'
                          )
cora_curve_list = []
cora_curve_list.append(cur_pca_cora)
cora_curve_list.append(cur_ae_cora)
ax[0].legend(handles=cora_curve_list,
           loc='best',
           labels=res,
           prop={'size': 12},
           framealpha=0.5)
ax[0].set_xlabel('dims', fontsize=12)
ax[0].set_ylabel('Accuracy(%)', fontsize=12)
# ax1.set_xticks(range(1, len(layers) + 1), layers)
ax[0].tick_params(labelsize=12)
ax[0].grid()
ax[0].set_title('Param Analysis on Cora', fontsize=12)
ax[0].axvline(x=128, color='r', linestyle='-.', linewidth=2)
ax[0].axvline(x=64, color='r', linestyle='-.', linewidth=2)

cur_pca_cite, = ax[1].plot(cite_list, pca_cite,
                           color=colorlist[0],
                           linestyle=linestylelist[0],
                           linewidth=line_width,
                           marker=marker_sign,
                           markersize=marker_size,
                           label='GCN-AF(PCA)'
                           )
cur_ae_cite,  = ax[1].plot(cite_list, ae_cite,
                          color=colorlist[1],
                          linestyle=linestylelist[0],
                          linewidth=line_width,
                          marker=marker_sign,
                          markersize=marker_size,
                          label='GCN-AF(AE)'
                          )
cite_curve_list = []
cite_curve_list.append(cur_pca_cite)
cite_curve_list.append(cur_ae_cite)
ax[1].legend(handles=cite_curve_list,
           loc='best',
           labels=res,
           prop={'size': 12},
           framealpha=0.5)
ax[1].set_xlabel('dims', fontsize=12)
ax[1].set_ylabel('Accuracy(%)', fontsize=12)
# ax[1].set_xticks(cite_list)
ax[1].axvline(x=256, color='r', linestyle='-.', linewidth=2)
# ax[1].axvline(x=128, color='r', linestyle='--', linewidth=0.8)
ax[1].tick_params(labelsize=12)
ax[1].grid()
ax[1].set_title('Param Analysis on CiteSeer', fontsize=12)
plt.show()