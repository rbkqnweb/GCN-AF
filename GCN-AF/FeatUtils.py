import numpy as np
from matplotlib.ticker import FuncFormatter


def setAxeFontSize(ax,title_size=14,axis_size=12,tick_size=12):
    tick_label_fontsize = tick_size
    axis_label_fontsize = axis_size
    title_fontsize = title_size

    ax.set_title(label=ax.get_title(), fontsize=title_fontsize)
    ax.set_xlabel(ax.get_xlabel(), fontsize=axis_label_fontsize)
    ax.set_ylabel(ax.get_ylabel(), fontsize=axis_label_fontsize)
    ax.tick_params(labelsize=tick_label_fontsize)

def to_percent(y, position):
    # 将y值乘以100并格式化为字符串，添加百分号
    s = "{:.0f}".format(y * 100)
    return s


def plot_bar(ax, x1, y_list, xlabel, ylabel, legends_list, color_list, bar_width=0.35, title=None):
    # mpl.rcParams["font.family"] = ["SimSun"]
    # mpl.rcParams["font.size"] = 12
    # params = {'font.weight': 'bold'}
    # font = FontProperties(fname='SimSun', size=14)
    index = np.arange(len(x1))
    ax.bar(index,y_list[0],bar_width,color=color_list[0], align="center", label=legends_list[0])
    ax.bar(index+bar_width,y_list[1],bar_width,color=color_list[1], align="center", label=legends_list[1],)
    ax.bar(index+bar_width*2,y_list[2],bar_width,color=color_list[2], align="center", label=legends_list[2],)
    ax.bar(index+bar_width*3,y_list[3],bar_width,color=color_list[3], align="center", label=legends_list[3],)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    # set xaxis ticks and ticklabels
    ax.set_xticks(index + bar_width*1.5)
    ax.set_xticklabels(x1)
    ax.tick_params(labelsize=14)
    # ax.legend(prop={'size': 14}, loc='upper left', framealpha=0.3)
    ax.set_title(title,fontsize=14)
    # 设置y轴的格式化器为我们定义的百分位数格式化函数
    # formatter = FuncFormatter(to_percent)
    # ax.yaxis.set_major_formatter(formatter)
    # ax.set_ylim(0,1.0)
    # plt.rcParams.update(params)
    # ax.grid()
    # plt.show()
