# 颜色列表
colorlist = [
"#4b76b2",
"#e1830f",
"#639f3a",
"#b63120",
"#8967ba",
"#7e574b",
"#ca79bf",
"#7f7f7f",
"#bcbd35",
"#73bdcf",
]

# 线的样式
# 实线 虚线 点划线 点虚线 无线
linestylelist = [
    '-',
    '--',
    '-.',
    ':',
    ''
]

# 点的样式
# 点 像素
# 上下左右三角形
# 上下左右三叉线
# 圆形
# 方形
# 五边形
# 六边形
# 五角星
# 十字交叉
# 	横线
markerlist = ['.', ',',
              '^', 'v', '>', '<',
              '1', '2', '3', '4',
              'o',
              's', 'D',
              'p',
              'h', 'H',
              '*',
              '+', 'x',
              '_']

# x = [1, 2, 3, 4, 5]
# y1 = [1, 1, 1, 1, 1]
# y2 = [2, 2, 2, 2, 2]
# y3 = [3, 3, 3, 3, 3]
# y4 = [4, 4, 4, 4, 4]
# y5 = [5, 5, 5, 5, 5]
#
# c1, = plt.plot(x, y1,
#                color=colorlist[0],
#                linestyle=linestylelist[0],
#                marker=markerlist[0],
#                linewidth=3)
# c2, = plt.plot(x, y2,
#                color=colorlist[1],
#                linestyle=linestylelist[2],
#                marker=markerlist[0],
#                linewidth=3)
# c3, = plt.plot(x, y3,
#                color=colorlist[2],
#                linestyle=linestylelist[1],
#                marker=markerlist[0],
#                linewidth=3)
# c4, = plt.plot(x, y4,
#                color=colorlist[3],
#                linestyle=linestylelist[2],
#                marker=markerlist[0],
#                linewidth=3)
# c5, = plt.plot(x, y5,
#                color=colorlist[4],
#                linestyle=linestylelist[2],
#                marker='s',
#                linewidth=3,
#                markersize=6)
# plt.grid()
#
# # loc:
# # best
# # upper right
# # upper left
# # lower left
# # lower right
# # right
# # center left
# # center right
# # lower center
# # upper center
# # center
# ncol=3				#图例分成3列
# frameon=False		#不要图例框架
# framealpha=0.5		#图例框架的透明度
# edgecolor			#图例框架边缘颜色
# prop=font2			#图例采用的字体参数
# fontsize=10			#图例所用字体大小，这个参数只有在没有指定prop参数的时候才会发挥作用。
# plt.legend(handles=[c1, c2, c3, c4, c5],
#            labels=['a', 'b', 'c', 'd', 'e'],
#            loc='upper right',
#            prop={'size': 14})
# plt.xlabel('x', fontsize=14)
# plt.ylabel('y', fontsize=14)
# plt.tick_params(labelsize=12)  # #调整坐标轴数字大小
# plt.title('Title', fontsize=14)
# plt.show()
