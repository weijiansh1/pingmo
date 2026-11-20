论文《基于飞行品质的飞控大模型构建与六通道扩展》（阶段稿）的源码。
data/ 由两个脚本从对象族数据生成（仅用 Python 标准库）：
  python3 -I scripts/make_data.py ../../six_channel_model_2026-10-29 data
  python3 -I scripts/make_scatter_pdf.py data      # 参数空间散点的点云 PDF
编译（XeTeX 引擎）：
  tectonic -X compile paper.tex
