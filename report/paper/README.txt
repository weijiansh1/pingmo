论文《基于飞行品质的六自由度对象族构建》的源码。
data/ 由 scripts/make_data.py 从对象族数据生成（仅用 Python 标准库）：
  python3 -I scripts/make_data.py ../../six_channel_model_2026-10-29 data
编译（XeTeX 引擎）：
  tectonic -X compile paper.tex
