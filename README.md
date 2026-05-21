# MovieRatingPrediction
电影评分预测（回归任务），基于 MindSpore 实现。

## 环境
- Python 3.8/3.9
- MindSpore 2.7.1.post1 (CPU)
- 安装：`pip install mindspore==2.7.1.post1 pandas numpy matplotlib scikit-learn`

## 目录结构
- `data/raw/`         原始数据
- `data/processed/`   划分后的 train.csv / test.csv
- `src/`              源代码
- `outputs/`          预测结果、图表
- `report/`           报告与PPT

## 分工
- 成员A：`data_loader.py` + `baselines.py`
- 成员B：`models.py` + `train.py`
- 成员C：`evaluate.py` + 报告/PPT

## 运行方式
   
