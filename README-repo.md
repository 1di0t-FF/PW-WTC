The paper is under review. Currently, input is only accessible to the editors and reviewers. The password is the article number. (26-TIE-XXXX)
# GNSS 欺骗检测对比实验
四种检测器对 12 个欺骗场景（4 环境 × FCS/PCS/JAM-SP）的独立复现包

## 运行

```bash
python compute_statistics.py            # 计算四种方法的检测量
python compute_results.py               # 计算检测率
python plot_comparison.py               # 绘图
```
场景可单独指定（如 `python compute_results.py E2-PCS`）。
依赖：Python（numpy / pandas / scipy）+ MATLAB R2024a（Wavelet Toolbox）。
MATLAB 路径在 `run_experiment.py` 顶部 `MATLAB` 常量处修改。

## 目录结构

```
config.json                 12 场景定义
compute_statistics.py       计算检测量
compute_results.py          计算检测结果
matlab/
  run_pwwtc_batch.m         批处理入口
  compute_matlab_wcoherence_batch.m   PW-WTC 内核
input/<MODE>/<场景>/         输入数据
results/<MODE>/<场景>/       运行产物，与 input 一一对应
figures/                    结果图
```

## 输入数据

每颗卫星三个等长序列：`C`（伪距，米）、`L`（载波相位，周）、`MP`（多径组合，米）。
FCS/PCS 为连续流；JAM-SP 按 `_au` / `_sp` 分段。
`receiver_clock*.csv` 为接收机钟漂 ΔclkB（ns/s）。
`thresholds.csv` 决策值：`PW_WTC_threshold`、`E_CCDC_threshold`、`CD2_P_threshold`、`CD2_P_center`。

## 方法

| 方法 | 统计量 | 报警 |
|---|---|---|
| PW-WTC | Base/Test MP 600 s 滑窗小波相干，COI 内全体点按 Base 日功率加权 | γ < 阈值 |
| E-CCDC | 多普勒一致性残差平方和（20 样本） | > 阈值 |
| CD2-P | 伪距二阶差分偏离中心的平方和（20 样本） | > 阈值 |
| clkBM | 钟漂出带（基线 = AU 前 600 有效样本均值 ± 15 ns/s） | 出带 |

统计量一律存于**窗口最后一个样本**（近似实时），窗不满留空；

## 产物

```
results/<MODE>/<场景>/satellites/Gxx.csv    检测量 PW_WTC, E_CCDC, CD2_P
results/<MODE>/<场景>/clkbm.csv             检测量 DeltaClkB
results/<MODE>/<场景>/detection_rates.csv   检测结果
```

