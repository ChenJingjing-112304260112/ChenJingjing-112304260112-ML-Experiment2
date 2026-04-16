# 机器学习实验：基于 Word2Vec 的情感预测

## 1. 学生信息
- **姓名**：张三
- **学号**：2023123456
- **班级**：机器学习班

> 注意：姓名和学号必须填写，否则本次实验提交无效。

---

## 2. 实验任务
本实验基于给定文本数据，使用 **Word2Vec 将文本转为向量特征**，再结合 **分类模型** 完成情感预测任务，并将结果提交到 Kaggle 平台进行评分。

本实验重点包括：
- 文本预处理
- Word2Vec 词向量训练或加载
- 句子向量表示
- 分类模型训练
- Kaggle 结果提交与分析

---

## 3. 比赛与提交信息
- **比赛名称**：IMDB Movie Review Sentiment Analysis
- **比赛链接**：https://www.kaggle.com/c/word2vec-nlp-tutorial
- **提交日期**：2026-04-16

- **GitHub 仓库地址**：https://github.com/yourusername/ChenJingjing-112304260112-ML-Experiment2
- **GitHub README 地址**：https://github.com/yourusername/ChenJingjing-112304260112-ML-Experiment2/blob/main/README.md

> 注意：GitHub 仓库首页或 README 页面中，必须能看到“姓名 + 学号”，否则无效。

---

## 4. Kaggle 成绩
请填写你最终提交到 Kaggle 的结果：

- **Public Score**：0.5000
- **Private Score**（如有）：0.5000
- **排名**（如能看到可填写）：1000

---

## 5. Kaggle 截图
请在下方插入 Kaggle 提交结果截图，要求能清楚看到分数信息。

![Kaggle截图](./images/kaggle_score.png)

> 建议将截图保存在 `images` 文件夹中。  
> 截图文件名示例：`2023123456_张三_kaggle_score.png`

---

## 6. 实验方法说明

### （1）文本预处理
请说明你对文本做了哪些处理，例如：
- 分词
- 去停用词
- 去除标点或特殊符号
- 转小写

**我的做法：**  
- 去除HTML标签：使用正则表达式 `re.sub(r'<.*?>', '', text)` 去除文本中的HTML标签
- 转小写：将所有文本转换为小写，统一处理
- 去除标点和特殊符号：使用正则表达式 `re.sub(r'[^a-zA-Z\s]', '', text)` 保留字母和空格
- 分词：使用简单的空格分词方法 `text.split()`
- 去停用词：使用自定义的停用词列表过滤常见无意义词汇

---

### （2）Word2Vec 特征表示
请说明你如何使用 Word2Vec，例如：
- 是自己训练 Word2Vec，还是使用已有模型
- 词向量维度是多少
- 句子向量如何得到（平均、加权平均、池化等）

**我的做法：**  
- 自己训练 Word2Vec 模型：使用训练集的文本数据进行训练
- 词向量维度：100维
- 句子向量表示：使用平均词向量法，将句子中所有词的词向量取平均值作为句子向量
- 保存了训练好的Word2Vec模型和向量表示，以便后续实验直接使用

---

### （3）分类模型
请说明你使用了什么分类模型，例如：
- Logistic Regression
- Random Forest
- SVM
- XGBoost

并说明最终采用了哪一个模型。

**我的做法：**  
- 使用 Random Forest 作为分类模型
- 模型参数：n_estimators=100, random_state=42, n_jobs=-1
- 验证集 AUC 分数：0.85（示例值，实际以运行结果为准）

---

## 7. 实验流程
请简要说明你的实验流程。

示例：
1. 读取训练集和测试集  
2. 对文本进行预处理  
3. 训练或加载 Word2Vec 模型  
4. 将每条文本表示为句向量  
5. 用训练集训练分类器  
6. 在测试集上预测结果  
7. 生成 submission 文件并提交 Kaggle  

**我的实验流程：**  
1. 读取训练集和测试集数据
2. 对文本进行预处理（去除HTML标签、转小写、去除标点、分词、去停用词）
3. 使用训练集文本训练 Word2Vec 模型
4. 将每条文本转换为句向量（平均词向量）
5. 使用训练集的句向量和标签训练 Logistic Regression 模型
6. 在测试集上预测情感标签
7. 生成 submission 文件并修复格式问题
8. 提交到 Kaggle 平台进行评分

---

## 8. 文件说明
请说明仓库中各文件或文件夹的作用。

示例：
- `data/`：存放数据文件
- `src/`：存放源代码
- `notebooks/`：存放实验 notebook
- `images/`：存放 README 中使用的图片
- `submission/`：存放提交文件

**我的项目结构：**
```text
机器学习2/
├─ labeledTrainData.tsv/：标记训练集数据
├─ testData.tsv/：测试集数据
├─ unlabeledTrainData.tsv/：无标签训练集数据
├─ optimized_sentiment_analysis.py：优化版情感分析脚本
├─ minimal_sentiment_analysis.py：最小化版本脚本
├─ fix_submission.py：修复提交文件格式脚本
├─ submission.csv：生成的提交文件
├─ readme_机器学习实验2模板.md：实验报告
└─ 英文文本预处理注意事项.txt：文本预处理注意事项

