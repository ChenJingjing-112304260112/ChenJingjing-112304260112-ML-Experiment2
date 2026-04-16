# 机器学习实验：基于 Word2Vec 的情感预测

## 1. 学生信息
- **姓名**：陈晶晶
- **学号**：112304260112
- **班级**：数据1231班

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

- **GitHub 仓库地址**：https://github.com/ChenJingjing-112304260112/ChenJingjing-112304260112-ML-Experiment2
- **GitHub README 地址**：https://github.com/ChenJingjing-112304260112/ChenJingjing-112304260112-ML-Experiment2/blob/main/README.md

> 注意：GitHub 仓库首页或 README 页面中，必须能看到“姓名 + 学号”，否则无效。

---

## 4. Kaggle 成绩
请填写你最终提交到 Kaggle 的结果：

- **Public Score**：0.86336
- **Private Score**（如有）：0.86336
- **排名**（如能看到可填写）：

---

## 5. Kaggle 截图
请在下方插入 Kaggle 提交结果截图，要求能清楚看到分数信息。

![Kaggle截图](./images/kaggle_score.png)

> 建议将截图保存在 `images` 文件夹中。  
> 截图文件名示例：`112304260112_ChenJingjing_kaggle_score.png`

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
- 去停用词：使用自定义的停用词列表过滤常见无意义词汇，保留否定词

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
- 使用 Logistic Regression 作为分类模型
- 模型参数：C=0.1, max_iter=1000, random_state=42, n_jobs=-1
- 验证集 AUC 分数：0.86+（实际以运行结果为准）

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
├─ README.md：实验报告
└─ 英文文本预处理注意事项.txt：文本预处理注意事项
```

---

## 9. 实验总结
请总结你的实验过程、遇到的问题及解决方法、以及对结果的分析。

**实验总结：**
- 本次实验使用 Word2Vec 结合 Logistic Regression 完成了情感分析任务
- 重点优化了文本预处理步骤，保留了否定词以提高模型性能
- 通过调整模型参数和特征工程，将 AUC 分数从 0.81720 提升到 0.86336
- 遇到的主要问题是内存不足，通过降低 Word2Vec 维度和使用更高效的模型解决
- 后续可以尝试使用更复杂的模型（如 XGBoost）和更多的特征工程来进一步提升性能

---

## 10. 参考资料
请列出你在实验过程中参考的资料，例如论文、博客、教程等。

**参考资料：**
- Word2Vec 官方文档：https://radimrehurek.com/gensim/models/word2vec.html
- scikit-learn 文档：https://scikit-learn.org/stable/
- Kaggle 比赛页面：https://www.kaggle.com/c/word2vec-nlp-tutorial