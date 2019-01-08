1.单个特征：
点击次数、数量、点击率

2.组合特征：点击次数、数量、点击率

3.prefix和query的相似特征：利用的是长度近似的特征

3.向量化：prefix、title、tag进行向量化


接下来要做的是：
1.统计特征：统计prefix、title和tag的数量
3_feature:338671
2_feature：322667

在val里面，可以找到38241条出现过的。
在test里面，可以找到38158条出现过的。

2.向量化+普通的特征。

3.在train中出现过的：
	val 38330 
	test 38232
没有出现过的：
	val 11670
	test 11768


4.title删除prefix，做统计特征。
打标签的特征用多少个

5.在向量化特征的时候没有lower

6.特征构造：提取关键词语，然后进行统计

-----------------------------------------------------------------
1.train_baseline.py
直接使用lightgbm对数据进行分类

2.train_baseline_split.py：基础版本
把数据进行分割：
test data 在train data中出现过; test data在train data中没有出现过

3.train_baseline_split2.py
把数据进行分割：test data 在train data中出现过; test data在train data中没有出现过

4.train_baseline_splie3.py:可以跑的版本，最后的结果可以达到0.738左右

-----------------------------------------------------------------
dnn开头的都是神经网络跑出来的结果，其最新版本为dnn_data5.py

-----------------------------------------------------------------
train_GBDT_LR.py:结合了GBDT和LR
