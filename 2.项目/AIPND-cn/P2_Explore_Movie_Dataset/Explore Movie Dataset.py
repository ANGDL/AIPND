
# coding: utf-8

# ## 探索电影数据集
# 
# 在这个项目中，你将尝试使用所学的知识，使用 `NumPy`、`Pandas`、`matplotlib`、`seaborn` 库中的函数，来对电影数据集进行探索。
# 
# 下载数据集：
# [TMDb电影数据](https://s3.cn-north-1.amazonaws.com.cn/static-documents/nd101/explore+dataset/tmdb-movies.csv)
# 

# 数据集各列名称的含义：
# <table>
# <thead><tr><th>列名称</th><th>id</th><th>imdb_id</th><th>popularity</th><th>budget</th><th>revenue</th><th>original_title</th><th>cast</th><th>homepage</th><th>director</th><th>tagline</th><th>keywords</th><th>overview</th><th>runtime</th><th>genres</th><th>production_companies</th><th>release_date</th><th>vote_count</th><th>vote_average</th><th>release_year</th><th>budget_adj</th><th>revenue_adj</th></tr></thead><tbody>
#  <tr><td>含义</td><td>编号</td><td>IMDB 编号</td><td>知名度</td><td>预算</td><td>票房</td><td>名称</td><td>主演</td><td>网站</td><td>导演</td><td>宣传词</td><td>关键词</td><td>简介</td><td>时常</td><td>类别</td><td>发行公司</td><td>发行日期</td><td>投票总数</td><td>投票均值</td><td>发行年份</td><td>预算（调整后）</td><td>票房（调整后）</td></tr>
# </tbody></table>
# 

# **请注意，你需要提交该报告导出的 `.html`、`.ipynb` 以及 `.py` 文件。**

# 
# 
# ---
# 
# ---
# 
# ## 第一节 数据的导入与处理
# 
# 在这一部分，你需要编写代码，使用 Pandas 读取数据，并进行预处理。

# 
# **任务1.1：** 导入库以及数据
# 
# 1. 载入需要的库 `NumPy`、`Pandas`、`matplotlib`、`seaborn`。
# 2. 利用 `Pandas` 库，读取 `tmdb-movies.csv` 中的数据，保存为 `movie_data`。
# 
# 提示：记得使用 notebook 中的魔法指令 `%matplotlib inline`，否则会导致你接下来无法打印出图像。

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


movie_data = pd.read_csv('tmdb-movies.csv')


# ---
# 
# **任务1.2: ** 了解数据
# 
# 你会接触到各种各样的数据表，因此在读取之后，我们有必要通过一些简单的方法，来了解我们数据表是什么样子的。
# 
# 1. 获取数据表的行列，并打印。
# 2. 使用 `.head()`、`.tail()`、`.sample()` 方法，观察、了解数据表的情况。
# 3. 使用 `.dtypes` 属性，来查看各列数据的数据类型。
# 4. 使用 `isnull()` 配合 `.any()` 等方法，来查看各列是否存在空值。
# 5. 使用 `.describe()` 方法，看看数据表中数值型的数据是怎么分布的。
# 
# 

# In[3]:


# .head()
movie_data.head(2)


# In[4]:


# .tril()
movie_data.tail(2)


# In[5]:


# .sample()
movie_data.sample()


# In[6]:


#..dtypes
movie_data.dtypes


# In[7]:


#.isnull() .any()
movie_data.isnull().any()


# In[8]:


# .describe()
movie_data.describe()


# ---
# 
# **任务1.3: ** 清理数据
# 
# 在真实的工作场景中，数据处理往往是最为费时费力的环节。但是幸运的是，我们提供给大家的 tmdb 数据集非常的「干净」，不需要大家做特别多的数据清洗以及处理工作。在这一步中，你的核心的工作主要是对数据表中的空值进行处理。你可以使用 `.fillna()` 来填补空值，当然也可以使用 `.dropna()` 来丢弃数据表中包含空值的某些行或者列。
# 
# 任务：使用适当的方法来清理空值，并将得到的数据保存。

# In[9]:


# 查看各个列的数据类型
movie_data.info()


# In[10]:


# 查看非数值类型的格列Nan的个数
movie_data.select_dtypes(include=object).isnull().sum()


# In[11]:


# 因为imdb_id、overview，genres的包含nan的值较少，
# 所以这里drop掉这些包含non值的行，不会过分影响数据
movie_data = movie_data.dropna(subset=['imdb_id', 'overview', 'genres'])
# 其他列含nan的值填充'Unknown'
movie_data[movie_data.select_dtypes(include=object).columns] = movie_data[movie_data.select_dtypes(include=object).columns].fillna('Unknown')


# In[12]:


# 查看数值类型的格列Nan的个数
movie_data.select_dtypes(exclude=object).isnull().sum()


# In[13]:


# 根据movie_data.describe()，budget、revenue、budget_adj、revenue_adj 
# 这4列数据存在大量的0
print(movie_data[(movie_data['budget']==0)]['budget'].value_counts())
print(movie_data[(movie_data['budget_adj']==0)]['budget_adj'].value_counts())
print(movie_data[(movie_data['revenue_adj']==0)]['revenue_adj'].value_counts())
print(movie_data[(movie_data['revenue']==0)]['revenue'].value_counts())


# In[14]:


# 把0值替换为 nan
movie_data[['budget', 'budget_adj', 'revenue_adj', 'revenue']] = movie_data[['budget', 'budget_adj', 'revenue_adj', 'revenue']].replace(0, np.nan)
# 把nan做插值
movie_data[['budget', 'budget_adj', 'revenue_adj', 'revenue']] = movie_data[['budget', 'budget_adj', 'revenue_adj', 'revenue']].interpolate()


# In[15]:


# 查看是否处理完
movie_data.describe()


# ---
# 
# ---
# 
# ## 第二节 根据指定要求读取数据
# 
# 
# 相比 Excel 等数据分析软件，Pandas 的一大特长在于，能够轻松地基于复杂的逻辑选择合适的数据。因此，如何根据指定的要求，从数据表当获取适当的数据，是使用 Pandas 中非常重要的技能，也是本节重点考察大家的内容。
# 
# 

# ---
# 
# **任务2.1: ** 简单读取
# 
# 1. 读取数据表中名为 `id`、`popularity`、`budget`、`runtime`、`vote_average` 列的数据。
# 2. 读取数据表中前1～20行以及48、49行的数据。
# 3. 读取数据表中第50～60行的 `popularity` 那一列的数据。
# 
# 要求：每一个语句只能用一行代码实现。

# In[16]:


movie_data[['id', 'popularity','budget', 'runtime', 'vote_average']]


# In[17]:


movie_data.iloc[list(range(0,20)) +  [47, 48]]


# In[18]:


movie_data['popularity'][49:60]


# ---
# 
# **任务2.2: **逻辑读取（Logical Indexing）
# 
# 1. 读取数据表中 **`popularity` 大于5** 的所有数据。
# 2. 读取数据表中 **`popularity` 大于5** 的所有数据且**发行年份在1996年之后**的所有数据。
# 
# 提示：Pandas 中的逻辑运算符如 `&`、`|`，分别代表`且`以及`或`。
# 
# 要求：请使用 Logical Indexing实现。

# In[19]:


movie_data[movie_data['popularity'] > 5]


# In[20]:


movie_data[(movie_data['popularity'] > 5) & (movie_data['release_year'] > 1996)]


# ---
# 
# **任务2.3: **分组读取
# 
# 1. 对 `release_year` 进行分组，使用 [`.agg`](http://pandas.pydata.org/pandas-docs/version/0.22/generated/pandas.core.groupby.DataFrameGroupBy.agg.html) 获得 `revenue` 的均值。
# 2. 对 `director` 进行分组，使用 [`.agg`](http://pandas.pydata.org/pandas-docs/version/0.22/generated/pandas.core.groupby.DataFrameGroupBy.agg.html) 获得 `popularity` 的均值，从高到低排列。
# 
# 要求：使用 `Groupby` 命令实现。

# In[21]:


movie_data.groupby('release_year').agg('mean')['revenue']


# In[22]:


movie_data.groupby('director').agg('mean').sort_values('popularity', ascending=False)['popularity']


# ---
# 
# ---
# 
# ## 第三节 绘图与可视化
# 
# 接着你要尝试对你的数据进行图像的绘制以及可视化。这一节最重要的是，你能够选择合适的图像，对特定的可视化目标进行可视化。所谓可视化的目标，是你希望从可视化的过程中，观察到怎样的信息以及变化。例如，观察票房随着时间的变化、哪个导演最受欢迎等。
# 
# <table>
# <thead><tr><th>可视化的目标</th><th>可以使用的图像</th></tr></thead><tbody>
#  <tr><td>表示某一属性数据的分布</td><td>饼图、直方图、散点图</td></tr>
#  <tr><td>表示某一属性数据随着某一个变量变化</td><td>条形图、折线图、热力图</td></tr>
#  <tr><td>比较多个属性的数据之间的关系</td><td>散点图、小提琴图、堆积条形图、堆积折线图</td></tr>
# </tbody></table>
# 
# 在这个部分，你需要根据题目中问题，选择适当的可视化图像进行绘制，并进行相应的分析。对于选做题，他们具有一定的难度，你可以尝试挑战一下～

# **任务3.1：**对 `popularity` 最高的20名电影绘制其 `popularity` 值。

# In[23]:


popularity_movies = movie_data.sort_values('popularity', ascending=False).head(20)


# In[24]:


popularity_movies.describe()['popularity']


# In[25]:


# 单独使用plt画图
bins = np.linspace(start=8.4, stop=33, num=20)
sb.barplot(data=popularity_movies, x='popularity', y='original_title')


# In[26]:


# 使用pandas中plt api画图
popularity_movies.set_index('original_title')['popularity'].plot.barh()
plt.xlabel('popularity')


# ---
# **任务3.2：**分析电影净利润（票房-成本）随着年份变化的情况，并简单进行分析。

# In[27]:


# 计算利润
movie_data['profit'] = movie_data['revenue'] - movie_data['budget']


# In[28]:


# 计算平均值和c
y_means = movie_data.groupby('release_year')['profit'].mean()
y_sems = movie_data.groupby('release_year')['profit'].sem()


# In[29]:


# 使用plt画图
plt.errorbar(x = y_means.index, y = y_means, yerr = y_sems)
plt.xlabel('release_year')
plt.ylabel('profit')


# In[30]:


# 使用pandas中plt api画图
movie_data.groupby('release_year').profit.mean().plot(x = y_means.index, yerr=y_sems)
plt.ylabel('profit')


# 分析：从总体的折线图看，利润并没有表现出明显的上升趋势或者下降趋势，但是在1970和1980年之间，出现了两个峰值散在2005年之后，电影净利润的均方差开始缩小，并趋于稳定，说明不同电影间的利润差别已经没有那么大。

# ---
# 
# **[选做]任务3.3：**选择最多产的10位导演（电影数量最多的），绘制他们排行前3的三部电影的票房情况，并简要进行分析。

# In[31]:


# 前10位导演和作品数
movie_data['director'].value_counts()[:10].plot.barh()


# In[32]:


# 前10位导演
most_movie_dierctor = movie_data['director'].value_counts()[:10].index.tolist()
most_movie_dierctor


# In[33]:


# 分别找到前10位导演的按popularity排名的前3部作品
temp_data = movie_data[movie_data.director.isin(most_movie_dierctor)].sort_values(by='popularity', ascending=False)
result_list = []
for a_dierctor in most_movie_dierctor:
    result_list.append(temp_data.set_index('director').loc[a_dierctor][:3][['revenue', 'original_title']])


# In[34]:


# 为他们的排行的前3部作品分级为 'A', 'B', 'C'
result_data = pd.concat(result_list)
level_values = ['A', 'B', 'C']*(result_data.shape[0] // 3)
result_data['popular_level'] = level_values
result_data[:5]


# In[35]:


# 使用pandas中plt api画图
plt.figure(figsize=(10, 8))
result_data.pivot_table(columns='popular_level', index='director').plot.barh()
plt.xlabel('revenue')


# In[36]:


# 使用seaborn画图
new_data = result_data.reset_index()
sb.barplot(data=new_data, y='director', x='revenue', hue='popular_level')


# 分析：Ron Howard有一部知名度最高并且票房超高的作品；Ridley Scott有一部知名度很高，但是票房比较低的作品，并且其他几位导演的作品也有这样的情况。说明高知名度并不代表高票房。

# ---
# 
# **[选做]任务3.4：**分析1968年~2015年六月电影的数量的变化。

# In[37]:


select_years_data = movie_data[(2015 >= movie_data['release_year']) & (movie_data['release_year'] >= 1968)]
select_years_data['release_date'] = pd.to_datetime(select_years_data['release_date'])
select_years_data[:2]


# In[38]:


# 算出每年月份等于6的电影数量
ret = select_years_data[select_years_data['release_date'].dt.month == 6]['release_date'].value_counts()
ret.index.year.value_counts().sort_index().plot()


# 分析：1968年到2015年6月分的电影数量总体呈现出上升趋势，且2000年之后的电影数量上升幅度很大。

# ---
# 
# **[选做]任务3.5：**分析1968年~2015年六月电影 `Comedy` 和 `Drama` 两类电影的数量的变化。

# In[39]:


# 筛选电影名称
ret_comedy = select_years_data[(select_years_data['release_date'].dt.month == 6) & select_years_data['genres'].str.contains('Comedy')]['release_date'].value_counts()
ret_drama = select_years_data[(select_years_data['release_date'].dt.month == 6) & select_years_data['genres'].str.contains('Drama')]['release_date'].value_counts()

fig = plt.figure()
# 统计最终结果
comedy_count = ret_comedy.index.year.value_counts().sort_index()
drama_count = ret_drama.index.year.value_counts().sort_index()
# 更改名称
comedy_count.name = 'Comedy'
drama_count.name = 'Drama'
# 画图
comedy_count.plot()
drama_count.plot()
plt.ylabel('nums')
plt.legend()


# 分析：对于每年6月份的Comedy 和 Drama电影， 从2000年开始大量上升，并且Comedy电影的增长数量整体要大于Drama。

# > 注意: 当你写完了所有的代码，并且回答了所有的问题。你就可以把你的 iPython Notebook 导出成 HTML 文件。你可以在菜单栏，这样导出**File -> Download as -> HTML (.html)、Python (.py)** 把导出的 HTML、python文件 和这个 iPython notebook 一起提交给审阅者。
