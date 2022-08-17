
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from jieba import posseg as peg
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn import metrics
from sklearn.cluster import AffinityPropagation

#加载停用词
# stopwords=pd.read_csv('D://input_py//day07//stopwords.txt',index_col=False,quoting=3,sep="\t",names=['stopword'], encoding='utf-8')
# stopwords=stopwords['stopword'].values

#删除语料的nan行
# laogong_df.dropna(inplace=True)
class ClusterJob:

    def __init__(self):
        self.datalist,df=self.read_data()
        self.stopw_dict=self.stopw()

    def merge_data(self,df):
        data = pd.read_excel('res_k12.xlsx')
        data['岗位名称']=data['content']
        df=pd.merge(df,data,how='left')
        print(df)
        df.to_excel('k12.xlsx',index=False)
        # print(data)
    def w_weight(self,word):
        '''
        給特徵加權重
        :return:
        '''
        # 給特贈加權重
        title = ['中级', '中高级', '初级', '后端', '工程师', '开发', '架构师', '研发', '组长', '资深', '软件', '高级']
        title_w = .8
        content_w = .2

        title_v = 1 / len(title) * title_w
        content_v = 1 / (len(word) - len(title)) * content_w

        w = [content_v] * len(word)
        for i in range(len(w)):
            if word[i] in title:
                w[i] = title_v

        self.weight = self.weight * w



    def start(self):
        # self.sentences = []
        # self.cut_text(self.datalist, self.sentences)  # 分词
        # print(self.sentences)

        # 向量化
        # self.weight,word = self.word_vector()



        # 給特贈加權重
        # self.w_weight(word)


        # 降维
        # self.dim_reduction()

        # 层次聚类
        # self.hier_clu(self.weight)

        # kmeans调参
        # self.k_select()
        # kmeans聚类
        # self.kmeans(13,save=True)#titile:5,content:8

        # DBSCAN
        # self.dbscan(save=True,min_samples=2,eps=0.551)#title
        # self.dbscan(save=True,min_samples=2,eps=0.001)#content

        # DBSCAN调参
        # self.dbscan_parameter()

        # 谱聚类
        # simbert相似度
        from sbert.s_matrix import SimBert
        dlist = self.datalist
        simdf = SimBert(dlist).sim_matrix()

        # 调参
        # self.spectral_cluster_para(simdf.values)
        # self.spectral_cluster(simdf.values,7,save=True)


        # AP相似度聚类
        self.AP_cluster(simdf.values,save=True)


    def clear_jobname(self,s):
        '''
        清洗岗位名称:
        Java工程师（后端开发）【初级】【珠海】-->java工程师

        去字符里面的空格，清洗后从新去重
        '''
        s=str(s).lower()
        s = re.sub(r'（.*?）', '', s)#去掉括号
        s = re.sub(r'【.*?】', '', s)
        s = re.sub(r'\(.*?\)', '', s)
        s=s.split('-')[0]
        s=''.join([i for i in s if i!=' '])

        return s

    def clear_jd(self,s):
        for i in ['任职要求', '任职条件', '任职资格', '岗位要求', '技能要求','招聘需求','职位要求','人员要求',' 任职需求','岗位基本需求',
                  '任职基本要求','任职标准','工作要求','工作 要求','工作职责','招聘要求','岗位需求','requirements','responsibilities','要求：','希望你',' 要求:']:
            if i in s:
                return s.split(i)[-1].replace('：','').strip().split('薪酬福利')[0]

        return ''

    def read_data(self):
        '''
        三个级别下的：
        教育培训	教务/教学管理	教务管理
        :return:
        '''
        # 加载语料
        data = pd.read_excel('job_test.xlsx')
        '''

        # 教务管理        df=data[(data['所属行业']=='教育培训')& (data['岗位分类']=='教务/教学管理')& (data['岗位分类.1']=='教务管理')]
        # print(data.columns)
        df=data[(data['所属行业']=='房地产、建筑业')& (data['岗位分类']=='建筑工程')& (data['岗位分类.1']=='电气工程')]
        # print(df)
        # print(df['岗位分类.1'].value_counts())

        df['岗位名称']=df['岗位名称'].map(lambda x:self.clear_jobname(x))
        df['岗位名称_l']=df['岗位名称'].map(lambda x:len(str(x)))
        df=df[df['岗位名称_l']>1]
        df['任职要求'] = df['职位描述+任职要求'].map(lambda x: self.clear_jd(str(x).lower()))
        df['任职要求'].dropna()

        df['content']=df['岗位名称']+df['任职要求']

        # df = df[df['岗位名称'].str.contains('java')]
        '''

        # java
        # 清洗任职要求
        data['任职要求']=data['职位描述+任职要求'].map(lambda x:self.clear_jd(str(x).lower()))
        # laogong_df.to_excel('jd.xlsx',index=False)

        # 清洗岗位名称
        data['岗位名称']=data['岗位名称'].map(lambda x:self.clear_jobname(x))

        #兩個字段一起
        data['content']=data['岗位名称']+data['任职要求']

        df=data[data['岗位名称'].str.contains('java')]
        df['任职要求'].dropna()
        df=df[df['任职要求']!='']
        df.to_excel('a.xlsx',index=False)

        #转换使用不同的字段：岗位名称、content/任职要求
        java_df = list(set(df['岗位名称'].tolist()))
        print('样本量：',len(java_df))

        return java_df,df

    def stopw(self):
        words=[line.replace('\n','')for line in open('stop_word.txt',encoding='utf-8').readlines()]
        return dict(zip(words,range(len(words))))

    # 定义分词函数preprocess_text
    def cut_text(self,content_lines, sentences):
        for line in content_lines:
            try:
                segs=[w for w,flag in peg.cut(line)  if (flag not in ['p','r','d','eng']) and (w not in self.stopw_dict)]#詞性過濾,停用词
                # segs=jieba.lcut(line)#数字？
                segs = list(filter(lambda x:x.strip(), segs))   #去左右空格
                segs = list(filter(lambda x:len(x)>1, segs)) #长度为1的字符
                # segs = list(filter(lambda x:x not in self.stopw_dict, segs)) #去掉停用词
                # segs = [v for v in segs if not str(v).isdigit()]#去数字
                sentences.append(" ".join(segs))
            except Exception:
                print(line)
                continue

    def word_vector(self):
        '''
        词向量化
        当min_df=1时只剩下12个词
        '''
        # 将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i文本下的词频
        # min_df = 10(职责)
        vectorizer = TfidfVectorizer(sublinear_tf=True,max_df=.9,min_df=2)

        # 第一个fit_transform是计算tf-idf 第二个fit_transform是将文本转为词频矩阵
        tfidf = TfidfTransformer().fit_transform(vectorizer.fit_transform(self.sentences))
        # 获取词袋模型中的所有词语
        word = vectorizer.get_feature_names()
        # 将tf-idf矩阵抽取出来，元素w[i][j]表示j词在i文本中的tf-idf权重
        weight = tfidf.toarray()

        # 查看詞典
        print(word)
        print(len(word))
        return weight,word

    def AP_cluster(self,X,save=False):
        '''
        AP相似度矩阵聚类
        :param X:
        :param save:
        :return:
        '''
        clu_res = AffinityPropagation(affinity='precomputed').fit_predict(X) # 设置damping : 阻尼系数，取值[0.5,1)
        print('AP聚类结果类别数：',len(np.unique(clu_res)))  # 类别

        if save:
            self.save_res(clu_res)  # 保存聚类结果
        return clu_res



    def spectral_cluster_para(self,X):

        calinski_harabasz_score_list=[]
        xscale = range(2, 30)
        for k in xscale:
            # for index, k in enumerate((3, 4, 5, 6)):
            y_pred = self.spectral_cluster(X,k)
            calinski_harabasz_score_list.append(metrics.calinski_harabasz_score(X, y_pred))
        self.plot_line(xscale,calinski_harabasz_score_list,'CH')




    def spectral_cluster(self,X,k,save=False):
        '''
        谱聚类：相似度矩阵聚类
        X:相似度矩阵
        k:簇数
        '''



        clu_res=SpectralClustering(affinity='precomputed', n_clusters=k).fit_predict(X)

        if save:
            self.save_res(clu_res)  # 保存聚类结果
        return clu_res



    def dbscan(self,min_samples,eps,save=False):
        '''
        密度聚类
        '''
        model = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean', algorithm='auto')

        model.fit(self.weight)

        clu_res=model.labels_
        if save:
            self.save_res(clu_res)  # 保存聚类结果

        return  clu_res, model
    def dbscan_parameter(self):
        '''
        迭代不同的值來調參數
        找參數的突變點
        '''
        # 构建空列表，用于保存不同参数组合下的结果
        res = []
        # 迭代不同的eps值
        for eps in np.arange(0.001, 1, 0.05):
            # 迭代不同的min_samples值
            for min_samples in range(2, 10):
                # dbscan = cluster.DBSCAN(eps=eps, min_samples=min_samples)
                labels, clf = self.dbscan(save=False, min_samples=min_samples, eps=eps)
                # 统计各参数组合下的聚类个数（-1表示异常点）
                n_clusters = len([i for i in set(labels) if i != -1])
                # 异常点的个数
                outliners = np.sum(np.where(labels == -1, 1, 0))
                raito = len(labels[labels[:] == -1]) / len(labels)  # 计算噪声点个数占总数的比例

                # 轮廓系数
                try:
                    k = silhouette_score(self.weight, labels)  # 轮廓系数评价聚类的好坏，值越大越好
                except:
                    k=0
                # 统计每个簇的样本个数
                stats = str(pd.Series([i for i in labels if i != -1]).value_counts().values)
                res.append({'eps': eps, 'min_samples': min_samples, 'n_clusters': n_clusters, 'outliners': outliners,
                            'stats': stats,'k':k,'raito':raito})
        # 将迭代后的结果存储到数据框中
        df = pd.DataFrame(res)
        df.to_excel('c1.xlsx',index=False)
        # plt.figure()
        # sns.relplot(x="eps", y="min_samples", size='k', data=df)
        # sns.relplot(x="eps", y="min_samples", size='raito', data=df)
        # plt.show()

    def dim_reduction(self):
        '''
        dimensionality reduction
        '''
        pca = PCA(n_components=10)  # 降维10
        self.weight = pca.fit_transform(self.weight)  # 载入N维
        print('保留的成分：',pca.explained_variance_ratio_)#所保留的n个成分各自的方差百分比。
        # print(pca.components_ )
        print('保留的成分所占的百分比和:',sum(pca.explained_variance_ratio_))

    def kmeans(self,numClass,save=False):
        '''
        聚类并保存结果
        '''
        clf = KMeans(n_clusters=numClass, max_iter=10000, init="k-means++", tol=1e-6)  #这里也可以选择随机初始化init="random"
        s = clf.fit(self.weight)

        clu_res=list(clf.predict(self.weight))
        if save:
            self.save_res(clu_res)#保存聚类结果
        return clu_res,clf

    def hier_clu(self,data_array):
        '''
        层次聚类：主要是探索性数据分析，可视化看数据划分
        '''
        Z = linkage(data_array, method='single', metric='euclidean')
        print(Z)
        # 画图
        dn = dendrogram(Z)
        # plt.savefig('plot_dendrogram.png')保存结果
        plt.show()
        # 根据linkage matrix Z得到聚类结果:
        cluster = sch.fcluster(Z, t=1, criterion='inconsistent')
        #看随着聚类次数，距离的变化
        plt.plot(range(0,len(Z),1),Z[:,2])
        plt.show()


    def save_res(self,clu_res):
        res_df = pd.DataFrame(self.datalist, columns=['content'])
        res_df['cluster'] = clu_res
        print(res_df['cluster'].value_counts())

        res_df.to_excel('java_title_AP.xlsx', index=False)

    def k_select(self):
        '''
        K值选取
        '''

        distortions = []  # 簇内误差平方和
        sil_score = []  # 轮廓系数

        # print(self.weight.shape)
        xscale=range(2,30)
        for nclass in xscale:

            clu_res,clf=self.kmeans(nclass)

            distortions.append(clf.inertia_)  # 簇内误差平方和
            sil_score.append(silhouette_score(self.weight, clu_res))  # 轮廓系数

        self.plot_line(xscale,distortions,'distortions')
        self.plot_line(xscale,sil_score,'sil_score')



    def plot_line(self,x_scale,Y,Yname):

        plt.plot(x_scale, Y, marker='x')
        plt.xlabel('Number of clusters')
        plt.title(Yname)
        plt.show()

    # 定义聚类结果可视化函数
    def plot_cluster(self,result,newData,numClass,clf):
        plt.figure(2)
        Lab = [[] for i in range(numClass)]
        index = 0
        for labi in result:
            Lab[labi].append(index)
            index += 1
        color = ['oy', 'ob', 'og', 'cs', 'ms', 'bs', 'ks', 'ys', 'yv', 'mv', 'bv', 'kv', 'gv', 'y^', 'm^', 'b^', 'k^',
                 'g^'] * 3
        for i in range(numClass):
            x1 = []
            y1 = []
            for ind1 in newData[Lab[i]]:
                # print ind1
                try:
                    y1.append(ind1[1])
                    x1.append(ind1[0])
                except:
                    pass
            plt.plot(x1, y1, color[i])

        # 绘制初始中心点
        x1 = []
        y1 = []
        for ind1 in clf.cluster_centers_:
            try:
                y1.append(ind1[1])
                x1.append(ind1[0])
            except:
                pass
        plt.plot(x1, y1, "rv") #绘制中心
        plt.show()



ClusterJob().start()
# ClusterJob()