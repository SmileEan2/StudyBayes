# StudyBayes
主要是将朴素贝叶斯分类器应用到了岩性识别上。数据是研究工区的89口井的M5段测井数据： 
   
* 六中测井曲线：AC CNL DEN GR PE RLLD     
* 7种岩性：石灰岩、白云质灰岩、泥质灰岩、白云岩、灰质白云岩、泥质白云岩、泥岩

## 主要研究内容
朴素贝叶斯分类方法关键点是推断样本的概率密度分布，本次研究主要进行了使用了两种方法：

- 非参数估计：核密度估计([kernel density estimation](https://en.wikipedia.org/wiki/Kernel_density_estimation))，在这里对比了固定带宽与自适应带宽的两种核密度估计方法的效果。MATALB本身自带的核密度估计函数ksdensity为固定带宽的估计方法，自适应带宽的核密度估计函数为[kde](http://cn.mathworks.com/matlabcentral/fileexchange/14034-kernel-density-estimator)。  
- 参数估计：假设样本服从高斯分布，对高斯分布的关键参数均值与方差进行估计，进而得到样本的概率密度函数。但实际任务中，样本恰好符合高斯分布的情况比较少，故而可以采用[GMM](https://en.wikipedia.org/wiki/Mixture_model#Gaussian_mixture_model)方法对样本的真实概率密度分布进行更为准确的拟合。对于多维属性，对比了方差与协方差矩阵作为sigma估计的效果。  

## 存在的问题
ksd的到的结果有问题（应该是编程上的问题），暂时没有时间查找

## 研究成果图  
绘图程序都在main.m中  
![ksd kde 对比图1](https://github.com/SmileEan2/StudyBayes/blob/master/Figures/kdeVSksd.jpg)
![ksd kde 对比图2](https://github.com/SmileEan2/StudyBayes/blob/master/Figures/kdeVSksd2.jpg)
![AttKsdGauGMM3](https://github.com/SmileEan2/StudyBayes/blob/master/Figures/AttKsdGauGMM3.jpg)
![RightRateCompare](https://github.com/SmileEan2/StudyBayes/blob/master/Figures/RightRateCompare.jpg)
