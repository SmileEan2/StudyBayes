% Bayes研究的主体函数
%% 导入训练数据
Data = importdata('/Users/ean2/Documents/GitHub/StudyBayes/Data/Data_train_log10.xlsx');
Data.Feature = Data.data(:,2:7);   % 特征数据
Data.Label = Data.data(:,8);       % 标签数据
%% 训练Bayes模型
% 固定带宽的核密度估计
model_ksd = MyBayesTrain(Data.Feature,Data.Label,'ksd'); 
% 自适应带宽的核密度估计
model_kde = MyBayesTrain(Data.Feature,Data.Label,'kde');
% 方差估计带宽的高斯密度估计
model_GauVar = MyBayesTrain(Data.Feature,Data.Label,'GaussianVar');
% 协方差估计带宽的高斯密度估计
model_GauCov = MyBayesTrain(Data.Feature,Data.Label,'GaussianCov');
% 3高斯混合的密度估计
model_GMM3 = MyBayesTrain(Data.Feature,Data.Label,'GMM3');

