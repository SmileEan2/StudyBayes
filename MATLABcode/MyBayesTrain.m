function model=MyBayesTrain(data,label,method)
% 本函数用于训练贝叶斯模型（此次将训练与预测分离开来）
% Input：feature  特征数据 
%        label   标签数据
%        method  训练方式，目前可选择：ksd kde GuassianVar GuassianCov GMM3
%Output:model 训练好的Bayes模型
%  包括:model.method
%      model.data
% .    model.label
% .    model.labelNames
% .    model.nlabels
% .    model.densityMatrix         概率密度估计方法所有
%      model.gaussParam            高斯函数估计方式所有
labelNames = unique(label);      % 返回label中有什么,是一个n*1的数组
nlabels = size(labelNames,1);    % 返回标签类别数
model.data = data;               % 训练特征数据
model.label = label;             % 训练的标签数据
model.labelNames = labelNames;
model.nlabels = nlabels;
switch method
    case 'kde'                   % 自适应带宽的核密度估计函数
        densityMatrix = MyKde(data,label,1);
        model.method = 'kde';
        model.densityMatrix = densityMatrix;
    case 'ksd'                   % 固定带宽的核密度估计函数
        densityMatrix = MyKde(data,label,2);
        model.method = 'ksd';
        model.densityMatrix = densityMatrix;
    case 'GaussianVar'           % 采用方差的高斯估计
        gaussParam = MyGaussian(data,label,'var');
        model.method = 'GaussianVar';
        model.gaussParam = gaussParam;
    case 'GaussianCov'           % 采用协方差的高斯估计
        gaussParam = MyGaussian(data,label,'cov');
        model.method = 'GaussianCov';
        model.gaussParam = gaussParam;
    case 'GMM3'                  % 3高斯混合模型
        gaussParam = MyGMM(data,label,3);
        model.method = 'GMM3';
        model.gaussParam = gaussParam;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function densityM = MyKde(data,label,d)
% 核密度估计函数
labelNames = unique(label);   % 返回label中有什么,是一个n*1的数组
nlabels = size(labelNames,1); % 返回标签类别数
[~,nfeatures] = size(data);   % 返回data中属性个数
numPoints = 128;              % 设置需要返回概率密度点的个数
% 设置最终需返回的参数：
% f为xi所在位置的概率密度值
% f xi 均为三维数组，第三维度1~7对应不同标签0~6
f = zeros(nfeatures,numPoints,nlabels);
xi = zeros(nfeatures,numPoints,nlabels);
% 循环计算 f xi
if d == 1                     % 自适应带宽的核密度估计函数
for i = 1:nlabels             % i对应不同的label
   for j = 1:nfeatures        % j=1~6分别对应AC CNL DEN GR PE RLLD
       [~,ff,xxi,~] = kde(data(label==labelNames(i),j),numPoints); 
       f(j,:,i) = ff+0.000000001;
       xi(j,:,i) = xxi;
   end
end
end
if d == 2                     % 固定带宽的核密度函数
    numPoints = 100;
    f = zeros(nfeatures,numPoints,nlabels);
    xi = zeros(nfeatures,numPoints,nlabels);
for i = 1:nlabels             % i对应不同的label，label=labelNames(i)
   for j = 1:nfeatures        % j=1~6分别对应AC CNL DEN GR PE RLLD
       [~,ff,xxi,~] = ksdensity(data(label==labelNames(i),j),'NumPoints',numPoints); 
       f(j,:,i) = ff+0.000000001;
       xi(j,:,i) = xxi;
   end
end
end
    
densityM.F = f;
densityM.Xi = xi;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function gaussParam=MyGaussian(data,label,sigmaM)
% 采用单高斯函数估计样本的概率密度函数
% 对于高斯密度函数中的带宽分两种方式进行估计：
%    1. 方差
%    2. 协方差
labelNames = unique(label);   % 返回label中有什么,是一个n*1的数组
nlabels = size(labelNames,1); % 返回标签类别数
[~,nfeatures] = size(data);   % 返回data中属性个数
mu = zeros(nlabels,nfeatures);
if strcmp(sigmaM,'cov')       % 方差 
    for i = 1:nlabels
        GMModel = fitgmdist(data(label==labelNames(i),:),1);
        mu(i,:) = GMModel.mu;
        sigma(:,:,i) = GMModel.Sigma;
    end
end
if strcmp(sigmaM,'var')       % 协方差 
   for i = 1:nlabels
       for j = 1:nfeatures
          mu(i,j) = mean(data(label==labelNames(i),j));
          sigma(:,j,i) = std(data(label==labelNames(i),j));
       end
   end 
end
gaussParam.mu = mu;
gaussParam.sigma = sigma;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function gaussParam = MyGMM(data,label,num)
labelNames = unique(label);        % 返回label中有什么,是一个n*1的数组
nlabels = size(labelNames,1);      % 返回标签类别数
%[~,nfeatures] = size(data);       % 返回data中样本的个数及其属性个数
options = statset('MaxIter',10000);% 设定fitgmdist的一些参数
GMMOK = 1;
while GMMOK == 1    
try
for i = 1:nlabels
    GMModl = fitgmdist(data(label==labelNames(i),:),num,'Options',options) ;
    P.weight(i,:) = GMModl.ComponentProportion;
    P.Means(:,:,i) = GMModl.mu;
    P.Covariance(:,:,:,i) = GMModl.Sigma;
end
GMMOK = 0;
catch 
    disp('There was an error !')
    GMMOK = 1;
end
end

gaussParam = P;






