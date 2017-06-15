function predict = MyBayesPredict(model,dataTest,labelTest)
% 本函数将MyBayesTrain训练好的model用于预测
% Input: model        MyBayesTrain 训练的模型
%        dataTest     用于预测的特征数据
%        labelTest    dataTest对应的真实标签
% Output：predict      对dataTest的预测结果
switch model.method
    case 'ksd'
        labelPred = MyKdePredict(model,dataTest);
    case 'kde'
        labelPred = MyKdePredict(model,dataTest);
    case 'GaussianVar'
        labelPred = MyGaussPredict(model,dataTest);
    case 'GaussianCov'
        labelPred = MyGaussPredict(model,dataTest);
    case 'GMM3'
        labelPred = MyGMMPredict(model,dataTest);
end
labelPred.method = model.method; % 显示判别的方式
labelPred.score = getRightRate(labelTest,labelPred.y);
predict = labelPred;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function labelPred = MyKdePredict(model,dataTest)
% 根据核密度估计方法得到的密度矩阵对dataTest进行预测
nlabels = model.nlabels;
nsample = size(dataTest,1);
p = zeros(nsample,nlabels);
for i = 1:nsample
    % 寻找一条data数据中每个维度对应的最近的Pksd.Xi
    % 然后根据Pksd.Xi来寻找对应Pksd.F
    for j = 1:nlabels
        distance = abs(model.densityMatrix.Xi(:,:,j)-dataTest(i,:)');
        [~,I] = min(distance,[],2); % 返回每一行的最小值的位置信息
        m = getM(model.densityMatrix.F(:,:,j),1:length(I),I);
        p(i,j) = sum(log(m));       % 概率密度信息，取Log相加即为当前类别可能概率
    end
end
[~,I_label] = max(p,[],2); % 
for ii = 1:nsample
   label_preding(ii) = model.labelNames(I_label(ii)) ;
end
labelPred.p = p;              % 不同类别的概率
labelPred.y = label_preding;  % 对dataTest中每个样本的预测
% 小函数：提取矩阵中对应位置的元素
function m = getM(M,row_index,line_index)
if length(row_index) == length(line_index)
   for i = 1:length(row_index)
      mm(i)  = M(row_index(i),line_index(i));
   end
else
    error('行、列数量不相等！')
end
m = mm;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%　将GMM方法得到概率密度估计用于判别
function labelPred = MyGaussPredict(model,dataTest)
% model.gaussParam中存储中GMM模型对应的mu sigma
nlabels = model.nlabels;
[nsample,~] = size(dataTest);
p = ones(nsample,nlabels);
if strcmp(model.method,'GaussianCov')
for i = 1:nsample
    for j = 1:nlabels
        pp = log(mvnpdf(dataTest(i,:),model.gaussParam.mu(j,:),...
            model.gaussParam.sigma(:,:,j)));%接受的sigma参数为协方差矩阵
        p(i,j) = pp;
    end
end
end
if strcmp(model.method,'GaussianVar') 
for i = 1:nsample
   for j = 1:nlabels
       %normpdf 中当mu sigma为向量时，每一对mu sigma都产生一个高斯分布，返回一个数值
       p(i,j) = sum(log(normpdf(dataTest(i,:),model.gaussParam.mu(j,:),...
                        model.gaussParam.sigma(:,:,j))));
   end
end
end
[~,I_label] = max(p,[],2);
for ii = 1:nsample
   label_preding(ii) = model.labelNames(I_label(ii)) ;
end
labelPred.p = p;              % 不同类别的概率
labelPred.y = label_preding;  % 对dataTest中每个样本的预测
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function labelPred = MyGMMPredict(model,dataTest)
% model.gaussParam中存储中GMM3模型对应的weight mu sigma
nlabels = model.nlabels;
[nsample,~] = size(dataTest);
p = ones(nsample,nlabels);
M = model.gaussParam; % 换个名字方便书写与阅读
for i = 1:nsample
   for j = 1:nlabels
      p(i,j) = log(M.weight(j,1)*mvnpdf(dataTest(i,:),M.Means(1,:,j),M.Covariance(:,:,1,j))+...
                   M.weight(j,2)*mvnpdf(dataTest(i,:),M.Means(2,:,j),M.Covariance(:,:,2,j))+...
                   M.weight(j,3)*mvnpdf(dataTest(i,:),M.Means(3,:,j),M.Covariance(:,:,3,j)));
   end
end
[~,I_label] = max(p,[],2);
for ii = 1:nsample
   label_preding(ii) = model.labelNames(I_label(ii)) ;
end
labelPred.p = p;
labelPred.y = label_preding';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 根据labelTest labelPred.y 进行正确率计算
function score = getRightRate(label_true,label_pred)
n_true = length(label_true);
if n_true == length(label_pred)
    %disp('Take getRightRate')
    k = 0;
    for i = 1:n_true
        if label_pred(i) == label_true(i)
            k = k+1;
        end
        
    end
else
   error('n_true != n_pred !!!') ;
end
score = k / n_true;