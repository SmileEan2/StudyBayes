function predict = MyBayesPredict(model,dataTest,labelTest)
% ��������MyBayesTrainѵ���õ�model����Ԥ��
% Input: model        MyBayesTrain ѵ����ģ��
%        dataTest     ����Ԥ�����������
%        labelTest    dataTest��Ӧ����ʵ��ǩ
% Output��predict      ��dataTest��Ԥ����
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
labelPred.method = model.method; % ��ʾ�б�ķ�ʽ
labelPred.score = getRightRate(labelTest,labelPred.y);
predict = labelPred;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function labelPred = MyKdePredict(model,dataTest)
% ���ݺ��ܶȹ��Ʒ����õ����ܶȾ����dataTest����Ԥ��
nlabels = model.nlabels;
nsample = size(dataTest,1);
p = zeros(nsample,nlabels);
for i = 1:nsample
    % Ѱ��һ��data������ÿ��ά�ȶ�Ӧ�������Pksd.Xi
    % Ȼ�����Pksd.Xi��Ѱ�Ҷ�ӦPksd.F
    for j = 1:nlabels
        distance = abs(model.densityMatrix.Xi(:,:,j)-dataTest(i,:)');
        [~,I] = min(distance,[],2); % ����ÿһ�е���Сֵ��λ����Ϣ
        m = getM(model.densityMatrix.F(:,:,j),1:length(I),I);
        p(i,j) = sum(log(m));       % �����ܶ���Ϣ��ȡLog��Ӽ�Ϊ��ǰ�����ܸ���
    end
end
[~,I_label] = max(p,[],2); % 
for ii = 1:nsample
   label_preding(ii) = model.labelNames(I_label(ii)) ;
end
labelPred.p = p;              % ��ͬ���ĸ���
labelPred.y = label_preding;  % ��dataTest��ÿ��������Ԥ��
% С��������ȡ�����ж�Ӧλ�õ�Ԫ��
function m = getM(M,row_index,line_index)
if length(row_index) == length(line_index)
   for i = 1:length(row_index)
      mm(i)  = M(row_index(i),line_index(i));
   end
else
    error('�С�����������ȣ�')
end
m = mm;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%����GMM�����õ������ܶȹ��������б�
function labelPred = MyGaussPredict(model,dataTest)
% model.gaussParam�д洢��GMMģ�Ͷ�Ӧ��mu sigma
nlabels = model.nlabels;
[nsample,~] = size(dataTest);
p = ones(nsample,nlabels);
if strcmp(model.method,'GaussianCov')
for i = 1:nsample
    for j = 1:nlabels
        pp = log(mvnpdf(dataTest(i,:),model.gaussParam.mu(j,:),...
            model.gaussParam.sigma(:,:,j)));%���ܵ�sigma����ΪЭ�������
        p(i,j) = pp;
    end
end
end
if strcmp(model.method,'GaussianVar') 
for i = 1:nsample
   for j = 1:nlabels
       %normpdf �е�mu sigmaΪ����ʱ��ÿһ��mu sigma������һ����˹�ֲ�������һ����ֵ
       p(i,j) = sum(log(normpdf(dataTest(i,:),model.gaussParam.mu(j,:),...
                        model.gaussParam.sigma(:,:,j))));
   end
end
end
[~,I_label] = max(p,[],2);
for ii = 1:nsample
   label_preding(ii) = model.labelNames(I_label(ii)) ;
end
labelPred.p = p;              % ��ͬ���ĸ���
labelPred.y = label_preding;  % ��dataTest��ÿ��������Ԥ��
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function labelPred = MyGMMPredict(model,dataTest)
% model.gaussParam�д洢��GMM3ģ�Ͷ�Ӧ��weight mu sigma
nlabels = model.nlabels;
[nsample,~] = size(dataTest);
p = ones(nsample,nlabels);
M = model.gaussParam; % �������ַ�����д���Ķ�
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
% ����labelTest labelPred.y ������ȷ�ʼ���
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