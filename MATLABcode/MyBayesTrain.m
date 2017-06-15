function model=MyBayesTrain(data,label,method)
% ����������ѵ����Ҷ˹ģ�ͣ��˴ν�ѵ����Ԥ����뿪����
% Input��feature  �������� 
%        label   ��ǩ����
%        method  ѵ����ʽ��Ŀǰ��ѡ��ksd kde GuassianVar GuassianCov GMM3
%Output:model ѵ���õ�Bayesģ��
%  ����:model.method
%      model.data
% .    model.label
% .    model.labelNames
% .    model.nlabels
% .    model.densityMatrix         �����ܶȹ��Ʒ�������
%      model.gaussParam            ��˹�������Ʒ�ʽ����
labelNames = unique(label);      % ����label����ʲô,��һ��n*1������
nlabels = size(labelNames,1);    % ���ر�ǩ�����
model.data = data;               % ѵ����������
model.label = label;             % ѵ���ı�ǩ����
model.labelNames = labelNames;
model.nlabels = nlabels;
switch method
    case 'kde'                   % ����Ӧ����ĺ��ܶȹ��ƺ���
        densityMatrix = MyKde(data,label,1);
        model.method = 'kde';
        model.densityMatrix = densityMatrix;
    case 'ksd'                   % �̶�����ĺ��ܶȹ��ƺ���
        densityMatrix = MyKde(data,label,2);
        model.method = 'ksd';
        model.densityMatrix = densityMatrix;
    case 'GaussianVar'           % ���÷���ĸ�˹����
        gaussParam = MyGaussian(data,label,'var');
        model.method = 'GaussianVar';
        model.gaussParam = gaussParam;
    case 'GaussianCov'           % ����Э����ĸ�˹����
        gaussParam = MyGaussian(data,label,'cov');
        model.method = 'GaussianCov';
        model.gaussParam = gaussParam;
    case 'GMM3'                  % 3��˹���ģ��
        gaussParam = MyGMM(data,label,3);
        model.method = 'GMM3';
        model.gaussParam = gaussParam;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function densityM = MyKde(data,label,d)
% ���ܶȹ��ƺ���
labelNames = unique(label);   % ����label����ʲô,��һ��n*1������
nlabels = size(labelNames,1); % ���ر�ǩ�����
[~,nfeatures] = size(data);   % ����data�����Ը���
numPoints = 128;              % ������Ҫ���ظ����ܶȵ�ĸ���
% ���������践�صĲ�����
% fΪxi����λ�õĸ����ܶ�ֵ
% f xi ��Ϊ��ά���飬����ά��1~7��Ӧ��ͬ��ǩ0~6
f = zeros(nfeatures,numPoints,nlabels);
xi = zeros(nfeatures,numPoints,nlabels);
% ѭ������ f xi
if d == 1                     % ����Ӧ����ĺ��ܶȹ��ƺ���
for i = 1:nlabels             % i��Ӧ��ͬ��label
   for j = 1:nfeatures        % j=1~6�ֱ��ӦAC CNL DEN GR PE RLLD
       [~,ff,xxi,~] = kde(data(label==labelNames(i),j),numPoints); 
       f(j,:,i) = ff+0.000000001;
       xi(j,:,i) = xxi;
   end
end
end
if d == 2                     % �̶�����ĺ��ܶȺ���
    numPoints = 100;
    f = zeros(nfeatures,numPoints,nlabels);
    xi = zeros(nfeatures,numPoints,nlabels);
for i = 1:nlabels             % i��Ӧ��ͬ��label��label=labelNames(i)
   for j = 1:nfeatures        % j=1~6�ֱ��ӦAC CNL DEN GR PE RLLD
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
% ���õ���˹�������������ĸ����ܶȺ���
% ���ڸ�˹�ܶȺ����еĴ�������ַ�ʽ���й��ƣ�
%    1. ����
%    2. Э����
labelNames = unique(label);   % ����label����ʲô,��һ��n*1������
nlabels = size(labelNames,1); % ���ر�ǩ�����
[~,nfeatures] = size(data);   % ����data�����Ը���
mu = zeros(nlabels,nfeatures);
if strcmp(sigmaM,'cov')       % ���� 
    for i = 1:nlabels
        GMModel = fitgmdist(data(label==labelNames(i),:),1);
        mu(i,:) = GMModel.mu;
        sigma(:,:,i) = GMModel.Sigma;
    end
end
if strcmp(sigmaM,'var')       % Э���� 
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
labelNames = unique(label);        % ����label����ʲô,��һ��n*1������
nlabels = size(labelNames,1);      % ���ر�ǩ�����
%[~,nfeatures] = size(data);       % ����data�������ĸ����������Ը���
options = statset('MaxIter',10000);% �趨fitgmdist��һЩ����
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






