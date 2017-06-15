% Bayes�о������庯��
%% ����ѵ������
Data = importdata('/Users/ean2/Documents/GitHub/StudyBayes/Data/Data_train_log10.xlsx');
Data.Feature = Data.data(:,2:7);   % ��������
Data.Label = Data.data(:,8);       % ��ǩ����
%% ѵ��Bayesģ��
% �̶�����ĺ��ܶȹ���
model_ksd = MyBayesTrain(Data.Feature,Data.Label,'ksd'); 
% ����Ӧ����ĺ��ܶȹ���
model_kde = MyBayesTrain(Data.Feature,Data.Label,'kde');
% ������ƴ���ĸ�˹�ܶȹ���
model_GauVar = MyBayesTrain(Data.Feature,Data.Label,'GaussianVar');
% Э������ƴ���ĸ�˹�ܶȹ���
model_GauCov = MyBayesTrain(Data.Feature,Data.Label,'GaussianCov');
% 3��˹��ϵ��ܶȹ���
model_GMM3 = MyBayesTrain(Data.Feature,Data.Label,'GMM3');

