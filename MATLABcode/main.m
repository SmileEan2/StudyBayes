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
%% 对测井数据进行预测
% 获得测井数据文件名
fileFolder=fullfile('/Users/ean2/Documents/GitHub/StudyBayes/Data/Txt_Litho_new');
dirOutput=dir(fullfile(fileFolder,'*.txt'));
fileNames={dirOutput.name}';
for i = 1:length(fileNames)
dataTest = importdata(['/Users/ean2/Documents/GitHub/StudyBayes/Data/Txt_Litho_new/',...
                        fileNames{i}]);              % 循环导入测井数据
dataTest.data(:,7) = log10(dataTest.data(:,7));    % 对RLLD进行log10处理
dataTest.feature = dataTest.data(:,2:7);           % 特征数据
dataTest.Label = dataTest.data(:,8);               % 标签数据
labelPred_kde    = MyBayesPredict(model_kde,dataTest.feature,dataTest.Label);
labelPred_ksd    = MyBayesPredict(model_ksd,dataTest.feature,dataTest.Label);
labelPred_GauVar = MyBayesPredict(model_GauVar,dataTest.feature,dataTest.Label);
labelPred_GauCov = MyBayesPredict(model_GauCov,dataTest.feature,dataTest.Label);
labelPred_GMM3   = MyBayesPredict(model_GMM3,dataTest.feature,dataTest.Label);
i
RightRate(i,:) = [labelPred_kde.score,   labelPred_ksd.score,...
                  labelPred_GauVar.score,labelPred_GauCov.score,...
                  labelPred_GMM3.score]; % 存在问题：labelPred_ksd.score不正确
end
% 对结果进行储存
columnsCell = {'WellName','kde','ksd','GauVar','GauCov','GMM3'};
xlswrite('/Users/ean2/Documents/GitHub/StudyBayes/Data/PredictScore.xlsx',...
          columnsCell,'A1:F1');
xlswrite('/Users/ean2/Documents/GitHub/StudyBayes/Data/PredictScore.xlsx',...
         fileNames,['A2:A',num2str(1+length(fileNames))]);
xlswrite('/Users/ean2/Documents/GitHub/StudyBayes/Data/PredictScore.xlsx',...
         RightRate,'B2'); %这里出现问题：警告：无法启动Excel服务器
%% plot1:
% 进行数据的概率密度函数估计，比较kde.m函数与ksdensity函数
for i = 1:6
   [bandwidth,density,xmesh,cdf] = kde(Data.Feature(:,i),2^10); 
   [f,xi] = ksdensity(Data.Feature(:,i));
   %figure()
   subplot(6,2,(i-1)*2+1);plot(xmesh,density);
   title([Data.textdata{i+1},' kde']);
   set(gca,'xlim',[min(xi),max(xi)]);
   set(gca,'ylim',[0,max(f)+max(f)/4])
   
   subplot(6,2,i*2);plot(xi,f);
   title([Data.textdata{i+1},' ksdensity']);
   set(gca,'xlim',[min(xi),max(xi)]);
   set(gca,'ylim',[0,max(f)+max(f)/4])
end
%% plot2
% 通过上一步，决定采用ksdensity 即可决定属性的概率密度函数的分布
% 对每种属性进行概率密度估计
color={'r','g','b','c','m','y','k'}; %颜色分别为红、绿、蓝、蓝绿、紫红、黄、黑
% 设置图像X轴的显示范围
Xlim = [140,220;-2,25;2.4,2.95;-5,100;2.5,5.8;0.5,5];
for feature = 1:6 % 6种属性
    subplot(3,2,feature)
    for label = 0:6 % 7种类别，编号为0~6
        [f1,xi1] = ksdensity(Data.Feature(Data.Label==label,feature));
        [bandwidth,f,xi,cdf] = kde(Data.Feature(Data.Label==label,feature),2^10);
        hold on
        plot(xi,f,[color{label+1},'--'],'LineWidth',1.2);
        hold on
        plot(xi1,f1,[color{label+1},'-'],'LineWidth',1.2);
    end
    hold off
    
    %set(gca,'xlim',[min(Data.Feature(:,feature)),max(Data.Feature(:,feature))]);
    set(gca,'xlim',Xlim(feature,:))
    title([Data.textdata{feature+1},' kde VS ksdensity'])
end
legend('kde石灰岩','kse石灰岩','kde云灰岩','ksd云灰岩','kde泥灰岩','ksd泥灰岩',...
        'kde白云岩','ksd白云岩','kde灰云岩','ksd灰云岩','kde泥云岩','ksd泥云岩','kde泥岩','ksd泥岩');
%% plot3
clc
k = 0;
N = [1,1;1,4;5,2;7,4];
ClassName = {'石灰岩','云灰岩','泥灰岩','白云岩','灰云岩','泥云岩','泥岩'};
FeatureName = {'AC','CNL','DEN','GR','PE','RLLD'};
P1 = model_GauCov.gaussParam;
P3 = model_GMM3.gaussParam;
%for k = 1:4
for i = 1:7 % 7个类别
    for j = 1:6 % 6个属性
        %i = N(k,1);
        %j = N(k,2);
        k = k +1;
        subplot(7,6,k)
        % 概率密度曲线
        [f,xi] = ksdensity(Data.Feature(Data.Label==(i-1),j));
        %[bandwidth,f,xi,cdf] = kde(Data.Feature(Data.Label==(i-1),j),2^10); 
        plot(xi,f,'k-','LineWidth',1);
        hold on
        % GMM3 曲线
        pd1 = makedist('Normal',P3.Means(1,j,i),sqrt(P3.Covariance(j,j,1,i)));
        pd2 = makedist('Normal',P3.Means(2,j,i),sqrt(P3.Covariance(j,j,2,i)));
        pd3 = makedist('Normal',P3.Means(3,j,i),sqrt(P3.Covariance(j,j,3,i)));
        
        f1 = P3.weight(i,1)*pdf(pd1,xi);
        f2 = P3.weight(i,2)*pdf(pd2,xi);
        f3 = P3.weight(i,3)*pdf(pd3,xi);
        F = f1+f2+f3;
        
        plot(xi,F,'r-','LineWidth',1);hold on;
        plot(xi,f1,'r--');hold on;
        plot(xi,f2,'r--');hold on;
        plot(xi,f3,'r--');hold on;
        
        % Gaussian 曲线
        pd = makedist('Normal',P1.mu(i,j),sqrt(P1.sigma(j,j,i)));
        ff = pdf(pd,xi);
        plot(xi,ff,'b-');hold off;
        xlabel([ClassName{j},' ',FeatureName{j}]);     
    end
end
%end
%% plot4
% 不同概率密度函数（kde ksd GauVar GauCov GMM3）预测结果对比
% 导入score.xlsx
clc
Score = importdata('/Users/ean2/Documents/GitHub/StudyBayes/Data/Score.xlsx');
% kde vs ksd
subplot(3,1,1)
ScoreSort1 = sortrows(Score.data,1); % 按照ksd升序排列
plot(1:89,ScoreSort1(:,1),'k-','LineWidth',1);hold on;
plot(1:89,ScoreSort1(:,2),'r-','LineWidth',1);hold off;
legend('kde','ksd')
legend('Location','southeast') % 图例设置在右下角
ylabel('RightRate%')
% GauVar vs GauCov
subplot(3,1,2)
ScoreSort3 = sortrows(Score.data,3); % 按照GauVar升序排列
plot(1:89,ScoreSort3(:,3),'k-','LineWidth',1);hold on;
plot(1:89,ScoreSort3(:,4),'r-','LineWidth',1);hold off;
legend('GauVar','GauCov')
legend('Location','southeast') 
ylabel('RightRate%')
% kde Vs GauCov Vs GMM3
subplot(3,1,3)
ScoreSort4 = sortrows(Score.data,4); % 按照GauCov升序排列
plot(1:89,ScoreSort4(:,1),'b-','LineWidth',1);hold on;
plot(1:89,ScoreSort4(:,4),'k-','LineWidth',1);hold on;
plot(1:89,ScoreSort4(:,5),'r-','LineWidth',1);hold off;
legend('ksd','GauCov','GMM3')
legend('Location','southeast')
ylabel('RightRate%')
