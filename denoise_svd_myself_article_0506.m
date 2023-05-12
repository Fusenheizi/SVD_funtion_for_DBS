% 0506_文章重新作图，修改fig图的字体字号
% function [data_after_pca,show_channel_denoise] = denoise_svd_myself_forapp(t_ROI,data_to_deal,para)
% data_after_pca: 降噪后的数据
% art_mean: 降噪过程中伪迹平均图像
% t 单伪迹的
tri =1;
% para = para_now;
%     data_to_deal = dead_data;
%     t_ROI = t_ROI_now;
para.sample_rate = 30000; % 设置采样率
para.art_thre = 500;   % 设置伪迹识别阈值
para.re_overlay = 1.5; % 设置识别伪迹的范围系数
% para.likewise_PCA = 0.01;  % 设置PCA需要舍弃的相似度-80Hz,0.01
para.likewise_PCA = 1;  % 设置PCA需要舍弃的相似度-160Hz,0.01
% para.likewise_PCA = 0.005;  % 设置PCA需要舍弃的相似度
para.tf_plot = 1;
para.precut = 400;
para.cutprct = 40;
art_thre = 5;% 确定阈值
re_overlay = para.re_overlay;% 设置识别伪迹的范围系数
k = para.likewise_PCA; % 设置主成分分析需要舍弃的相似度(奇异值最大）
tf_plot = para.tf_plot;
data_to_deal = rhs_data_set.trial{tri}(1,:);

%% 运行完read——Intan之后，提取某个通道的数据

%     data_to_deal = rhs_data_set.trial{1,1}(1,:);
data_roi_origin = data_to_deal';% 导入数据
%   art_thre = 200;% 确定阈值
%   re_overlay = 0.4;% 设置识别伪迹的范围系数
% 截取数据提前量
pre_cut_index = para.precut;
cut_prct = para.cutprct; % 默认切多少波宽




%   likewise_PCA = 0.005; % 设置主成分分析需要舍弃的相似度
%% 寻找异常值（明显刺激伪迹，并且利用伪迹推算刺激频率）
diff_data = diff(data_roi_origin);%将原始信号信号data_roi_block差分
change_p = abs(diff_data)>art_thre*std(abs(diff_data),0,'all');%找到差分值超过阈值的全部索引点的逻辑值
find_change_p = find(change_p);%找到上述逻辑值的索引
diff_find_change = diff(find_change_p);%再一次差分，找到变化较大的索引序列

%借助最大值的一半，推算这些最大值的位置，即平均变化值的位置
gap = floor(prctile(diff_find_change,98));

sample_rate = para.sample_rate;%已知采样率
%     freq2get = floor(sample_rate/gap);%通过间隔，得到人工伪迹推算刺激频率，取整数

%% 截取每段伪迹的起始点放入cut_data_bg_p和cut_data_ed_p
% 1122-重新推算伪迹截取点
st_p_indx = true;%第一个点是起始点
ed_p_indx = (diff_find_change>(gap/2) & diff_find_change<(1.5*gap));%找到剩余的间隔点（即伪迹的最小后边界）
st_p_indx = [st_p_indx;ed_p_indx];% 将所有逻辑索引后移一个索引点，首位相接
change_p_indx = find_change_p(st_p_indx); % 找到数据变化值最为强烈的数据点索引位置

%% 为每一个刺激伪迹片段规定开始索引和结束索引
cut_data_bg_p = change_p_indx-pre_cut_index ;%每个尾迹的数据从平均gap的开头
cut_data_ed_p = min(change_p_indx+floor(1.2*gap)-pre_cut_index,length(data_roi_origin));% 整个间隔
arti_pulse_t = (0:floor(1.2*gap))./sample_rate;

%     %  提取第一个刺激伪迹并画图（此小结可删除）
%     arti_data_cuttest = data_roi_origin(cut_data_bg_p(3):cut_data_ed_p(3));
%     arti_t_cuttest = t(cut_data_bg_p(3):cut_data_ed_p(3));
%     figure
%     plot(arti_pulse_t,arti_data_cuttest)

%% 绘制某一个试次下的伪迹的平均图形(单独试次下的伪迹模板)

pusle_count = length(change_p_indx);% 伪迹脉冲的个数，就是change_p_indx的长度
arti_data = zeros(pusle_count,floor(1.2*gap)); % 初始化,为伪迹堆叠数据预留空间

for art_pulse_i = 1:pusle_count
    bg_p = cut_data_bg_p(art_pulse_i);
    ed_p = min(cut_data_ed_p(art_pulse_i),size(data_roi_origin,1));% 防止超出索引
    arti_data(art_pulse_i,1:ed_p-bg_p+1) = ...
        data_roi_origin(bg_p:ed_p);
end % 伪迹脉冲的个数遍历，填充数据，得到所有伪迹的脉冲堆叠

%为每一个脉冲做一个基线矫正（）
art_detr = arti_data;% art_detr 脉冲个数x脉冲采样点
art_mean = mean(art_detr,1);

% 绘图
if tf_plot
    % 伪迹的假定时间和伪迹数据堆叠的时间中挑选较小的采样点）
    p2plot = min(size(arti_pulse_t,2),size(art_detr,2));
    art_mean_all = mean(art_detr,'all') ;%全平均值(临时绘图用）
    figure
    plot(arti_pulse_t(1,1:p2plot),art_detr(:,1:p2plot),'y')% 绘制基线矫正后，捕获的脉冲伪迹堆叠
    hold on
    % plot(arti_pulse_t(1,1:p2plot),detrend(art_mean(:,1:p2plot)),'k');
    plot(arti_pulse_t(1,1:p2plot),art_mean(:,1:p2plot),'r');% 伪迹叠加后的形态
    % plot(arti_pulse_t,art_mean_1,'b');
    % % 画出伪迹的值界限（蓝色虚线）
    plot([arti_pulse_t(1),arti_pulse_t(end)],[art_mean_all,art_mean_all],'--b');% 蓝色虚线是伪迹的平局值
    title('Wave of the artifacts', 'FontSize',13,...
'FontName','Times New Roman'...
)
ylabel('Voltage(\muV)','FontName','Times New Roman', 'FontSize',13);
xlabel('Time(s)','FontName','Times New Roman', 'FontSize',13);
set(gca,'FontSize',18,'Fontname', 'Times New Roman');
end % 绘图


%% 利用SVD消除伪迹偏移
X = arti_data;% 原始数据
%     Y = art_mean-mean(art_mean,2);% 均值
t = arti_pulse_t; % 时间
% 奇异值分解SVD
% 将原始矩阵X进行SVD分解
[U,S,V] = svd(X);
% 将S矩阵中前k个奇异值保留，其余设为0，得到S'矩阵
%     k = 1;  % 保留前1个奇异值（最大的偏移）
Sprime = S;
Sprime(k+1:end,:) = 0;
% 构造近似矩阵X'
Xprime = U * Sprime * V';

% 删除SVD方差最大方向上的数据，得到剩余的数据
X_residual = X - Xprime;

if tf_plot
    % 绘图
    figure
    subplot(2,1,1)
    h1 = plot(t,X,'r');% 原始图
       
    hold on
    h2 = plot(t,X_residual,'b'); % 剩余图
     set(gca,'FontSize',13,'FontName','Times New Roman');
    legend([h1(1),h2(1)],'Origin signal','Signal after SVD','FontSize',13,'FontName','Times New Roman')
    title('Pre/Post SVD of the artifacts', 'FontSize',18,...
'FontName','Times New Roman'...
)
ylabel('Voltage(\muV)','FontName','Times New Roman','FontSize',15);
xlabel('Time(s)','FontName','Times New Roman','FontSize',15);
% set(gca,'FontSize',18,'Fontname', 'Times New Roman');
    subplot(2,1,2)
    
        plot(t,Xprime,'g');% 删除的部分
        set(gca,'FontSize',13,'FontName','Times New Roman');
    title(' Artifacts by decomposition', 'FontSize',18,...
'FontName','Times New Roman'...
)
legend('Artifact','FontSize',13,'FontName','Times New Roman')
ylabel('Voltage(\muV)','FontName','Times New Roman','FontSize',15);
xlabel('Time(s)','FontName','Times New Roman','FontSize',15);
% set(gca,'FontSize',18,'Fontname', 'Times New Roman');
end

%%
which_pusle =floor(0.5 * size(arti_data,1));% 绘制其中的一个波形
%     which_pusle =size(arti_data,1); % 最后一个伪迹
recunst3 = X_residual;
% 进一步识别伪迹范围
dif_re3 = diff(recunst3,1,2);

% 标准差
 change_p_re3 = dif_re3>art_thre * std(abs(dif_re3),0,'all');%% 大于阈值
%
%     which_pusle = 3;
%     change_p_re3 = abs(dif_re3)>art_thre;% 大于阈值

% find_change_re3 = find(change_p_re3);%找到上述逻辑值的索引
[~,re3_col] = find(change_p_re3);
%     during_re3 = [min(re3_col),max(re3_col)];% 找到导数差异最大区间
during_re3 = [min(re3_col),floor(prctile(re3_col,cut_prct))];

% 根据overlay 确定伪迹严重的范围
re_bg = max(1,during_re3(1)-floor(re_overlay*diff(during_re3)));
re_ed = during_re3(2)+floor(re_overlay*diff(during_re3));


%% 伪迹严重的数据区间进行内插运算

sig_select = true(1,size(arti_data,2));
sig_select(re_bg+1:re_ed) = false;
t_int_pre = arti_pulse_t(sig_select);% 创建选择模版

sig_int_post = recunst3; % 排除了伪迹主成分的数据
for i = 1:size(arti_data,1) % 遍历每一个波形
    sig_int_pre = recunst3(i,:);
    sig_int_pre = sig_int_pre(sig_select);
    sig_int_post(i,:) = interp1(t_int_pre,sig_int_pre,arti_pulse_t,'linear');% 替换新的数据
end

if tf_plot
    % 绘图
    figure
    % 原始数据
    subplot(3,2,1)
    plot(arti_pulse_t,X(which_pusle,:),'r')
    hold on
    plot(arti_pulse_t,X_residual(which_pusle,:),'b')
    legend('Pre','Post')
    title('Pre/Post denoise of single pulse', 'FontSize',18,...
'FontName','Times New Roman'...
)

ylabel('Voltage(\muV)','FontName','Times New Roman','FontSize',18);
xlabel('Time(s)','FontName','Times New Roman','FontSize',18);
set(gca,'FontSize',13,'Fontname', 'Times New Roman');
    
    % 主成分
    subplot(3,2,2)
    %         signalscores2 = arti_data*signal_vecs(:,end-numPCs+1:end);% 选用前x个数据进行还原
    %         recunst2 = signalscores2(which_pusle,:)*signal_vecs(:,end-numPCs+1:end)';
    %         plot(arti_pulse_t,recunst2)
    plot(arti_pulse_t,Xprime(which_pusle,:),'g')
        title('SVD result', 'FontSize',18,...
'FontName','Times New Roman'...
)
ylabel('Voltage(\muV)','FontName','Times New Roman','FontSize',18);
xlabel('Time(s)','FontName','Times New Roman','FontSize',18);
set(gca,'FontSize',13,'Fontname', 'Times New Roman');
    
    % 原始-主成分
    subplot(3,2,3)
    plot(arti_pulse_t,X_residual(which_pusle,:))
    title('Signal remain', 'FontSize',18,...
'FontName','Times New Roman'...
)
ylabel('Voltage(\muV)','FontName','Times New Roman','FontSize',18);
xlabel('Time(s)','FontName','Times New Roman','FontSize',18);
set(gca,'FontSize',13,'Fontname', 'Times New Roman');
    hold on
    % 绘制舍弃区段的分界线，蓝色虚线
    plot([arti_pulse_t(re_bg),arti_pulse_t(re_bg)],[max(art_mean),min(art_mean)],'--b');
    plot([arti_pulse_t(re_ed),arti_pulse_t(re_ed)],[max(art_mean),min(art_mean)],'--b');
    
    % 剩余信号的导数（原来是这个代码么？）
    subplot(3,2,4)
    plot(arti_pulse_t(1,1:end-1),dif_re3);
    hold on
    plot([arti_pulse_t(re_bg),arti_pulse_t(re_bg)],[max(art_mean),min(art_mean)],'--b');
    plot([arti_pulse_t(re_ed),arti_pulse_t(re_ed)],[max(art_mean),min(art_mean)],'--b');
    title('Signal remain differential', 'FontSize',18,...
'FontName','Times New Roman'...
)
ylabel('Voltage(\muV)','FontName','Times New Roman','FontSize',18);
xlabel('Time(s)','FontName','Times New Roman','FontSize',18);
set(gca,'FontSize',13,'Fontname', 'Times New Roman');
    
    % 绘制新的数据
    % 绘制其中的一个波形补全后的图形
    subplot(3,2,6)
    plot(arti_pulse_t,sig_int_post(which_pusle,:))
        title('Single pulse after interpolation', 'FontSize',18,...
'FontName','Times New Roman'...
)
ylabel('Voltage(\muV)','FontName','Times New Roman','FontSize',18);
xlabel('Time(s)','FontName','Times New Roman','FontSize',18);
set(gca,'FontSize',13,'Fontname', 'Times New Roman');    
    subplot(3,2,5)
    plot(arti_pulse_t,recunst3)
    hold on
    plot([arti_pulse_t(re_bg),arti_pulse_t(re_bg)],[max(art_mean),min(art_mean)],'--b');
    plot([arti_pulse_t(re_ed),arti_pulse_t(re_ed)],[max(art_mean),min(art_mean)],'--b');
    title('All pulses overlap before interpolation', 'FontSize',18,...
'FontName','Times New Roman'...
)
ylabel('Voltage(\muV)','FontName','Times New Roman','FontSize',18);
xlabel('Time(s)','FontName','Times New Roman','FontSize',18);
set(gca,'Fontname', 'Times New Roman','FontSize',13);

end % 绘图

%% 把 arti_data 经过伪迹处理后的数据放回原始数据
%     data_wave = sig_int_post; % 不做漂移矫正
data_wave = bsxfun(@minus,sig_int_post,mean(sig_int_post,1)); % 做平均矫正
%     data_wave = data_wave-repmat(mean(data_wave,2),1,size(data_wave,2));
data_after_pca = data_roi_origin';
for art_pulse_i = 1:(pusle_count-1)
    % 要替换的原始数据中数据索引
    bg_p = cut_data_bg_p(art_pulse_i);
    ed_p = min(cut_data_bg_p(art_pulse_i+1),size(data_roi_origin,1));% 防止超出索引
    % 原始数据中的bg_p-ed_p这段数据，被替换为降噪修整后的数据
    data_after_pca(bg_p:ed_p) = data_wave(art_pulse_i,1:ed_p-bg_p+1);
end % 伪迹脉冲的个数遍历，填充数据，得到所有伪迹的脉冲堆叠

% 按照变化点切割包含伪迹的区域
bg_p = cut_data_bg_p(pusle_count);
ed_p = min(cut_data_bg_p(pusle_count)+floor(0.5*gap)-100,size(data_roi_origin,1));
% 处理最后一个数据段，防止超出索引
% 原始数据中的bg_p-ed_p这段数据，被替换为降噪修整后的数据
data_after_pca(bg_p:ed_p) = data_wave(art_pulse_i,1:ed_p-bg_p+1);

if tf_plot
    figure
    plot(t_ROI,data_roi_origin,'r');% 原始
    
    hold on
    plot(t_ROI,data_after_pca,'b'); % 绘制滤波后的图像
    %         data_after_pca = data_after_pca';
    legend('Origin signal','After SVD denoise')
        title('Pre/Post Denoise', 'FontSize',13,...
'FontName','Times New Roman'...
)
ylabel('Voltage(\muV)','FontName','Times New Roman');
xlabel('Time(s)','FontName','Times New Roman');
set(gca,'FontSize',18,'Fontname', 'Times New Roman');
end

% 用于输出图像
show_channel_denoise = []; % 输出图像用
show_channel_denoise.ori = X; % 原始数据
show_channel_denoise.remain = X_residual;% 剩余数据
show_channel_denoise.noise = Xprime;% 噪声
show_channel_denoise.t = t;% 时间轴
show_channel_denoise.art_mean = art_mean; % 平均伪迹
show_channel_denoise.bg = re_bg; % 伪迹删除起止点
show_channel_denoise.ed = re_ed;
show_channel_denoise.sig_int_post =sig_int_post; %插值后的数据
show_channel_denoise.recunst3 = recunst3;

% end
%% 时频图绘制
dataout.trialinfo = [1;2];
dataout.trial{1,1} = data_roi_origin';
dataout.trial{1,2} = data_after_pca;
dataout.time{1,1} = t_ROI;
dataout.time{1,2} = t_ROI;
dataout.label{1,1} = 'ch1';
dataout.fsample = 30000;

rejectvisual = false;
% 手动查看噪声，选择试次
cfg = [];                       % 初始化配置文件
cfg.demean = 'yes';
cfg.continuous     = 'no';      % 非连续数据
data = ft_preprocessing(cfg,dataout); % 预处理函数
% cfg = [];
% %             cfg.latency = [-2 2];
% data = ft_selectdata(cfg,data);
% 查看数据
cfg = [];
cfg.method = 'summary';
cfg.metric = 'var';%'var'       variance within each channel (default)
%                       'min'       minimum value in each channel
%                       'max'       maximum value in each channel
%                       'maxabs'    maximum absolute value in each channel
%                       'range'     range from min to max in each channel
%                       'kurtosis'  kurtosis, i.e. measure of peakedness of the amplitude distribution
%                       'zvalue'    mean and std computed over all time and trials, per channel
cfg.trials = 'all';
cfg.channel = 'all';
cfg.keeptrials = 'no';
cfg.feedback = 'no';
cfg.winlength = 50;
cfg.winstep = 25;
if rejectvisual
    data_clean_visual= ft_rejectvisual(cfg, data);
    ft_databrowser(cfg,data_clean_visual); % 查看数据
else
    data_clean_visual = data;
end
%% 时频图
%降采样
cfg = [];
% cfg.lpfilter       = 'yes';     % 低通滤波
% cfg.lpfreq         = app.LowPassfreqEditField.Value;% 截止频率
cfg.resamplefs = 500; % 重采样
cfg.detrend = 'yes'; % 去漂移
cfg.demean = 'yes'; % 去DC
cfg.trials = 'all'; % 1xN 选择试次
data = ft_resampledata(cfg,data_clean_visual); % 得到干净数据
%
cfg = [];
cfg.trials         = 2;
% cfg.trials         = xx;
cfg.output         = 'pow';
cfg.channel        = 'all';
cfg.method         = 'mtmconvol';
cfg.taper          = 'hanning';
cfg.foi            = 2:1:160;
cfg.t_ftimwin      = ones(length(cfg.foi),1).*0.5;
cfg.toi            = -3:0.01:3;
cfg.pad            = 'nextpow2';

% TFRhann            = ft_freqanalysis(cfg, data2);
TFRhann            = ft_freqanalysis(cfg, data);
% 时频图
cfg = [];
% cfg.baseline = [-1.5,-0.05];
% cfg.baselinetype = 'zscore';
% cfg.zlim = [-1.5e-27 1.5e-27];
% cfg.channel = 'bla5-2'; % top figure

% 时频图绘制
figure;
%  ft_singleplotER(cfg,TFRhann);
ft_singleplotTFR(cfg, TFRhann);

caxis([-1,1000])
% caxis([-3,3])
% xlim([-3,3])
%% 功率谱绘制
n = numel(t_ROI);
srate = 30000;
hz    = linspace(0,srate/2,floor(n/2));
sigA = data_roi_origin';
sigB = data_after_pca;
sigAx = fft(sigA)/n;
sigBx = fft(sigB)/n;
specX = abs(sigAx.*conj(sigBx)).^2;
figure(7), clf
plot(hz,abs(sigAx(1:length(hz))).*2,'r')
hold on
plot(hz,abs(sigBx(1:length(hz))).*2,'b')
set(gca,'xlim',[2 160],'FontSize',18,'FontName','Times New Roman')
xlabel('Frequency (Hz)','FontSize',20,'FontName','Times New Roman')
ylabel('Power(\muW^2)','FontSize',20,'FontName','Times New Roman')
title('Power spectrum compare','FontSize',23,'FontName','Times New Roman')
legend({'Origin signal','After SVD denoise'},'FontSize',15,'FontName','Times New Roman')