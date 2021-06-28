clear all
close all
clc
% test dataset
%load('Test_dict_1chan_2chan_numbers_2.mat')
load('D:/Sergio/Desktop/TFM/Uri_sergio_tests/Number of exocists/U-NET_2/2 channels/1chan_2chan/Files/Test_dict_1chan_2chan_numbers_7_part3.mat')
% parameters of training dataset
%load('Training_dict_1chan_2chan_numbers_2.mat', 'mean_imps', 'std_imps')
load('D:/Sergio/Desktop/TFM/Uri_sergio_tests/Number of exocists/U-NET_2/2 channels/1chan_2chan/Files/Training_dict_1chan_2chan_numbers_7_part3.mat', 'mean_imps', 'std_imps')
%%
nexs=nexs'; % ground truth numb of proteins
shapes=shapes'; % ground truth classes (0 based)

Pred_map = (Pred(:, :, :, 2) +1) /2; % predicted map
Pred = (Pred(:, :, :, 1) * std_imps) + mean_imps; % predicted image of proteins

Pred_nexs=(Pred_nexs(:,1)+1) * (0.5 * (max(nexs) - min(nexs))) + min(nexs); % predicted numb of proteins

Predt= squeeze(sum(sum(Pred(:,:,:,1), 2),3)); % predicted numb of proteins from images
Origt= squeeze(sum(sum(locs(:,:,:), 2),3)); % total number of localizations per image

%Pred_shape
%% number of exocysts
figure(1)
subplot(2,2,1)
plot(nexs,Pred_nexs,'bo')
xlabel('true number of proteins per image')
ylabel('predicted number of proteins per image')


subplot(2,2,2)
plot(nexs,Predt,'g^')
xlabel('true number of proteins per image')
ylabel('predicted number of proteins per image')


MAE = mean(abs(Pred_nexs-nexs))
RMSE = sqrt(mean((Pred_nexs-nexs).^2))

nexx=unique(nexs');

for j=1:length(nexx)
    i_this=find(nexs==nexx(j));
    MAE_i(j)=mean(abs(Pred_nexs(i_this)-nexx(j)));
    RMSE_i(j)=sqrt(mean((Pred_nexs(i_this)-nexx(j)).^2));
    mean_pred(j)=mean(Pred_nexs(i_this)); % predicted val
    std_pred(j)=std(Pred_nexs(i_this));
    mean_pred_im(j)=mean(Predt(i_this)); % predicted val from image
    std_pred_im(j)=std(Predt(i_this));
    mean_locs_est(j)=mean(Origt(i_this)/12); %estimated val
    std_locs_est(j)=std(Origt(i_this)/12);
    
    clear i_this
end

subplot(2,2,3)
errorbar(nexx, mean_pred, std_pred,'o-b')
hold on
errorbar(nexx, mean_locs_est, std_locs_est,'s-r')
errorbar(nexx, mean_pred_im, std_pred_im,'^-g')
plot([5:24],[5:24],'k-.','linewidth',2)
xlabel('true number of proteins per image')
ylabel('average of predicted number of proteins per image')
legend({'Net'; 'estimation'; 'prediction from images'},'Location','NW')


subplot(2,2,4)
plot(nexx, MAE_i,'o-b', nexx, RMSE_i,'s-r')
hold on
plot(nexx, MAE*ones(size(nexx)),':b', nexx, RMSE*ones(size(nexx)),':r', 'linewidth',2)
xlabel('number of proteins per image')
ylabel('metrics')
legend({'MAE'; 'RMSE'; 'average MAE'; 'average RMSE'},'Location','NW')

%% shape classification
figure(2)

%models={'1'; '2'; '3'; '4'; '5'};
models={'spot_r30';'spot_r60';'spot_r120';'ring_unif';'ring_zip'};

[~,pred_shapes]=max(Pred_shapes',[],1);
pred_shapes=pred_shapes-1;

[ micro, macro ] = micro_macro_PR( shapes , pred_shapes);
micro.fscore
D = confusionmat(shapes , pred_shapes);
confusionchart(D,models,'Normalization','row-normalized');

%% image similarity (0 for identical images)
% proteins
mini=min(Pred,[],[2 3]);
P=Pred-repmat(mini,[1,128,128]);
s=sum(sum(P,2),3);
P=P./repmat(s,[1,128,128]);
clear mini s
mini=min(imps,[],[2 3]);
Q=imps-repmat(mini,[1,128,128]);
s=sum(sum(Q,2),3);
Q=Q./repmat(s,[1,128,128]);
clear mini s

% for i=1:size(Pred,1)
%     i
% P(i,:,:)=Pred(i,:,:)+min(Pred(i,:,:));
% P(i,:,:)=P(i,:,:)./sum(P(i,:,:));
% Q(i,:,:)=imps(i,:,:)+min(imps(i,:,:));
% Q(i,:,:)=Q(i,:,:)./sum(Q(i,:,:));
% end

M=0.5*(P+Q);
Dkl_P=zeros(size(P));
iP=find(P~=0);
Dkl_P(iP) = P(iP).*log2(P(iP)./M(iP));
Dkl_Q=zeros(size(Q));
iQ=find(Q~=0);
Dkl_Q(iQ) = Q(iQ).*log2(Q(iQ)./M(iQ));
Shan_imp=sqrt(sum(sum(0.5*(Dkl_P+Dkl_Q),2),3));
%
for j=1:length(nexx)
    i_this=find(nexs==nexx(j));
    S(j)=mean(Shan_imp(i_this));
    sS(j)=std(Shan_imp(i_this));
    clear i_this
end
figure(3)
subplot(1,2,1)
errorbar(nexx,S,sS,'ob')
ylim([0 1])
xlabel('true number of proteins per image')
ylabel('average Jensen-Shannon distance')

legend({'images of proteins'})

%
clear S sS  P Q

% maps
mini=min(Pred_map,[],[2 3]);
P=Pred_map-repmat(mini,[1,128,128]);
s=sum(sum(P,2),3);
P=P./repmat(s,[1,128,128]);
clear mini s
mini=min(maps,[],[2 3]);
Q=maps-repmat(mini,[1,128,128]);
s=sum(sum(Q,2),3);
Q=Q./repmat(s,[1,128,128]);
clear mini s


M=0.5*(P+Q);
Dkl_P=zeros(size(P));
iP=find(P~=0);
Dkl_P(iP) = P(iP).*log2(P(iP)./M(iP));
Dkl_Q=zeros(size(Q));
iQ=find(Q~=0);
Dkl_Q(iQ) = Q(iQ).*log2(Q(iQ)./M(iQ));
Shan_maps=sqrt(sum(sum(0.5*(Dkl_P+Dkl_Q),2),3));
%
for j=1:length(nexx)
    i_this=find(nexs==nexx(j));
    S(j)=mean(Shan_maps(i_this));
    sS(j)=std(Shan_maps(i_this));
    clear i_this
end
subplot(1,2,2)
errorbar(nexx,S,sS,'ob')
ylim([0 1])
xlabel('true number of proteins per image')
ylabel('average Jensen-Shannon distance')
legend({'maps'})


%% show results on 3 random images; nexs

io=random('unid',size(Pred,1),1,1);
figure(4)
%
for i=1:3

subplot(3,3,1+i-1)
imagesc(squeeze(locs(io+i-1,:,:)))
title([  'JSdist =' ,num2str(Shan_imp(io+i-1))])
h(1)=gca;

subplot(3,3,4+i-1)
imagesc(squeeze(imps(io+i-1,:,:)))
title(['# of prot = ', num2str(nexs(io+i-1)) ,' - ',])
h(2)=gca;

subplot(3,3,7+i-1)
imagesc(squeeze(Pred(io+i-1,:,:)))
title(['# of prot = ', num2str(Pred_nexs(io+i-1)) ,' - ',])
h(3)=gca;

linkaxes(h,'xy')
end




%% show results on 3 random images; maps images

geometries={'spot_r30';'spot_r60';'spot_r120';'ring_unif';'ring_zip'};

io=random('unid',size(Pred,1),1,1);
figure(5)
%
for i=1:3

subplot(3,3,1+i-1)
imagesc(squeeze(locs(io+i-1,:,:)))
title([ '# of prot = ', num2str(nexs(io+i-1)) ,' - ', 'JSdist =' ,num2str(Shan_maps(io+i-1))])
h(1)=gca;

subplot(3,3,4+i-1)
imagesc(squeeze(maps(io+i-1,:,:)))
title(geometries(shapes(io+i-1)+1))
h(2)=gca;

subplot(3,3,7+i-1)
imagesc(squeeze(Pred_map(io+i-1,:,:)))
title(geometries(pred_shapes(io+i-1)+1))
h(3)=gca;

linkaxes(h,'xy')
end


