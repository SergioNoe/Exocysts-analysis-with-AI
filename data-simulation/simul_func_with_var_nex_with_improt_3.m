clear all
close all
clc
% 
% from discussion with Oriol and Marta
%- Patch petit (50-60 nm de diàmetre? no estem segurs de la mida, caldria discutir-ho): exocysts unint la part central inferior de la vesícula amb la membrana
%- Patch de 100 nm de diàmetre: exocysts units a tota la vesícula
%- Patch de 300 nm de diàmetre: exocysts repartits en tot el "site" que agafem
%- Anells: exocyst estructurats al voltant de la vesícula en forma d'anell (el que ja havíem parlat). Dins d'aquest, podríem trobar
% diverses estructures compatibles amb diversos mecanismes:
%        - Random: els exocyst es situen de manera random al voltant de la vesícula
%        - 1 zipper: punt de nucleació que va creixent (s'extén per un o pels dos costats de la nucleació i forma, per exemple, arcs o línies corbes)

%show=0;
cur=cd;
%datapath = [cur,'/images/second'];
datapath = [cur,'/try'];
mat_filename = 'try.mat';

imsize=128;%112; %pixels
pixelsize=3;%3; %nm

%locs=[0:100];
%x=geopdf(locs,.125)
lab_prob=0.5; %labeling probability
geopar=.125; %to get average of 8 locs per mol
%plot(locs,x)

nprot_per_ex=3; % number of proteins per exocyst
%nex=12; % number of exocysts per site (fixed)
min_d=1; % minimum distance between exocysts (nm)

nsims=1000; % number of simulations to run
M=[];% i:simul_index j:exocyst_index x:x_exo y:y_exo I_GFP:intensity_of_GFP k:protein_index x_pro y_pro l:localization_index x_locs y_locs

err_pro= 10; % std distance of protein from exocyst center (nm)
loc_prec=20/sqrt(2); %(nm)

geometries={'spot_r30';'spot_r60';'spot_r120';'ring_unif';'ring_zip'};
radius=[30,60,120,45,45];

maps=zeros(imsize,imsize,100);
locs_raw=zeros(imsize,imsize,100);
locs=zeros(imsize,imsize,100);
im_p=zeros(imsize,imsize,100);
label=zeros(100,1); % python based, 0-4
nexs=zeros(100,1); % 
Max_locs_raw=0;
Max_im_p=0;
save(fullfile(datapath,mat_filename),'locs_raw','locs','maps','im_p','label','nexs','Max_locs_raw','Max_im_p','-v7.3');
mf = matfile(fullfile(datapath,mat_filename),'Writable',true);

sigma_conv=(488/(2*1.49))/(2*sqrt(2*log(2))); % we can get the experimental one from Uri

for i=1:nsims

struc=random('unid',5);        
%here we fix the struc
%struc=3;
geometry=geometries{struc};

%but vary the number of exocysts per site
nex=random('unid',20)+4; % between 4 and 24
%nex=random('unid',13)+7; % between 8 and 20
%nex=random('unid',9)+7; % between 8 and 16

im_disk=zeros(imsize,imsize);
im_disk(imsize/2,imsize/2)=1;

switch geometry
    case {'ring_unif','ring_zip'}
        r_ring= radius(struc); % ring radius (nm)
        err_ring= 5; % std of the ring (nm)
        k1=imfilter(im_disk,fspecial('disk',(r_ring+3*err_ring)/pixelsize));
        k2=imfilter(im_disk,fspecial('disk',(r_ring-3*err_ring)/pixelsize));
        im_disk(imsize/2,imsize/2)=0;
        im_disk=im_disk+k1-k2;
        im_disk(find(im_disk<0))=0;
        im_disk(find(im_disk>0))=1;
    otherwise %case  'spot'
        r_spot= radius(struc); % ring radius (nm) % 30 - 60 - 120
        k1=imfilter(im_disk,fspecial('disk',(r_spot)/pixelsize));
        im_disk(imsize/2,imsize/2)=0;
        im_disk=im_disk+k1;
        im_disk(find(im_disk<0))=0;
        im_disk(find(im_disk>0))=1;
end

%imagesc(im_disk)

    n(i)=0; %localization per site
    
    switch geometry
        case 'ring_unif'
            th=random('unif',0,2*pi,nex,1);
            rr=random('norm',0,err_ring,nex,1);
            x_e=(r_ring+rr).*cos(th);
            y_e=(r_ring+rr).*sin(th);
        case 'ring_zip'
            th0=random('unif',0,2*pi,1,1);
            arcs=random('norm',min_d,min_d/5,nex-1,1)*(2*pi/r_ring);
            th=[th0; th0+cumsum(arcs)];
            rr=random('norm',0,err_ring,nex,1);
            x_e=(r_ring+rr).*cos(th);
            y_e=(r_ring+rr).*sin(th);
        otherwise %case 'spot'
            th=random('unif',0,2*pi,nex,1);
            rr = r_spot*sqrt(random('unif',0,1,nex,1));
            x_e=(rr).*cos(th);
            y_e=(rr).*sin(th);
            
    end
    
    
    
    for j=1:nex % for each exocyst, get its x and y pos (along a ring structure)
        x=x_e(j);
        y=y_e(j);
        I_GFP=random('unif',0,255); % get GFP intensity, if above a detection threshold (effect of noise)
        if I_GFP<80
            I_GFP=0;
        end
        
        for k=1:nprot_per_ex % for each protein per exocyst, get its x and y pos respect to corresponding exocyst
            x_pro=x+random('norm',0,err_pro);
            y_pro=y+random('norm',0,err_pro);
            if rand>lab_prob % include finite labeling probability, not all proteins emit
                nn=geornd(geopar,1); % this is the number of localizations per protein
                n(i)=n(i)+nn;
                for l =1:nn
                    x_locs=x_pro+random('norm',0,loc_prec);
                    y_locs=y_pro+random('norm',0,loc_prec);
                    M=[M; i j x y I_GFP k x_pro y_pro l x_locs y_locs];
                end
            else %nn=0;
                M=[M; i j x y I_GFP k x_pro y_pro NaN NaN NaN];
            end
        end
    end
    
    
    %create image locs
    i_this=find(M(:,1)==i); % index corresponding to this simulation
    m=M(i_this,:); % subset to this simulation
    [j_this,ia]=unique(m(:,2)); % eliminate multiple entries for GFP
    xGFP_m=sum(m(ia,3).*m(ia,5))/sum(m(ia,5)); % center of mass of GFP
    yGFP_m=sum(m(ia,4).*m(ia,5))/sum(m(ia,5)); % center of mass of GFP
    
%     if show==1
%         figure(1)
%         subplot(2,3,1)
%         imagesc(im_disk)
%         colormap(gray)
%         hold on
%         plot(m(ia,3)/pixelsize+imsize/2,m(ia,4)/pixelsize+imsize/2,'go','MarkerFaceColor','g')
%         plot(xGFP_m/pixelsize+imsize/2,yGFP_m/pixelsize+imsize/2,'ms','MarkerFaceColor','m')
%         set(gca,'YDir','normal')
%         axis image
%         xlim([0 imsize])
%         ylim([0 imsize])
%         title(['Original f.o.r.' newline 'map + exocysts/GFPs + apparent center'])
%         hold off
%     end
    
    
    im_GFP=sparse(round(m(ia,4)/pixelsize+imsize/2+1),round(m(ia,3)/pixelsize+imsize/2+1),m(ia,5)/max(m(ia,5)),imsize,imsize);
    im_GFP=imgaussfilt(full(im_GFP),sigma_conv/pixelsize); % get GFP conventional image
    
%     if show==1
%         subplot(2,3,2)
%         imagesc(im_GFP)
%         hold on
%         plot(xGFP_m/pixelsize+imsize/2,yGFP_m/pixelsize+imsize/2,'ms','MarkerFaceColor','m')
%         yline(imsize/2)
%         xline(imsize/2)
%         set(gca,'YDir','normal')
%         axis image
%         title(['Original f.o.r.' newline 'GFP image + apparent center'])
%         hold off
%     end
     
    % center with respect to GFP center of mass
    M(i_this, [3,7,10]) = M(i_this, [3,7,10]) -xGFP_m;
    M(i_this, [4,8,11]) = M(i_this, [4,8,11]) -yGFP_m;
    
    clear j_this ia m
    m=M(i_this,:); %repet subsetting
    [j_this,ia]=unique(m(:,2)); % to get index ia (elements without repetition)
    
    
%     if show==1
%         subplot(2,3,3)
%         plot(m(ia,3)/pixelsize,m(ia,4)/pixelsize,'go','MarkerFaceColor','g')
%         axis image
%         xlim([-imsize/2 imsize/2])
%         ylim([-imsize/2 imsize/2])
%         title(['GFP-centered f.o.r.' newline 'exocyst/GFP positions'])
%         hold off
%         
%         subplot(2,3,4)
%         plot(m(:,10)/pixelsize,m(:,11)/pixelsize,'r+');%,'MarkerFaceColor','')
%         axis image
%         xlim([-imsize/2 imsize/2])
%         ylim([-imsize/2 imsize/2])
%         title(['GFP-centered f.o.r.' newline 'localization positions'])
%         hold off
%     end
    
    isnum=~isnan(m(:,10)); % only "appearing" molecules
    %im_mMaple=sparse(round(m(isnum,11)/pixelsize+imsize/2+1),round(m(isnum,10)/pixelsize+imsize/2+1),1,imsize,imsize);
    %im_mMaple=imgaussfilt(full(im_mMaple),loc_prec/(1.75*pixelsize)); %superresolution image
    
    % use a larger image to prevent cutting localizations sitting outside
    IM_SIZE=imsize+imsize/2;
    im_mMaple0=full(sparse(round(m(isnum,11)/pixelsize+IM_SIZE/2+1),round(m(isnum,10)/pixelsize+IM_SIZE/2+1),1,IM_SIZE,IM_SIZE));
    im_mMaple=imgaussfilt(im_mMaple0,loc_prec/(1.75*pixelsize)); %superresolution image
    im_mMaple0=im_mMaple0(imsize/4+1:end-imsize/4,imsize/4+1:end-imsize/4);
    im_mMaple=im_mMaple(imsize/4+1:end-imsize/4,imsize/4+1:end-imsize/4);
        
    
%     if show==1
%         subplot(2,3,5)
%         imagesc(im_mMaple)
%         set(gca,'YDir','normal')
%         axis image
%         colormap(hot)
%         title(['GFP-centered f.o.r.' newline 'rendered SMLM'])
%         hold off
%     end
    
    % recalculate the disk with respect to the GFP center
    im_disk_shift=zeros(imsize,imsize);
    im_disk_shift(-round(yGFP_m/pixelsize)+imsize/2,-round(xGFP_m/pixelsize)+imsize/2)=1;
    switch geometry
        case {'ring_unif','ring_zip'}
            k1=imfilter(im_disk_shift,fspecial('disk',(r_ring+3*err_ring)/pixelsize));
            k2=imfilter(im_disk_shift,fspecial('disk',(r_ring-3*err_ring)/pixelsize));
            im_disk_shift=im_disk_shift+k1-k2;
            im_disk_shift(-round(yGFP_m/pixelsize)+imsize/2,-round(xGFP_m/pixelsize)+imsize/2)=0;
            im_disk_shift(find(im_disk_shift<0))=0;
            im_disk_shift(find(im_disk_shift>0))=1;
        otherwise %case 'spot'
            k1=imfilter(im_disk_shift,fspecial('disk',(r_spot)/pixelsize));
            im_disk_shift=im_disk_shift+k1;
            im_disk_shift(-round(yGFP_m/pixelsize)+imsize/2,-round(xGFP_m/pixelsize)+imsize/2)=0;
            im_disk_shift(find(im_disk_shift<0))=0;
            im_disk_shift(find(im_disk_shift>0))=1;
    end
    
    
    
    
%     if show==1
%         subplot(2,3,6)
%         imagesc(im_disk_shift)
%         hold on
%         plot(m(:,10)/pixelsize+imsize/2,m(:,11)/pixelsize+imsize/2,'r+');
%         set(gca,'YDir','normal')
%         axis image
%         colormap(hot)
%         hold off
%         title(['GFP-centered f.o.r.' newline 'shifted map'])
%         %pause
%     end
    

    
    %calculate protein actual map
         m2=m;%(isnum,:);
         clear j_this ia
         [j_this,ia]=unique(m2(:,2)); % to get index ia (elements without repetition)
         
         rows = round(m2(ia,8)/pixelsize+imsize/2+1); 
         cols = round(m2(ia,7)/pixelsize+imsize/2+1);
         iwithin=find(m2(ia,8)/pixelsize+imsize/2+1<imsize & m2(ia,7)/pixelsize+imsize/2+1<imsize);
         rows=rows(iwithin);
         cols=cols(iwithin);
%          rows = round(m2(ia,8)/pixelsize+imsize/2+1);
%          for i = 1:size(rows, 1)
%              if rows(i) > imsize
%                  rows(i) = imsize;
%              end
%          end
%          
%          cols = round(m2(ia,7)/pixelsize+imsize/2+1);
%          for i = 1:size(cols, 1)
%              if cols(i) > imsize
%                  cols(i) = imsize;
%              end
%          end
         
         im_prot=sparse(rows,cols,1,imsize,imsize);
         
%         im_prot=sparse(round(m2(ia,8)/pixelsize+imsize/2+1),round(m2(ia,7)/pixelsize+imsize/2+1),1,imsize,imsize);
         im_prot=full(im_prot);
         im_prot=imgaussfilt(full(im_prot),loc_prec/(5*pixelsize)); %superresolution image

         
    %         figure(100)
    %plot(nex,sum(im_mMaple(:))/8,'.')
    %hold on
    %plot(nex,sum(im_prot(:)),'+r')
    
    %     pause
         
    %
    %     if show==1
    %         subplot(2,3,6)
    %         imagesc(im_prot)
    %         hold on
    %         plot(m2(ia,7)/pixelsize+imsize/2,m2(ia,8)/pixelsize+imsize/2,'k+');
    %         set(gca,'YDir','normal')
    %         axis image
    %         colormap(hot)
    %         hold off
    %         title(['GFP-centered f.o.r.' newline 'rendered protein positions'])
    %         pause
    %     end
    %
    %locs=;
    %locs=(locs-min(locs(:)))./(max(locs(:))-min(locs(:)));
    %locs(:,:)=log(1+ (im_mMaple));
    %locs=(locs-min(locs(:)))./(max(locs(:))-min(locs(:)));
    %locs=(locs-mean(locs(:)))./(std(locs(:)));
    %locs=(locs-min(locs(:)))./(max(locs(:))-min(locs(:)));
    
    
    
    % progressively save images
    if rem(i,100)==0
        maps(:,:,100)=im_disk_shift;
        im_p(:,:,100)=im_prot;
        locs_raw(:,:,100)= im_mMaple0;
        locs(:,:,100)= im_mMaple;
        label(100,1)=struc-1;
        nexs(100,1)=nex;
        
        Max_locs_raw=max([Max_locs_raw, max(locs(:))]);
        Max_im_p=max([Max_im_p, max(im_p(:))]);
        
        if i==100
            mf.maps(:,:,1:100)=maps;
            mf.im_p(:,:,1:100)=im_p;
            mf.locs_raw(:,:,1:100)=locs_raw;
            mf.locs(:,:,1:100)=locs;
            mf.label(1:100,1)=label;
            mf.nexs(1:100,1)=nexs;
        else
            mf.maps(:,:,end+1:end+100)=maps;
            mf.im_p(:,:,end+1:end+100)=im_p;
            mf.locs_raw(:,:,end+1:end+100)=locs_raw;
            mf.locs(:,:,end+1:end+100)=locs;
            mf.label(end+1:end+100,1)=label;
            mf.nexs(end+1:end+100,1)=nexs;
        end
        maps=zeros(imsize,imsize,100);
        im_p=zeros(imsize,imsize,100);
        locs_raw=zeros(imsize,imsize,100);
        locs=zeros(imsize,imsize,100);
        label=zeros(100,1);
        nexs=zeros(100,1);
    else
        maps(:,:,rem(i,100))=im_disk_shift;
        im_p(:,:,rem(i,100))=im_prot;
        locs_raw(:,:,rem(i,100))= im_mMaple0;
        locs(:,:,rem(i,100))= im_mMaple;
        label(rem(i,100),1)=struc-1;
        nexs(rem(i,100),1)=nex;
    end
    clear m m2 xGFP_m yGFP_m im_mMaple im_mMaple0 im_GFP ia jthis im_prot struc

end
mf.Max_locs_raw=Max_locs_raw;
mf.Max_im_p=Max_im_p;


%save(fullfile(datapath,mat_filename),'locs','maps','-v7.3');
%%
%if show==1
%figure, montage(locs)
%colormap(hot)


%iok=find(n>0);
%figure,
%histogram(n(iok),max(n(iok)))
%end
