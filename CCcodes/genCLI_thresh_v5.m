clear
format long;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Percentiles to be calculated%%%%%%%%%%%
PRCP_thresh = [20, 25, 33.3, 40, 50, 60, 66.6, 75, 80, 90, 95, 99, 99.9];
TAIR_thresh = [10 25 50 75 90];

%%%%%%%%%%%%%%number of years in climatology%%%%%%%%%%%%
nc=30;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%load mesonet station data
non_temporal=readtable('site.csv',"VariableNamingRule","preserve","TreatAsMissing",'NaN');
sites=string(non_temporal{:,2});
nsM=numel(sites);
latM=non_temporal{:,3};
lonM=non_temporal{:,4};
elvM=non_temporal{:,5}.*0.3048;

%%%%%%%%%%%%%%%%load NLDAS2 data bilinearly interpolated to mesonet sites%%%%%%%%%%%%%%
dirin1='/Users/erappin/Documents/Mesonet/ClimateIndices/cliSITES/';
for i=1:nsM
    dirin=strcat(dirin1,strcat(sites(i),'/'));
    filein_year=strcat(dirin,strcat('1980',strcat('_',(strcat(sites(i),"_hourly")))));
    load(filein_year);
    TT_hourlyF=TT_hourly;
    clear TT_hourly;
    for j=1:nc
          m=i+1;
          YEAR=string(1980+j);
          filein_year=strcat(dirin,strcat(YEAR,strcat('_',(strcat(sites(i),"_hourly")))));
          load(filein_year);
          TT_hourlyF=[TT_hourlyF;TT_hourly];
          clear TT_hourly;
    end
    
    [nt,nv]=size(TT_hourlyF);
    TT_daily=retime(TT_hourlyF,'daily','mean');
    TT_tmpX=retime(TT_hourlyF,'daily','max');
    TT_tmpN=retime(TT_hourlyF,'daily','min');
    TT_tmpP=retime(TT_hourlyF(2:nt,:),'regular','sum','TimeStep',minutes(1440),'IncludedEdge','right');
    
    TT_daily=removevars(TT_daily,'PRCP');
    TT_daily=addvars(TT_daily,TT_tmpX.TAIR,'After','TAIR','NewVariableNames','TAIRx');
    TT_daily=addvars(TT_daily,TT_tmpN.TAIR,'After','TAIRx','NewVariableNames','TAIRn');
    TT_daily=addvars(TT_daily,TT_tmpP.PRCP,'Before','PRES','NewVariableNames','PRCP');
    
    TIME=TT_daily.TIMESTAMP;
    [y,m,d]=ymd(TIME);
    TT_daily(y==double(YEAR),:)=[];
    
    isLeapDay = month(TIME)==2 & day(TIME)==29; 
    TT_daily(isLeapDay,:) = [];  % remove leap days. 

    PRCP_ib1(i,:,:) = TT_daily.PRCP;
    TAIR_ib1(i,:,:) = TT_daily.TAIR;
    TAIRx_ib1(i,:,:) = TT_daily.TAIRx;
    TAIRn_ib1(i,:,:) = TT_daily.TAIRn;

    %%%%%%%%%%%%%%%%%%calculate out of base and in base data sets%%%%%%%%%%
    for j=1:nc
        PRCP_ob(i,:) = PRCP_ib1(i,j,:);
        TAIR_ob(i,:) = TAIR_ib1(i,j,:);
        TAIRx_ob(i,:) = TAIRx_ib1(i,j,:);
        TAIRn_ob(i,:) = TAIRn_ib1(i,j,:);
    
        valid_vals = setdiff(1:30, j);
        result = valid_vals( randi(length(valid_vals), 1, 1) );
        PRCP_ib1(i,j,:) =PRCP_ib1(i,result,:);
        TAIR_ib1(i,j,:)=TAIR_ib1(i,result,:);
        TAIRx_ib1(i,j,:)=TAIRx_ib1(i,result,:);
        TAIRn_ib1(i,j,:)=TAIRn_ib1(i,result,:);
       
        %%%%%%%%%%%%%%%%%%%%apply rlowess smoother%%%%%%%%%%%%%%%%%
        PRCP_ib5(i,j,:) = smoothdata(PRCP_ib1(i,j,:),3,"rlowess",5);
        TAIR_ib5(i,j,:) = smoothdata(TAIR_ib1(i,j,:),3,"rlowess",5);
        TAIRx_ib5(i,j,:) = smoothdata(TAIRx_ib1(i,j,:),3,"rlowess",5);
        TAIRn_ib5(i,j,:) = smoothdata(TAIRn_ib1(i,j,:),3,"rlowess",5);
        PRCP_ib25(i,j,:) = smoothdata(PRCP_ib1(i,j,:),3,"rlowess",25);
        TAIR_ib25(i,j,:) = smoothdata(TAIR_ib1(i,j,:),3,"rlowess",25);
        TAIRx_ib25(i,j,:) = smoothdata(TAIRx_ib1(i,j,:),3,"rlowess",25);
        TAIRn_ib25(i,j,:) = smoothdata(TAIRn_ib1(i,j,:),3,"rlowess",25);
    end
       
    %%%%%%%%%calculate percentiles for in base years
    for j=1:nc
        PRCP_perc1(i,j,:,:)=prctile(PRCP_ib1(i,:,:),PRCP_thresh,2);
        TAIR_perc1(i,j,:,:)=prctile(TAIR_ib1(i,:,:),TAIR_thresh,2);
        TAIRx_perc1(i,j,:,:)=prctile(TAIRx_ib1(i,:,:),TAIR_thresh,2);
        TAIRn_perc1(i,j,:,:)=prctile(TAIRn_ib1(i,:,:),TAIR_thresh,2);
        PRCP_perc5(i,j,:,:)=prctile(PRCP_ib5(i,:,:),PRCP_thresh,2);
        TAIR_perc5(i,j,:,:)=prctile(TAIR_ib5(i,:,:),TAIR_thresh,2);
        TAIRx_perc5(i,j,:,:)=prctile(TAIRx_ib5(i,:,:),TAIR_thresh,2);
        TAIRn_perc5(i,j,:,:)=prctile(TAIRn_ib5(i,:,:),TAIR_thresh,2);
        PRCP_perc25(i,j,:,:)=prctile(PRCP_ib25(i,:,:),PRCP_thresh,2);
        TAIR_perc25(i,j,:,:)=prctile(TAIR_ib25(i,:,:),TAIR_thresh,2);
        TAIRx_perc25(i,j,:,:)=prctile(TAIRx_ib25(i,:,:),TAIR_thresh,2);
        TAIRn_perc25(i,j,:,:)=prctile(TAIRn_ib25(i,:,:),TAIR_thresh,2);
    end
    
    PRCP_meanpercs1(:,:)=mean(PRCP_perc1(i,:,:,:),2);
    TAIR_meanpercs1(:,:)=mean(TAIR_perc1(i,:,:,:),2);
    TAIRx_meanpercs1(:,:)=mean(TAIRx_perc1(i,:,:,:),2);
    TAIRn_meanpercs1(:,:)=mean(TAIRn_perc1(i,:,:,:),2);
    PRCP_meanpercs5(:,:)=mean(PRCP_perc5(i,:,:,:),2);
    TAIR_meanpercs5(:,:)=mean(TAIR_perc5(i,:,:,:),2);
    TAIRx_meanpercs5(:,:)=mean(TAIRx_perc5(i,:,:,:),2);
    TAIRn_meanpercs5(:,:)=mean(TAIRn_perc5(i,:,:,:),2);
    PRCP_meanpercs25(:,:)=mean(PRCP_perc25(i,:,:,:),2);
    TAIR_meanpercs25(:,:)=mean(TAIR_perc25(i,:,:,:),2);
    TAIRx_meanpercs25(:,:)=mean(TAIRx_perc25(i,:,:,:),2);
    TAIRn_meanpercs25(:,:)=mean(TAIRn_perc25(i,:,:,:),2);
      
    dirout=strcat(strcat(dirin1,sites(i)),'/');
    fileout=strcat(dirout,strcat(sites(i),"_CLIthresh_daily"));
    
    save(fileout,"sites","nsM","latM","lonM","elvM","PRCP_meanpercs1","TAIR_meanpercs1","TAIRx_meanpercs1", ...
    "TAIRn_meanpercs1","PRCP_meanpercs5","TAIR_meanpercs5","TAIRx_meanpercs5","TAIRn_meanpercs5", ...
    "PRCP_meanpercs25","TAIR_meanpercs25","TAIRx_meanpercs25","TAIRn_meanpercs25", ...
    "PRCP_thresh","TAIR_thresh","TT_hourlyF","TT_daily");
end


